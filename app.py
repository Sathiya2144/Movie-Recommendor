from flask import Flask, request, render_template, jsonify
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import logging
import json
from typing import List, Dict, Any
import re
import asyncio
import threading
import atexit
import sqlite3
from cachetools import TTLCache

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TMDB_API_KEY = '169194264ef0b3a77d5096b48ba82a85'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

FALLBACK_GENRE_MAP = {
    28: 'action', 12: 'adventure', 16: 'animation', 35: 'comedy', 80: 'crime',
    99: 'documentary', 18: 'drama', 10751: 'family', 14: 'fantasy', 36: 'history',
    27: 'horror', 10402: 'music', 9648: 'mystery', 10749: 'romance', 878: 'sci-fi',
    10770: 'tvmovie', 53: 'thriller', 10752: 'war', 37: 'western'
}

app = Flask(__name__)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Server-side cache for TMDB API responses (TTL: 1 hour)
tmdb_cache = TTLCache(maxsize=1000, ttl=3600)

# SQLite database setup
def init_db():
    with sqlite3.connect('reviews.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_type TEXT NOT NULL,
                media_id INTEGER NOT NULL,
                name TEXT,
                rating INTEGER NOT NULL,
                review TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

init_db()

class MovieRecommender:
    def __init__(self):
        self.genre_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
        self.adult_filter = True
        self.genre_map = self._fetch_genre_map('movie')
        self.tv_genre_map = self._fetch_genre_map('tv')
        if not self.genre_map or not self.tv_genre_map:
            logger.warning("Using fallback genre maps due to API fetch failure")
            self.genre_map = self.genre_map or FALLBACK_GENRE_MAP
            self.tv_genre_map = self.tv_genre_map or FALLBACK_GENRE_MAP
        else:
            logger.info("Successfully fetched genre maps from TMDB")
            logger.debug(f"Genre map: {self.genre_map}")
            logger.debug(f"TV Genre map: {self.tv_genre_map}")
        self.MIN_RECOMMENDATIONS = 50
        self._lock = threading.Lock()
        self.offline_mode = not self.genre_map or not self.tv_genre_map

    def _fetch_genre_map(self, media_type: str) -> Dict[int, str]:
        cache_key = f"genre_map_{media_type}"
        if cache_key in tmdb_cache:
            return tmdb_cache[cache_key]
        url = f"{TMDB_BASE_URL}/genre/{media_type}/list"
        params = {'api_key': TMDB_API_KEY}
        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            genre_map = {g['id']: g['name'].lower().replace(' ', '') for g in response.json()['genres']}
            tmdb_cache[cache_key] = genre_map
            return genre_map
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {media_type} genre map: {str(e)}")
            return None

    async def _fetch_media_from_tmdb(self, query: str = None, year: str = None, media_type: str = 'multi', pages: int = 2) -> pd.DataFrame:
        if self.offline_mode:
            logger.warning("Offline mode: Skipping TMDB fetch, returning empty DataFrame")
            return pd.DataFrame()
        cache_key = f"media_{media_type}_{query}_{year}_{pages}"
        if cache_key in tmdb_cache:
            logger.debug(f"Cache hit for {cache_key}")
            return tmdb_cache[cache_key]
        all_results = []
        
        async def fetch_page(page):
            url = f"{TMDB_BASE_URL}/search/{'multi' if media_type == 'multi' else media_type}"
            params = {
                'api_key': TMDB_API_KEY,
                'query': query if query else '',
                'year': year if media_type == 'movie' and year else None,
                'first_air_date_year': year if media_type == 'tv' and year else None,
                'page': page,
                'include_adult': 'false' if self.adult_filter else 'true'
            }
            try:
                response = await asyncio.get_event_loop().run_in_executor(None, 
                    lambda: session.get(url, params=params, timeout=10))
                response.raise_for_status()
                return response.json().get('results', [])
            except requests.RequestException as e:
                logger.error(f"Failed to fetch page {page} for {media_type}: {str(e)}")
                return []

        tasks = [fetch_page(page) for page in range(1, pages + 1)]
        results = await asyncio.gather(*tasks)
        
        for page_results in results:
            all_results.extend(page_results)
        
        if not all_results:
            logger.debug(f"No results for query: {query}, media_type: {media_type}")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_results)
        if media_type == 'multi':
            df = df[df['media_type'].isin(['movie', 'tv'])]
        df['type'] = df.get('media_type', media_type)
        df['title'] = df.apply(lambda x: x['title'] if x['type'] == 'movie' else x.get('name', ''), axis=1)
        df['release_date'] = df.apply(lambda x: x.get('release_date') if x['type'] == 'movie' else x.get('first_air_date'), axis=1)
        df['genres'] = df['genre_ids'].apply(lambda x: ','.join([self.genre_map.get(gid, self.tv_genre_map.get(gid, '')) for gid in (x if isinstance(x, (list, tuple)) else [])]))
        df['adult'] = df.get('adult', False).astype(bool)
        df['rating'] = df['vote_average'].fillna(0)
        df['vote_count'] = df['vote_count'].fillna(0).astype(int)
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['poster_path'] = df['poster_path'].fillna('')
        df['overview'] = df['overview'].fillna('')
        df = self._apply_adult_filter(df)
        logger.debug(f"Fetched {len(df)} {media_type} items for query: {query}")
        tmdb_cache[cache_key] = df
        return df[['id', 'title', 'genres', 'rating', 'vote_count', 'release_date', 'poster_path', 'adult', 'overview', 'type']]

    def _fetch_media_details(self, media_id: int, media_type: str) -> Dict[str, Any]:
        cache_key = f"details_{media_type}_{media_id}"
        if cache_key in tmdb_cache:
            return tmdb_cache[cache_key]
        url = f"{TMDB_BASE_URL}/{media_type}/{media_id}"
        params = {'api_key': TMDB_API_KEY, 'append_to_response': 'credits,production_companies'}
        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            details = response.json()
            tmdb_cache[cache_key] = details
            return details
        except requests.RequestException as e:
            logger.error(f"Failed to fetch details for {media_type} {media_id}: {str(e)}")
            return {}

    def _apply_adult_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.adult_filter and 'adult' in df.columns:
            return df[~df['adult']]
        return df

    def _format_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        if df.empty:
            return []
        df['poster_url'] = df['poster_path'].apply(lambda x: f"https://image.tmdb.org/t/p/w185{x}" if x else None)
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_date'] = df['release_date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '')
        logger.debug(f"Formatted {len(df)} recommendations")
        return df[['id', 'title', 'genres', 'rating', 'release_date', 'overview', 'poster_url', 'type']].replace({np.nan: None}).to_dict('records')

    def _fallback_recommendations(self, num_recommendations: int, media_type: str) -> List[Dict]:
        if self.offline_mode:
            logger.warning("Offline mode: Returning dummy recommendations")
            dummy_data = [
                {'id': i, 'title': f"Dummy {media_type} {i}", 'genres': 'action,comedy', 'rating': 7.0,
                 'release_date': '2023-01-01', 'overview': 'A dummy movie for testing', 
                 'poster_url': 'https://via.placeholder.com/185', 'type': media_type}
                for i in range(1, num_recommendations + 1)
            ]
            return dummy_data
        url = f"{TMDB_BASE_URL}/{media_type}/popular"
        params = {'api_key': TMDB_API_KEY, 'page': 1}
        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get('results', [])
            df = pd.DataFrame(data)
            df['type'] = media_type
            df['title'] = df.apply(lambda x: x['title'] if media_type == 'movie' else x.get('name', ''), axis=1)
            df['release_date'] = df.apply(lambda x: x.get('release_date') if media_type == 'movie' else x.get('first_air_date'), axis=1)
            df['genres'] = df['genre_ids'].apply(lambda x: ','.join([self.genre_map.get(gid, self.tv_genre_map.get(gid, '')) for gid in (x if isinstance(x, (list, tuple)) else [])]))
            df['adult'] = df.get('adult', False).astype(bool)
            df['rating'] = df['vote_average'].fillna(0)
            df['vote_count'] = df['vote_count'].fillna(0).astype(int)
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['poster_path'] = df['poster_path'].fillna('')
            df['overview'] = df['overview'].fillna('')
            recommendations = self._apply_adult_filter(df)
            recommendations = recommendations.sort_values('rating', ascending=False).head(num_recommendations)
            return self._format_recommendations(recommendations)
        except requests.RequestException as e:
            logger.error(f"Fallback fetch failed for {media_type}: {str(e)}")
            return []

    def toggle_adult_filter(self, enable: bool) -> None:
        with self._lock:
            self.adult_filter = enable
        logger.info(f"Adult filter {'enabled' if enable else 'disabled'}")

    def _is_genre_term(self, term: str) -> bool:
        term_lower = term.lower().replace(' ', '')
        genres = [g.lower().replace(' ', '') for g in self.genre_map.values()] + \
                 [g.lower().replace(' ', '') for g in self.tv_genre_map.values()]
        is_genre = term_lower in genres
        logger.debug(f"Checking if '{term_lower}' is a genre: {is_genre}")
        return is_genre

    def _is_year_term(self, term: str) -> bool:
        is_year = bool(re.match(r'^\d{4}$', term)) and 1888 <= int(term) <= 2100
        logger.debug(f"Checking if '{term}' is a year: {is_year}")
        return is_year

    def get_recommendations_by_keywords(self, keywords: str, media_type: str = 'multi', num_recommendations: int = 50) -> List[Dict]:
        keywords_lower = keywords.lower().strip()
        logger.debug(f"Processing query: {keywords_lower} for {media_type}")
        
        year_match = re.search(r'\b(\d{4})\b', keywords_lower)
        year = year_match.group(1) if year_match else None
        search_term = re.sub(r'\b\d{4}\b', '', keywords_lower).strip() if year else keywords_lower
        
        try:
            all_media = asyncio.run(self._fetch_media_from_tmdb(search_term or '', year, media_type, pages=2))
            recommendations = pd.DataFrame()
            
            if year:
                all_media = all_media[all_media['release_date'].dt.year.astype(str) == year]
                logger.debug(f"Filtered by year {year}: {len(all_media)} items remain")
            
            if not search_term and year and self._is_year_term(year):
                recommendations = all_media.sort_values('rating', ascending=False)
                logger.debug(f"Year-only search for {year}: {len(recommendations)} matches")
            
            elif search_term:
                is_genre = self._is_genre_term(search_term)
                
                if is_genre:
                    genre_matches = all_media[all_media['genres'].str.lower().str.contains(search_term.lower(), na=False)]
                    recommendations = genre_matches
                    logger.debug(f"Genre search for '{search_term}': {len(genre_matches)} matches")
                    
                    if recommendations.empty:
                        logger.debug(f"No genre matches found for '{search_term}', trying keywords")
                
                if not is_genre or recommendations.empty:
                    keyword_parts = search_term.split()
                    keyword_matches = all_media[
                        all_media['overview'].str.lower().str.contains('|'.join(keyword_parts), na=False) |
                        all_media['title'].str.lower().str.contains('|'.join(keyword_parts), na=False)
                    ]
                    recommendations = keyword_matches
                    logger.debug(f"Keyword search for '{search_term}': {len(keyword_matches)} matches")
                    
                    if len(recommendations) < self.MIN_RECOMMENDATIONS:
                        exact_matches = all_media[all_media['title'].str.lower() == search_term.lower()]
                        recommendations = pd.concat([recommendations, exact_matches]).drop_duplicates(subset='id')
                        logger.debug(f"Added {len(exact_matches)} exact title matches")
            
            recommendations = recommendations.sort_values('rating', ascending=False)
            logger.debug(f"Total matches after initial filtering: {len(recommendations)}")
            
            if len(recommendations) < num_recommendations:
                remaining = num_recommendations - len(recommendations)
                logger.info(f"Insufficient matches ({len(recommendations)}), fetching {remaining} fallback recommendations")
                if media_type == 'multi':
                    movie_fallback = pd.DataFrame(self._fallback_recommendations(remaining // 2, 'movie'))
                    tv_fallback = pd.DataFrame(self._fallback_recommendations(remaining // 2, 'tv'))
                    fallback = pd.concat([movie_fallback, tv_fallback])
                else:
                    fallback = pd.DataFrame(self._fallback_recommendations(remaining, media_type))
                recommendations = pd.concat([recommendations, fallback]).drop_duplicates(subset='id')
                logger.debug(f"Added {len(fallback)} fallback recommendations")
            
            recs = self._format_recommendations(recommendations.head(num_recommendations))
            logger.debug(f"Returning {len(recs)} recommendations")
            return recs
        except Exception as e:
            logger.error(f"Error in get_recommendations_by_keywords: {str(e)}")
            return self._fallback_recommendations(num_recommendations, media_type)

    def get_recommendations_from_watchlist(self, watchlist_df_tuple: tuple, num_recommendations: int = 50) -> List[Dict]:
        try:
            watchlist_df = pd.DataFrame(watchlist_df_tuple)
            column_mapping = {'Genres': 'genres', 'IMDb Rating': 'rating', 'Num Votes': 'vote_count', 'Title': 'title', 'Type': 'type'}
            
            for orig, mapped in column_mapping.items():
                watchlist_df[orig] = watchlist_df.get(orig, watchlist_df.get(mapped, ''))
            
            watchlist_df['Genres'] = watchlist_df['Genres'].fillna('').str.lower().str.replace(' ', '')
            genre_counts = pd.Series(' '.join(watchlist_df['Genres']).split(',')).value_counts()
            preferred_genres = ','.join(genre_counts.index[:3]) if not genre_counts.empty else ''
            
            if not preferred_genres:
                logger.debug("No preferred genres found, using fallback")
                return self._fallback_recommendations(num_recommendations, 'multi')
            
            all_media = pd.DataFrame()
            pages_to_fetch = 2
            for genre in preferred_genres.split(','):
                movies = asyncio.run(self._fetch_media_from_tmdb(query=genre, media_type='multi', pages=pages_to_fetch))
                all_media = pd.concat([all_media, movies]).drop_duplicates(subset='id')
            
            if len(all_media) < num_recommendations:
                extra_media = asyncio.run(self._fetch_media_from_tmdb(pages=pages_to_fetch))
                all_media = pd.concat([all_media, extra_media]).drop_duplicates(subset='id')
            
            if 'Title' in watchlist_df:
                all_media = all_media[~all_media['title'].str.lower().isin(set(watchlist_df['Title'].str.lower()))]
            
            user_prefs = pd.DataFrame([{'genres': preferred_genres}])
            genre_matrix = self.genre_vectorizer.fit_transform(all_media['genres'])
            user_matrix = self.genre_vectorizer.transform(user_prefs['genres'])
            
            cosine_sim = cosine_similarity(user_matrix, genre_matrix)[0]
            sim_scores = list(enumerate(cosine_sim))
            top_matches = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            recommendations = pd.DataFrame()
            for idx, _ in top_matches:
                if len(recommendations) >= num_recommendations:
                    break
                media = all_media.iloc[[idx]]
                recommendations = pd.concat([recommendations, media])
            
            if len(recommendations) < num_recommendations:
                remaining = num_recommendations - len(recommendations)
                movie_fallback = pd.DataFrame(self._fallback_recommendations(remaining // 2, 'movie'))
                tv_fallback = pd.DataFrame(self._fallback_recommendations(remaining // 2, 'tv'))
                recommendations = pd.concat([recommendations, movie_fallback, tv_fallback]).drop_duplicates(subset='id')
            
            recs = self._format_recommendations(recommendations.head(num_recommendations))
            return recs
        except Exception as e:
            logger.error(f"Error in get_recommendations_from_watchlist: {str(e)}")
            return self._fallback_recommendations(num_recommendations, 'multi')

recommender = MovieRecommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_watchlist():
    file = request.files.get('file')
    if not file or file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a valid CSV file'}), 400
    
    try:
        watchlist_df = pd.read_csv(file)
        column_mapping = {
            'title': ['Title', 'MovieName', 'Media', 'Name'],
            'genres': ['Genres', 'Genre', 'Category'],
            'rating': ['IMDb Rating', 'Score', 'UserRating'],
            'vote_count': ['Num Votes', 'VoteCount', 'Votes'],
            'type': ['Type', 'MediaType']
        }
        
        for expected, alternates in column_mapping.items():
            if expected not in watchlist_df.columns:
                for alt in alternates:
                    if alt in watchlist_df.columns:
                        watchlist_df[expected] = watchlist_df[alt]
                        break
        
        missing = [col for col in ['title', 'genres', 'rating'] if col not in watchlist_df.columns]
        if missing:
            return jsonify({'error': f'Missing required columns: {", ".join(missing)}'}), 400
        
        recommendations = recommender.get_recommendations_from_watchlist(tuple(watchlist_df.to_dict('records')))
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Error processing CSV: {str(e)}'}), 400

@app.route('/search', methods=['POST'])
def search_keywords():
    if not request.is_json or not (data := request.get_json()) or not (keywords := data.get('keywords')):
        return jsonify({'error': 'Invalid JSON request or missing keywords'}), 400
    
    try:
        adult_filter = data.get('adult_filter', None)
        if adult_filter is not None:
            recommender.toggle_adult_filter(bool(adult_filter))
        
        media_type = data.get('media_type', 'multi')
        recommendations = recommender.get_recommendations_by_keywords(keywords, media_type)
        if not recommendations:
            return jsonify({'error': 'No recommendations found for query', 'adult_filter_enabled': recommender.adult_filter}), 200
        return jsonify({
            'recommendations': recommendations,
            'adult_filter_enabled': recommender.adult_filter
        })
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': f'Error processing search: {str(e)}'}), 500

@app.route('/media/<media_type>/<int:media_id>', methods=['GET'])
def get_media_details(media_type, media_id):
    if media_type not in ['movie', 'tv']:
        return jsonify({'error': 'Invalid media type'}), 400
    try:
        details = recommender._fetch_media_details(media_id, media_type)
        if not details:
            return jsonify({'error': 'Media not found'}), 404
        return jsonify(details)
    except Exception as e:
        logger.error(f"Error fetching media details: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reviews/<media_type>/<int:media_id>', methods=['GET'])
def get_reviews(media_type, media_id):
    if media_type not in ['movie', 'tv']:
        return jsonify({'error': 'Invalid media type'}), 400
    try:
        with sqlite3.connect('reviews.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name, rating, review FROM reviews WHERE media_type = ? AND media_id = ?', (media_type, media_id))
            reviews = [{'name': row[0], 'rating': row[1], 'review': row[2]} for row in cursor.fetchall()]
        return jsonify(reviews)
    except Exception as e:
        logger.error(f"Error fetching reviews: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reviews/<media_type>/<int:media_id>', methods=['POST'])
def add_review(media_type, media_id):
    if media_type not in ['movie', 'tv']:
        return jsonify({'error': 'Invalid media type'}), 400
    if not request.is_json or not (data := request.get_json()):
        return jsonify({'error': 'Invalid JSON request'}), 400
    
    name = data.get('name', 'Anonymous')
    rating = data.get('rating')
    review = data.get('review')
    
    if not isinstance(rating, (int, float)) or rating < 1 or rating > 10:
        return jsonify({'error': 'Rating must be a number between 1 and 10'}), 400
    if not isinstance(review, str) or not review.strip():
        return jsonify({'error': 'Review must be a non-empty string'}), 400
    
    try:
        with sqlite3.connect('reviews.db') as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO reviews (media_type, media_id, name, rating, review) VALUES (?, ?, ?, ?, ?)',
                           (media_type, media_id, name, rating, review.strip()))
            conn.commit()
        return jsonify({'message': 'Review added successfully'})
    except Exception as e:
        logger.error(f"Error adding review: {str(e)}")
        return jsonify({'error': str(e)}), 500

def cleanup():
    session.close()
    loop.close()

atexit.register(cleanup)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=3000, debug=True, use_reloader=False)
    finally:
        cleanup()