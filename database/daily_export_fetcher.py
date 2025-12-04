"""
Fetcher wykorzystujący TMDB Daily Export do pobrania pełnej bazy filmów.
Pobiera listę wszystkich ID z daily export, następnie dla każdego filmu pobiera szczegóły i credits.
"""

import requests
import gzip
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import sys

# Dodaj ścieżkę do modułów
sys.path.append(str(Path(__file__).parent))
from database_setup import DatabaseSetup
from tmdb_client import TMDBClient


class DailyExportFetcher:
    """Fetcher wykorzystujący TMDB Daily Export."""
    
    def __init__(self, db_path: str, api_key: str):
        """
        Args:
            db_path: Ścieżka do bazy SQLite
            api_key: Klucz API TMDB
        """
        self.db_path = db_path
        self.client = TMDBClient(api_key)
        
        # Połączenie z bazą
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Setup bazy
        db_setup = DatabaseSetup(db_path)
        db_setup.create_tables()
    
    def find_latest_export(self, media_type: str = 'movie') -> Optional[str]:
        """
        Znajduje najnowszy dostępny export.
        
        Args:
            media_type: 'movie' lub 'tv'
            
        Returns:
            URL do pliku lub None
        """
        today = datetime.now()
        
        for days_ago in range(0, 7):
            date = today - timedelta(days=days_ago)
            date_str = date.strftime('%m_%d_%Y')
            
            if media_type == 'movie':
                url = f"https://files.tmdb.org/p/exports/movie_ids_{date_str}.json.gz"
            else:
                url = f"https://files.tmdb.org/p/exports/tv_series_ids_{date_str}.json.gz"
            
            try:
                response = requests.head(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ Znaleziono export z {date_str}")
                    size_mb = int(response.headers.get('content-length', 0)) / 1024 / 1024
                    print(f"   Rozmiar: {size_mb:.1f} MB")
                    return url
            except:
                continue
        
        return None
    
    def download_and_parse_export(
        self,
        url: str,
        min_popularity: float = 0.0,
        exclude_adult: bool = True,
        exclude_video: bool = True
    ) -> list:
        """
        Pobiera i parsuje export, zwraca listę ID do pobrania.
        
        Args:
            url: URL do pliku export
            min_popularity: Minimalny próg popularności
            exclude_adult: Czy wykluczyć treści dla dorosłych
            exclude_video: Czy wykluczyć video content (straight-to-video itp.)
            
        Returns:
            Lista ID filmów do pobrania
        """
        print(f"\n📥 Pobieram export z TMDB...")
        response = requests.get(url, stream=True, timeout=120)
        
        ids_to_fetch = []
        total_scanned = 0
        
        print("🔍 Filtruję filmy...")
        
        with gzip.GzipFile(fileobj=response.raw) as f:
            for line in f:
                try:
                    data = json.loads(line.decode('utf-8'))
                    total_scanned += 1
                    
                    # Filtry
                    if exclude_adult and data.get('adult', False):
                        continue
                    
                    if exclude_video and data.get('video', False):
                        continue
                    
                    if data.get('popularity', 0) < min_popularity:
                        continue
                    
                    ids_to_fetch.append(data['id'])
                    
                    if total_scanned % 100000 == 0:
                        print(f"   Przeskanowano {total_scanned:,} filmów, zakwalifikowano {len(ids_to_fetch):,}")
                
                except:
                    continue
        
        print(f"\n✅ Zakończono skanowanie")
        print(f"   Wszystkich filmów: {total_scanned:,}")
        print(f"   Po filtrach: {len(ids_to_fetch):,}")
        
        return ids_to_fetch
    
    def _insert_or_get_genre(self, genre_id: int, name: str) -> int:
        """Wstawia lub pobiera ID gatunku."""
        self.cursor.execute("SELECT id FROM genres WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        self.cursor.execute("INSERT INTO genres (name) VALUES (?)", (name,))
        return self.cursor.lastrowid
    
    def _insert_or_get_actor(self, tmdb_id: int, name: str) -> Optional[int]:
        """Wstawia lub pobiera ID aktora."""
        self.cursor.execute("SELECT id FROM actors WHERE tmdb_id = ?", (tmdb_id,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        self.cursor.execute("INSERT INTO actors (tmdb_id, name) VALUES (?, ?)", (tmdb_id, name))
        return self.cursor.lastrowid
    
    def _insert_or_get_director(self, tmdb_id: int, name: str) -> Optional[int]:
        """Wstawia lub pobiera ID reżysera."""
        self.cursor.execute("SELECT id FROM directors WHERE tmdb_id = ?", (tmdb_id,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        self.cursor.execute("INSERT INTO directors (tmdb_id, name) VALUES (?, ?)", (tmdb_id, name))
        return self.cursor.lastrowid
    
    def _save_movie(self, movie_data: Dict, media_type: str = 'movie') -> Optional[int]:
        """Zapisuje film do bazy."""
        try:
            tmdb_id = movie_data['id']
            title = movie_data.get('title') or movie_data.get('name', 'Unknown')
            original_title = movie_data.get('original_title') or movie_data.get('original_name', title)
            
            # Rok wydania
            year = None
            if media_type == 'movie':
                release_date = movie_data.get('release_date', '')
                year = int(release_date[:4]) if release_date else None
            else:
                first_air = movie_data.get('first_air_date', '')
                year = int(first_air[:4]) if first_air else None
            
            description = movie_data.get('overview', '')
            rating = movie_data.get('vote_average')
            vote_count = movie_data.get('vote_count')
            popularity = movie_data.get('popularity')
            
            # Sprawdź czy już istnieje
            self.cursor.execute("SELECT id FROM movies WHERE tmdb_id = ?", (tmdb_id,))
            existing = self.cursor.fetchone()
            if existing:
                return existing[0]
            
            # Wstaw film
            self.cursor.execute("""
                INSERT INTO movies (tmdb_id, title, original_title, year, description, 
                                   rating, vote_count, popularity, type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (tmdb_id, title, original_title, year, description, 
                  rating, vote_count, popularity, media_type))
            
            movie_id = self.cursor.lastrowid
            
            # Dodaj gatunki
            for genre in movie_data.get('genres', []):
                genre_id = self._insert_or_get_genre(genre['id'], genre['name'])
                self.cursor.execute(
                    "INSERT OR IGNORE INTO movie_genres (movie_id, genre_id) VALUES (?, ?)",
                    (movie_id, genre_id)
                )
            
            return movie_id
            
        except Exception as e:
            print(f"❌ Błąd zapisu filmu {movie_data.get('title', 'Unknown')}: {e}")
            return None
    
    def _save_credits(self, movie_id: int, credits_data: Dict):
        """Zapisuje obsadę i reżyserów."""
        try:
            # Top 10 aktorów
            cast = credits_data.get('cast', [])[:10]
            for actor in cast:
                actor_id = self._insert_or_get_actor(actor['id'], actor['name'])
                if actor_id:
                    self.cursor.execute("""
                        INSERT OR IGNORE INTO movie_actors 
                        (movie_id, actor_id, character_name, cast_order)
                        VALUES (?, ?, ?, ?)
                    """, (movie_id, actor_id, actor.get('character'), actor.get('order')))
            
            # Reżyserzy
            crew = credits_data.get('crew', [])
            directors = [c for c in crew if c.get('job') == 'Director']
            for director in directors:
                director_id = self._insert_or_get_director(director['id'], director['name'])
                if director_id:
                    self.cursor.execute(
                        "INSERT OR IGNORE INTO movie_directors (movie_id, director_id) VALUES (?, ?)",
                        (movie_id, director_id)
                    )
        except Exception as e:
            print(f"❌ Błąd zapisu credits: {e}")
    
    def fetch_movies_from_export(
        self,
        min_popularity: float = 1.0,
        max_movies: Optional[int] = None
    ):
        """
        Pobiera filmy z daily export.
        
        Args:
            min_popularity: Minimalny próg popularności (0.0 = wszystkie)
            max_movies: Maksymalna liczba filmów do pobrania (None = wszystkie)
        """
        print("🎬 Rozpoczynam pobieranie filmów z TMDB Daily Export")
        print(f"   Min popularity: {min_popularity}")
        if max_movies:
            print(f"   Max filmów: {max_movies:,}")
        
        # 1. Znajdź export
        export_url = self.find_latest_export('movie')
        if not export_url:
            print("❌ Nie znaleziono exportu")
            return
        
        # 2. Pobierz i przefiltruj listę ID
        movie_ids = self.download_and_parse_export(
            export_url,
            min_popularity=min_popularity,
            exclude_adult=True,
            exclude_video=True
        )
        
        if max_movies:
            movie_ids = movie_ids[:max_movies]
            print(f"\n⚠️  Ograniczam do {max_movies:,} filmów")
        
        # 3. Pobierz szczegóły dla każdego filmu
        print(f"\n📡 Pobieram szczegóły dla {len(movie_ids):,} filmów...")
        
        fetched = 0
        errors = 0
        start_time = time.time()
        
        for i, tmdb_id in enumerate(movie_ids, 1):
            try:
                # Pobierz szczegóły
                details = self.client.get_movie_details(tmdb_id)
                if not details:
                    errors += 1
                    continue
                
                # Zapisz film
                db_movie_id = self._save_movie(details, 'movie')
                if not db_movie_id:
                    errors += 1
                    continue
                
                # Pobierz i zapisz credits
                credits = self.client.get_movie_credits(tmdb_id)
                if credits:
                    self._save_credits(db_movie_id, credits)
                
                fetched += 1
                
                # Progress co 100 filmów
                if i % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    remaining = len(movie_ids) - i
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    print(f"   [{i:,}/{len(movie_ids):,}] Pobrano: {fetched:,}, Błędy: {errors}, "
                          f"Tempo: {rate:.1f} film/s, ETA: {eta_minutes:.0f} min")
                    
                    self.conn.commit()
            
            except KeyboardInterrupt:
                print("\n⚠️  Przerwano przez użytkownika")
                break
            except Exception as e:
                errors += 1
                if errors % 10 == 0:
                    print(f"   Błąd przy filmie {tmdb_id}: {e}")
        
        # Final commit
        self.conn.commit()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Zakończono pobieranie")
        print(f"   Pobrano: {fetched:,} filmów")
        print(f"   Błędy: {errors}")
        print(f"   Czas: {elapsed/60:.1f} min")
    
    def fetch_tv_from_export(
        self,
        min_popularity: float = 5.0,
        max_shows: Optional[int] = None
    ):
        """Pobiera seriale z daily export."""
        print("📺 Rozpoczynam pobieranie seriali z TMDB Daily Export")
        print(f"   Min popularity: {min_popularity}")
        if max_shows:
            print(f"   Max seriali: {max_shows:,}")
        
        # Znajdź export
        export_url = self.find_latest_export('tv')
        if not export_url:
            print("❌ Nie znaleziono exportu")
            return
        
        # Pobierz listę ID
        tv_ids = self.download_and_parse_export(
            export_url,
            min_popularity=min_popularity,
            exclude_adult=True,
            exclude_video=False  # Dla seriali video=False nie ma sensu
        )
        
        if max_shows:
            tv_ids = tv_ids[:max_shows]
            print(f"\n⚠️  Ograniczam do {max_shows:,} seriali")
        
        # Pobierz szczegóły
        print(f"\n📡 Pobieram szczegóły dla {len(tv_ids):,} seriali...")
        
        fetched = 0
        errors = 0
        start_time = time.time()
        
        for i, tmdb_id in enumerate(tv_ids, 1):
            try:
                details = self.client.get_tv_details(tmdb_id)
                if not details:
                    errors += 1
                    continue
                
                db_tv_id = self._save_movie(details, 'tv')
                if not db_tv_id:
                    errors += 1
                    continue
                
                credits = self.client.get_tv_credits(tmdb_id)
                if credits:
                    self._save_credits(db_tv_id, credits)
                
                fetched += 1
                
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    remaining = len(tv_ids) - i
                    eta_minutes = (remaining / rate / 60) if rate > 0 else 0
                    
                    print(f"   [{i:,}/{len(tv_ids):,}] Pobrano: {fetched:,}, Błędy: {errors}, "
                          f"ETA: {eta_minutes:.0f} min")
                    
                    self.conn.commit()
            
            except KeyboardInterrupt:
                print("\n⚠️  Przerwano przez użytkownika")
                break
            except Exception as e:
                errors += 1
        
        self.conn.commit()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Zakończono pobieranie")
        print(f"   Pobrano: {fetched:,} seriali")
        print(f"   Błędy: {errors}")
        print(f"   Czas: {elapsed/60:.1f} min")
    
    def close(self):
        """Zamyka połączenie z bazą."""
        self.conn.close()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    
    if not api_key:
        print("❌ Brak klucza API! Ustaw TMDB_API_KEY w pliku .env")
        exit(1)
    
    # Ścieżki
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "database" / "movies.db"
    
    fetcher = DailyExportFetcher(str(db_path), api_key)
    
    try:
        # Pobierz 50k najpopularniejszych filmów (popularity >= 1.0)
        fetcher.fetch_movies_from_export(min_popularity=1.0, max_movies=50000)
        
        # Seriale - 2000 najpopularniejszych
        fetcher.fetch_tv_from_export(min_popularity=5.0, max_shows=2000)
        
    finally:
        fetcher.close()
    
    print("\n🎉 Gotowe!")
