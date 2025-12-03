"""Skrypt do pobierania i zapisywania danych z TMDB do bazy."""

import sqlite3
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from tmdb_client import TMDBClient
from database_setup import DatabaseSetup

# Załaduj zmienne środowiskowe z .env
load_dotenv(Path(__file__).parent.parent.parent / '.env')

class DataFetcher:
    """Klasa do pobierania i zapisywania danych z TMDB."""
    
    def __init__(self, api_key: str, db_path: str = "data/movies.db"):
        """
        Inicjalizacja fetchera.
        
        Args:
            api_key: Klucz API TMDB
            db_path: Ścieżka do bazy danych
        """
        self.client = TMDBClient(api_key)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def _insert_or_get_genre(self, genre_id: int, genre_name: str) -> int:
        """Wstawia gatunek lub zwraca istniejący ID."""
        self.cursor.execute(
            "INSERT OR IGNORE INTO genres (id, name) VALUES (?, ?)",
            (genre_id, genre_name)
        )
        return genre_id
    
    def _insert_or_get_actor(self, actor_id: int, actor_name: str) -> int:
        """Wstawia aktora lub zwraca istniejący ID."""
        self.cursor.execute(
            "INSERT OR IGNORE INTO actors (tmdb_id, name) VALUES (?, ?)",
            (actor_id, actor_name)
        )
        self.cursor.execute(
            "SELECT id FROM actors WHERE tmdb_id = ?", (actor_id,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def _insert_or_get_director(self, director_id: int, director_name: str) -> int:
        """Wstawia reżysera lub zwraca istniejący ID."""
        self.cursor.execute(
            "INSERT OR IGNORE INTO directors (tmdb_id, name) VALUES (?, ?)",
            (director_id, director_name)
        )
        self.cursor.execute(
            "SELECT id FROM directors WHERE tmdb_id = ?", (director_id,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def _save_movie(self, movie_data: Dict, media_type: str) -> Optional[int]:
        """
        Zapisuje film/serial do bazy.
        
        Args:
            movie_data: Dane z TMDB
            media_type: 'movie' lub 'tv'
            
        Returns:
            ID wstawionego filmu lub None
        """
        try:
            # Wyciągnij podstawowe dane
            tmdb_id = movie_data.get('id')
            title = movie_data.get('title') or movie_data.get('name')
            original_title = movie_data.get('original_title') or movie_data.get('original_name')
            
            # Rok produkcji
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
            self.cursor.execute(
                "SELECT id FROM movies WHERE tmdb_id = ?", (tmdb_id,)
            )
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
        """
        Zapisuje obsadę i reżyserów.
        
        Args:
            movie_id: ID filmu w bazie
            credits_data: Dane z credits endpoint
        """
        try:
            # Zapisz top 10 aktorów
            cast = credits_data.get('cast', [])[:10]
            for actor in cast:
                actor_id = self._insert_or_get_actor(actor['id'], actor['name'])
                if actor_id:
                    self.cursor.execute("""
                        INSERT OR IGNORE INTO movie_actors 
                        (movie_id, actor_id, character_name, cast_order)
                        VALUES (?, ?, ?, ?)
                    """, (movie_id, actor_id, actor.get('character'), actor.get('order')))
            
            # Zapisz reżyserów
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
            print(f"❌ Błąd zapisu obsady: {e}")
    
    def fetch_movies(self, num_pages: int = 50, min_vote_count: int = 100, min_year: int = 1980):
        """
        Pobiera filmy z TMDB i zapisuje do bazy.
        
        Args:
            num_pages: Liczba stron do pobrania (20 filmów/stronę)
            min_vote_count: Minimalna liczba głosów
            min_year: Minimalny rok produkcji
        """
        print(f"🎬 Rozpoczynam pobieranie filmów (maks. {num_pages * 20} filmów)...")
        
        for page in range(1, num_pages + 1):
            print(f"\n📄 Strona {page}/{num_pages}")
            
            # Pobierz listę filmów
            discover_data = self.client.discover_movies(page, min_vote_count, min_year)
            results = discover_data.get('results', [])
            
            if not results:
                print("Brak więcej filmów")
                break
            
            for movie in results:
                movie_id = movie['id']
                title = movie.get('title', 'Unknown')
                
                # Pobierz szczegóły
                details = self.client.get_movie_details(movie_id)
                if not details:
                    continue
                
                # Zapisz film
                db_movie_id = self._save_movie(details, 'movie')
                if not db_movie_id:
                    continue
                
                # Pobierz i zapisz obsadę
                credits = self.client.get_movie_credits(movie_id)
                if credits:
                    self._save_credits(db_movie_id, credits)
                
                print(f"✅ {title} ({details.get('release_date', '')[:4]})")
            
            self.conn.commit()
        
        print(f"\n🎉 Pobieranie filmów zakończone!")
    
    def fetch_tv_shows(self, num_pages: int = 25, min_vote_count: int = 100, min_year: int = 1980):
        """
        Pobiera seriale z TMDB i zapisuje do bazy.
        
        Args:
            num_pages: Liczba stron do pobrania (20 seriali/stronę)
            min_vote_count: Minimalna liczba głosów
            min_year: Minimalny rok produkcji
        """
        print(f"📺 Rozpoczynam pobieranie seriali (maks. {num_pages * 20} seriali)...")
        
        for page in range(1, num_pages + 1):
            print(f"\n📄 Strona {page}/{num_pages}")
            
            # Pobierz listę seriali
            discover_data = self.client.discover_tv(page, min_vote_count, min_year)
            results = discover_data.get('results', [])
            
            if not results:
                print("Brak więcej seriali")
                break
            
            for tv in results:
                tv_id = tv['id']
                title = tv.get('name', 'Unknown')
                
                # Pobierz szczegóły
                details = self.client.get_tv_details(tv_id)
                if not details:
                    continue
                
                # Zapisz serial
                db_tv_id = self._save_movie(details, 'tv')
                if not db_tv_id:
                    continue
                
                # Pobierz i zapisz obsadę
                credits = self.client.get_tv_credits(tv_id)
                if credits:
                    self._save_credits(db_tv_id, credits)
                
                print(f"✅ {title} ({details.get('first_air_date', '')[:4]})")
            
            self.conn.commit()
        
        print(f"\n🎉 Pobieranie seriali zakończone!")
    
    def close(self):
        """Zamyka połączenie z bazą."""
        self.conn.close()


if __name__ == "__main__":
    # Pobierz klucz API z pliku .env
    API_KEY = os.getenv('TMDB_API_KEY')
    
    if not API_KEY:
        raise ValueError("Brak klucza TMDB_API_KEY w pliku .env!")
    
    # Utwórz bazę danych jeśli nie istnieje
    print("🔧 Inicjalizacja bazy danych...")
    db_setup = DatabaseSetup()
    db_setup.create_tables()
    db_setup.close()
    
    # Pobierz dane
    fetcher = DataFetcher(API_KEY)
    
    try:
        # Pobierz 500 stron filmów (około 10000 filmów)
        fetcher.fetch_movies(num_pages=500, min_vote_count=300, min_year=1950)
        
        # Pobierz 50 stron seriali (około 1000 seriali)
        fetcher.fetch_tv_shows(num_pages=50, min_vote_count=300, min_year=1950)
        
    finally:
        fetcher.close()
    
    print("\n✨ Gotowe! Baza danych wypełniona danymi z TMDB.")
