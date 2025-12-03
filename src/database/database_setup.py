"""Moduł do tworzenia i zarządzania strukturą bazy danych SQLite."""

import sqlite3
from pathlib import Path

class DatabaseSetup:
    """Klasa do inicjalizacji i zarządzania bazą danych."""
    
    def __init__(self, db_path: str = "data/movies.db"):
        """
        Inicjalizacja połączenia z bazą danych.
        
        Args:
            db_path: Ścieżka do pliku bazy danych
        """
        # Upewnij się, że folder data istnieje
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def create_tables(self):
        """Tworzy wszystkie tabele w bazie danych."""
        
        # Tabela filmów/seriali
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tmdb_id INTEGER UNIQUE NOT NULL,
                title TEXT NOT NULL,
                original_title TEXT,
                year INTEGER,
                description TEXT,
                rating REAL,
                vote_count INTEGER,
                popularity REAL,
                type TEXT CHECK(type IN ('movie', 'tv')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela gatunków
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS genres (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        # Tabela aktorów
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS actors (
                id INTEGER PRIMARY KEY,
                tmdb_id INTEGER UNIQUE NOT NULL,
                name TEXT NOT NULL
            )
        """)
        
        # Tabela reżyserów
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS directors (
                id INTEGER PRIMARY KEY,
                tmdb_id INTEGER UNIQUE NOT NULL,
                name TEXT NOT NULL
            )
        """)
        
        # Tabela łącząca filmy z gatunkami
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS movie_genres (
                movie_id INTEGER,
                genre_id INTEGER,
                PRIMARY KEY (movie_id, genre_id),
                FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE,
                FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
            )
        """)
        
        # Tabela łącząca filmy z aktorami
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS movie_actors (
                movie_id INTEGER,
                actor_id INTEGER,
                character_name TEXT,
                cast_order INTEGER,
                PRIMARY KEY (movie_id, actor_id),
                FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE,
                FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE CASCADE
            )
        """)
        
        # Tabela łącząca filmy z reżyserami
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS movie_directors (
                movie_id INTEGER,
                director_id INTEGER,
                PRIMARY KEY (movie_id, director_id),
                FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE,
                FOREIGN KEY (director_id) REFERENCES directors(id) ON DELETE CASCADE
            )
        """)
        
        # Indeksy dla szybszych zapytań
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id ON movies(tmdb_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_movies_type ON movies(type)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_movies_rating ON movies(rating)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_actors_tmdb_id ON actors(tmdb_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_directors_tmdb_id ON directors(tmdb_id)")
        
        self.conn.commit()
        print("✅ Tabele zostały utworzone pomyślnie!")
    
    def close(self):
        """Zamyka połączenie z bazą danych."""
        self.conn.close()

if __name__ == "__main__":
    # Test - utworzenie bazy
    db = DatabaseSetup()
    db.create_tables()
    db.close()
    print("Baza danych gotowa do użycia!")
