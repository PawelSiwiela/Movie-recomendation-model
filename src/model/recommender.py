"""
System rekomendacji wykorzystujący wytrenowany model.

Zawiera:
- Ładowanie wytrenowanego modelu
- Generowanie rekomendacji dla użytkownika
- Ranking filmów według przewidywanych ocen
- Filtrowanie (gatunki, rok, aktorzy)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Tuple
import sys

# Dodaj src/model do ścieżki
sys.path.insert(0, str(Path(__file__).parent))
from model import create_model


class MovieRecommender:
    """
    System rekomendacji filmów.
    
    UWAGA: Model przewiduje wynik dopasowania w skali 0-1.
    """
    
    def __init__(
        self,
        model_path: str,
        enriched_data_path: str,
        encoders_path: str,
        db_path: str,
        device: str = 'cpu'
    ):
        """
        Inicjalizacja systemu rekomendacji.
        
        Args:
            model_path: Ścieżka do wytrenowanego modelu (.pth)
            enriched_data_path: Ścieżka do enriched_movies.csv (filmy użytkownika)
            encoders_path: Ścieżka do encoders.pkl
            db_path: Ścieżka do database/movies.db (pełna baza TMDB)
            device: 'cpu' lub 'cuda'
        """
        self.device = torch.device(device)
        self.db_path = db_path
        
        # Załaduj dane
        print("Laduje dane...")
        self.movies_df = pd.read_csv(enriched_data_path)
        
        # Załaduj encodery
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        self.scaler = self.encoders['scaler']
        self.genre_encoder = self.encoders['mlb_genres']
        self.top_directors = list(self.encoders['top_directors'])
        self.top_actors = list(self.encoders['top_actors'])
        self.actor_to_idx = self.encoders['actor_to_idx']
        
        self.n_genres = len(self.genre_encoder.classes_)
        self.n_directors = len(self.top_directors) + 1
        self.n_actors = len(self.top_actors)
        
        print(f"   Wymiary enkoderów: gatunki={self.n_genres}, reżyserzy={self.n_directors}, aktorzy={self.n_actors}")
        
        # Załaduj model
        print("Laduje model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'input_dim' in checkpoint:
            input_dim = checkpoint['input_dim']
            print(f"   Wykryto wymiar modelu: {input_dim} (z metadanych)")
        else:
            raise ValueError("Nie można wykryć wymiaru wejściowego modelu z checkpoint!")
        
        self.model = create_model(input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"OK Recommender gotowy!")
        print(f"   Filmy uzytkownika: {len(self.movies_df)}")

    def _fetch_movie_from_db(self, movie_id: int) -> Optional[pd.Series]:
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT 
            m.id as tmdb_id, m.title, m.year, m.rating as tmdb_rating, m.popularity as tmdb_popularity, m.type as tmdb_type,
            GROUP_CONCAT(DISTINCT g.name) as genres,
            (SELECT d.name FROM movie_directors md2 JOIN directors d ON md2.director_id = d.id WHERE md2.movie_id = m.id LIMIT 1) as director,
            (SELECT GROUP_CONCAT(a.name) FROM movie_actors ma2 JOIN actors a ON ma2.actor_id = a.id WHERE ma2.movie_id = m.id ORDER BY ma2.cast_order LIMIT 5) as actors
        FROM movies m
        LEFT JOIN movie_genres mg ON m.id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE m.id = ?
        GROUP BY m.id
        """
        result = pd.read_sql_query(query, conn, params=(movie_id,))
        conn.close()
        if result.empty: return None
        row = result.iloc[0].copy()
        for col in ['genres', 'actors']:
            if pd.notna(row[col]): row[col] = str([s.strip() for s in row[col].split(',')])
            else: row[col] = '[]'
        for col in ['director', 'tmdb_rating', 'tmdb_popularity', 'year']:
            if pd.isna(row.get(col)): row[col] = 0 if 'rating' in col or 'popularity' in col else 'Unknown'
        return row

    def _create_features(self, movie_row: pd.Series) -> np.ndarray:
        features = []
        num_vals = movie_row[['tmdb_rating', 'tmdb_popularity', 'year']].values.reshape(1, -1)
        features.extend(self.scaler.transform(num_vals)[0])
        
        try: genres = eval(movie_row['genres']) if isinstance(movie_row['genres'], str) else []
        except: genres = []
        known_genres = [g for g in genres if g in self.genre_encoder.classes_]
        features.extend(self.genre_encoder.transform([known_genres])[0])
        
        director_vec = np.zeros(self.n_directors)
        if pd.notna(movie_row['director']) and movie_row['director'] in self.top_directors:
            director_vec[self.top_directors.index(movie_row['director'])] = 1
        else: director_vec[-1] = 1
        features.extend(director_vec)
        
        try: actors = eval(movie_row['actors']) if isinstance(movie_row['actors'], str) else []
        except: actors = []
        actor_vec = np.zeros(self.n_actors)
        for actor in actors[:5]:
            if actor in self.actor_to_idx:
                idx = self.actor_to_idx[actor]
                if idx < self.n_actors: actor_vec[idx] = 1
        features.extend(actor_vec)
        
        movie_type = movie_row.get('tmdb_type', 'movie')
        features.extend([1 if movie_type == 'movie' else 0, 1 if movie_type == 'tv' else 0])
        
        return np.array(features, dtype=np.float32)

    def predict_score(self, movie_id: int) -> float:
        """
        Przewiduje wynik dopasowania dla pojedynczego filmu.
        
        Args:
            movie_id: ID filmu w TMDB
            
        Returns:
            Przewidywany wynik dopasowania [0, 1]
        """
        movie = self.movies_df[self.movies_df['tmdb_id'] == movie_id]
        movie_data = movie.iloc[0] if not movie.empty else self._fetch_movie_from_db(movie_id)
        if movie_data is None: raise ValueError(f"Film {movie_id} nie został znaleziony")
        
        features = self._create_features(movie_data)
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            prediction = self.model(X).item()
        return prediction

    def get_top_recommendations(
        self, watched_movie_ids: List[int], n: int = 10, min_score: Optional[float] = None,
        movie_type: Optional[str] = None, min_popularity: float = 5.0, **kwargs
    ) -> pd.DataFrame:
        """
        Generuje top rekomendacje, zwracając filmy z najwyższymi wynikami dopasowania.
        
        Args:
            min_score: Minimalny przewidywany wynik dopasowania [0, 1]
            ... (pozostałe argumenty bez zmian)
        """
        import sqlite3
        print(f"Generuje top {n} rekomendacji z bazy danych...")
        conn = sqlite3.connect(self.db_path)
        
        where_clauses = [f"m.id NOT IN ({','.join(map(str, watched_movie_ids))})", f"m.popularity >= {min_popularity}"]
        if movie_type: where_clauses.append(f"m.type = '{movie_type}'")
        
        query = f"""
        SELECT DISTINCT m.id as tmdb_id, m.title, m.year, m.rating as tmdb_rating, m.popularity as tmdb_popularity, m.type as tmdb_type,
            (SELECT GROUP_CONCAT(DISTINCT g2.name) FROM movie_genres mg2 JOIN genres g2 ON mg2.genre_id = g2.id WHERE mg2.movie_id = m.id) as genres,
            (SELECT d.name FROM movie_directors md2 JOIN directors d ON md2.director_id = d.id WHERE md2.movie_id = m.id LIMIT 1) as director,
            (SELECT GROUP_CONCAT(a.name) FROM movie_actors ma2 JOIN actors a ON ma2.actor_id = a.id WHERE ma2.movie_id = m.id ORDER BY ma2.cast_order LIMIT 5) as actors
        FROM movies m WHERE {' AND '.join(where_clauses)} ORDER BY m.popularity DESC LIMIT 1000
        """
        candidates = pd.read_sql_query(query, conn)
        conn.close()
        print(f"   Pobrаno {len(candidates)} kandydatów z bazy")
        if candidates.empty: return pd.DataFrame()

        for idx, row in candidates.iterrows():
            for col in ['genres', 'actors']:
                if pd.notna(row[col]): candidates.at[idx, col] = str(row[col].split(','))
                else: candidates.at[idx, col] = '[]'
        
        print("   Przewidywanie wyników dopasowania...")
        predictions = []
        for _, row in candidates.iterrows():
            try:
                features = self._create_features(row)
                with torch.no_grad():
                    X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    pred = self.model(X).item()
                predictions.append(pred)
            except Exception: predictions.append(0.0)
        
        candidates['match_score'] = predictions
        
        if min_score: candidates = candidates[candidates['match_score'] >= min_score]
        
        recommendations = candidates.nlargest(n, 'match_score')
        
        result = recommendations[[
            'tmdb_id', 'title', 'year', 'tmdb_type', 'match_score', 
            'tmdb_rating', 'tmdb_popularity', 'genres', 'director', 'actors'
        ]].reset_index(drop=True)
        result = result.rename(columns={'tmdb_type': 'type'})
        print(f"✅ Wygenerowano {len(result)} rekomendacji")
        return result

def format_recommendations(df: pd.DataFrame, content_type: str = "FILMOW") -> None:
    """Wyświetla rekomendacje w czytelny sposób, używając wyniku dopasowania."""
    print("\n" + "="*100)
    print(f"REKOMENDACJE {content_type}")
    print("="*100)
    
    for idx, row in df.iterrows():
        print(f"\n{idx + 1}. {row['title']} ({row['year']}) [{row['type'].upper()}]")
        
        match_score = row.get('match_score', 0)
        print(f"   Wynik dopasowania: {match_score:.0%}")
        
        if 'tmdb_rating' in row:
            print(f"   TMDB: {row['tmdb_rating']:.1f}/10 | Popularnosc: {row['tmdb_popularity']:.1f}")
        
        try: genres = eval(row['genres']) if isinstance(row['genres'], str) else []
        except: genres = []
        print(f"   Gatunki: {', '.join(genres)}")
        
        if pd.notna(row['director']):
            print(f"   Rezyser: {row['director']}")
        
        try: actors = eval(row['actors']) if isinstance(row['actors'], str) else []
        except: actors = []
        if actors:
            print(f"   Obsada: {', '.join(actors[:3])}")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    # Ten blok testowy wymaga aktualizacji, aby odzwierciedlić zmiany
    # Na razie go pomijamy, ponieważ główna logika jest w pipeline.py
    print("✅ Recommender refactored. Run pipeline.py to test.")