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
    """System rekomendacji filmów.
    
    UWAGA: Model przewiduje oceny w skali 0-5 z krokiem 0.5 (skala Letterboxd).
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
        self.genre_encoder = self.encoders['mlb_genres']  # MultiLabelBinarizer dla gatunków
        self.top_directors = list(self.encoders['top_directors'])  # Lista top reżyserów
        self.top_actors = list(self.encoders['top_actors'])  # Lista top aktorów
        self.actor_to_idx = self.encoders['actor_to_idx']  # Mapowanie aktor -> indeks
        
        # Dynamiczne wymiary (z enkoderów, nie hardcoded!)
        self.n_genres = len(self.genre_encoder.classes_)
        self.n_directors = len(self.top_directors) + 1  # +1 dla "other"
        self.n_actors = len(self.top_actors)
        
        print(f"   Wymiary enkoderów: gatunki={self.n_genres}, reżyserzy={self.n_directors}, aktorzy={self.n_actors}")
        
        # Załaduj model
        print("Laduje model...")
        
        # Załaduj checkpoint najpierw, żeby sprawdzić wymiar modelu
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Najpierw spróbuj użyć zapisanych metadanych (nowsze checkpointy)
        if 'input_dim' in checkpoint:
            input_dim = checkpoint['input_dim']
            print(f"   Wykryto wymiar modelu: {input_dim} (z metadanych)")
        else:
            # Fallback dla starszych checkpointów bez metadanych
            print("   ⚠️  Stary checkpoint bez metadanych, wykrywam wymiary...")
            state_dict = checkpoint['model_state_dict']
            input_dim = None
            
            # Próbuj różne możliwe klucze dla pierwszej warstwy
            possible_keys = [
                'input_layer.0.weight',     # MovieRecommenderNet input layer
                'attention.0.weight'        # Jeśli attention jest pierwsza
            ]
            
            for key in possible_keys:
                if key in state_dict:
                    input_dim = state_dict[key].shape[1]  # [out_features, in_features]
                    print(f"   Wykryto wymiar modelu: {input_dim} (z klucza: {key})")
                    break
            
            if input_dim is None:
                # Fallback - znajdź pierwszą warstwę Linear
                for key, param in state_dict.items():
                    if 'weight' in key and len(param.shape) == 2:
                        input_dim = param.shape[1]
                        print(f"   Wykryto wymiar modelu: {input_dim} (z klucza: {key})")
                        break
            
            if input_dim is None:
                raise ValueError("Nie można wykryć wymiaru wejściowego modelu z checkpoint!")
        
        # Utwórz model o odpowiednim rozmiarze
        self.model = create_model(input_dim)
        
        # Załaduj wagi
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"OK Recommender gotowy!")
        print(f"   Filmy uzytkownika: {len(self.movies_df)}")
    
    def _fetch_movie_from_db(self, movie_id: int) -> Optional[pd.Series]:
        """
        Pobiera dane filmu z bazy danych.
        
        Args:
            movie_id: ID filmu w TMDB
            
        Returns:
            Series z danymi filmu lub None jeśli nie znaleziono
        """
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            m.id as tmdb_id,
            m.title,
            m.year,
            m.rating as tmdb_rating,
            m.popularity as tmdb_popularity,
            m.type as tmdb_type,
            GROUP_CONCAT(DISTINCT g.name) as genres,
            (SELECT d.name 
             FROM movie_directors md2 
             JOIN directors d ON md2.director_id = d.id 
             WHERE md2.movie_id = m.id 
             LIMIT 1) as director,
            (SELECT GROUP_CONCAT(a.name)
             FROM movie_actors ma2
             JOIN actors a ON ma2.actor_id = a.id
             WHERE ma2.movie_id = m.id
             ORDER BY ma2.cast_order
             LIMIT 5) as actors
        FROM movies m
        LEFT JOIN movie_genres mg ON m.id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE m.id = ?
        GROUP BY m.id
        """
        
        result = pd.read_sql_query(query, conn, params=(movie_id,))
        conn.close()
        
        if result.empty:
            return None
        
        # Konwertuj stringi na listy i upewnij się że wszystkie wartości są poprawne
        row = result.iloc[0].copy()
        
        # Gatunki
        if pd.notna(row['genres']) and row['genres']:
            row['genres'] = str([g.strip() for g in row['genres'].split(',')])
        else:
            row['genres'] = '[]'
        
        # Aktorzy
        if pd.notna(row['actors']) and row['actors']:
            row['actors'] = str([a.strip() for a in row['actors'].split(',')])
        else:
            row['actors'] = '[]'
        
        # Reżyser - upewnij się że nie jest None
        if pd.isna(row['director']) or not row['director']:
            row['director'] = 'Unknown'
        
        # Upewnij się że numeryczne wartości nie są None
        if pd.isna(row.get('tmdb_rating')):
            row['tmdb_rating'] = 0.0
        if pd.isna(row.get('tmdb_popularity')):
            row['tmdb_popularity'] = 0.0
        if pd.isna(row.get('year')):
            row['year'] = 2000
        
        return row
    
    def _create_features(self, movie_row: pd.Series, debug=False) -> np.ndarray:
        """
        Tworzy wektor cech dla pojedynczego filmu.
        
        Args:
            movie_row: Wiersz z DataFrame zawierający dane filmu
            debug: Czy wyświetlać informacje debugowania
            
        Returns:
            Wektor cech (177 wymiarów)
        """
        features = []
        
        # 1. Cechy numeryczne (3)
        rating = movie_row.get('tmdb_rating', movie_row.get('rating', movie_row.get('vote_average', 0)))
        popularity = movie_row.get('tmdb_popularity', movie_row.get('popularity', 0))
        year = movie_row.get('tmdb_year', movie_row.get('year', 2000))
        
        numerical = np.array([rating, popularity, year]).reshape(1, -1)
        numerical_scaled = self.scaler.transform(numerical)[0]
        features.extend(numerical_scaled)
        
        if debug:
            print(f"  Numerical: rating={rating}, popularity={popularity}, year={year}")
            print(f"  Scaled: {numerical_scaled}")
        
        # 2. Gatunki (21) - multi-hot encoding
        try:
            if isinstance(movie_row['genres'], str):
                genres = eval(movie_row['genres'])
            elif isinstance(movie_row['genres'], list):
                genres = movie_row['genres']
            else:
                genres = []
        except:
            genres = []
        
        # Filtruj tylko znane gatunki aby uniknąć ostrzeżeń
        known_genres = [g for g in genres if g in self.genre_encoder.classes_]
        genre_vector = self.genre_encoder.transform([known_genres])[0]
        features.extend(genre_vector)
        
        if debug:
            print(f"  Genres: {genres[:5]} -> Known: {known_genres}")
        
        # 3. Reżyser - one-hot encoding (dynamiczny wymiar!)
        director = movie_row['director']
        director_vector = np.zeros(self.n_directors)
        if pd.notna(director) and director in self.top_directors:
            idx = self.top_directors.index(director)
            director_vector[idx] = 1
        else:
            director_vector[-1] = 1  # "other"
        features.extend(director_vector)
        
        # 4. Aktorzy - binary encoding (dynamiczny wymiar!)
        try:
            if isinstance(movie_row['actors'], str):
                actors = eval(movie_row['actors'])
            elif isinstance(movie_row['actors'], list):
                actors = movie_row['actors']
            else:
                actors = []
        except:
            actors = []
        
        actor_vector = np.zeros(self.n_actors)
        for actor in actors[:5]:  # Top 5 actors z filmu
            if actor in self.actor_to_idx:
                idx = self.actor_to_idx[actor]
                if idx < self.n_actors:
                    actor_vector[idx] = 1
        features.extend(actor_vector)
        
        if debug:
            print(f"  Actors: {actors[:3]} -> Vector non-zero: {actor_vector.sum()}")
        
        # 5. Typ - one-hot encoding (2 kolumny jak pd.get_dummies!)
        movie_type = movie_row.get('tmdb_type', movie_row.get('type', 'movie'))
        # pd.get_dummies tworzy 2 kolumny: type_movie i type_tv
        type_movie = 1 if movie_type == 'movie' else 0
        type_tv = 1 if movie_type == 'tv' else 0
        features.append(type_movie)
        features.append(type_tv)
        
        if debug:
            print(f"  Type: {movie_type} -> [movie={type_movie}, tv={type_tv}]")
        
        return np.array(features, dtype=np.float32)
    
    def predict_rating(self, movie_id: int) -> float:
        """
        Przewiduje ocenę dla pojedynczego filmu.
        
        Args:
            movie_id: ID filmu w TMDB
            
        Returns:
            Przewidywana ocena (0-5 w skali Letterboxd)
        """
        # Szukaj w filmach użytkownika
        movie = self.movies_df[self.movies_df['tmdb_id'] == movie_id]
        
        if movie.empty:
            # Jeśli nie ma w filmach użytkownika, pobierz z bazy
            movie_data = self._fetch_movie_from_db(movie_id)
            if movie_data is None:
                raise ValueError(f"Film {movie_id} nie został znaleziony")
        else:
            movie_data = movie.iloc[0]
        
        # Utwórz cechy
        features = self._create_features(movie_data)
        
        # Predykcja
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            prediction = self.model(X).item()
        
        # Ogranicz do 0-5 (bez zaokrąglania)
        prediction = max(0.0, min(5.0, prediction))
        
        return prediction
    
    def get_top_recommendations(
        self,
        watched_movie_ids: List[int],
        n: int = 10,
        min_rating: Optional[float] = None,
        genres: Optional[List[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        movie_type: Optional[str] = None,
        min_popularity: float = 5.0
    ) -> pd.DataFrame:
        """
        Generuje top rekomendacje na podstawie preferencji użytkownika.
        
        Pobiera nieobejrzane filmy z bazy danych i przewiduje oceny dla każdego.
        Zwraca filmy z najwyższymi przewidywanymi ocenami.
        
        Args:
            watched_movie_ids: Lista ID obejrzanych filmów (do wykluczenia)
            n: Liczba rekomendacji
            min_rating: Minimalna przewidywana ocena (0-5)
            genres: Lista gatunków do filtrowania
            year_min: Minimalny rok produkcji
            year_max: Maksymalny rok produkcji
            movie_type: 'movie' lub 'tv'
            min_popularity: Minimalna popularność w TMDB (domyślnie 5.0)
            
        Returns:
            DataFrame z rekomendacjami (posortowane malejąco według przewidywanej oceny)
        """
        import sqlite3
        
        print(f"Generuje top {n} rekomendacji z bazy danych...")
        
        # Buduj zapytanie SQL
        conn = sqlite3.connect(self.db_path)
        
        where_clauses = [f"m.id NOT IN ({','.join(map(str, watched_movie_ids))})"]
        where_clauses.append(f"m.popularity >= {min_popularity}")
        
        if year_min:
            where_clauses.append(f"m.year >= {year_min}")
        
        if year_max:
            where_clauses.append(f"m.year <= {year_max}")
        
        if movie_type:
            where_clauses.append(f"m.type = '{movie_type}'")
        
        genre_join = ""
        if genres:
            genre_filter = "', '".join(genres)
            where_clauses.append(f"g.name IN ('{genre_filter}')")
            genre_join = """
            INNER JOIN movie_genres mg ON m.id = mg.movie_id
            INNER JOIN genres g ON mg.genre_id = g.id
            """
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT DISTINCT
            m.id as tmdb_id,
            m.title,
            m.year,
            m.rating as tmdb_rating,
            m.popularity as tmdb_popularity,
            m.type as tmdb_type,
            (SELECT GROUP_CONCAT(DISTINCT g2.name)
             FROM movie_genres mg2
             JOIN genres g2 ON mg2.genre_id = g2.id
             WHERE mg2.movie_id = m.id) as genres,
            (SELECT d.name 
             FROM movie_directors md2 
             JOIN directors d ON md2.director_id = d.id 
             WHERE md2.movie_id = m.id 
             LIMIT 1) as director,
            (SELECT GROUP_CONCAT(a.name)
             FROM movie_actors ma2
             JOIN actors a ON ma2.actor_id = a.id
             WHERE ma2.movie_id = m.id
             ORDER BY ma2.cast_order
             LIMIT 5) as actors
        FROM movies m
        {genre_join}
        WHERE {where_clause}
        ORDER BY m.popularity DESC
        LIMIT 1000
        """
        
        candidates = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"   Pobrано {len(candidates)} kandydatów z bazy")
        
        if candidates.empty:
            print("   Brak kandydatów spełniających kryteria")
            return pd.DataFrame()
        
        # Konwertuj stringi na listy
        for idx, row in candidates.iterrows():
            if pd.notna(row['genres']):
                candidates.at[idx, 'genres'] = str(row['genres'].split(','))
            else:
                candidates.at[idx, 'genres'] = '[]'
            
            if pd.notna(row['actors']):
                candidates.at[idx, 'actors'] = str(row['actors'].split(','))
            else:
                candidates.at[idx, 'actors'] = '[]'
        
        # Przewiduj oceny dla wszystkich kandydatów
        print(f"   Przewidywanie ocen...")
        predictions = []
        for idx, row in candidates.iterrows():
            try:
                features = self._create_features(row)
                with torch.no_grad():
                    X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    pred = self.model(X).item()
                    # Ogranicz do 0-5 (bez zaokrąglania)
                    pred = max(0.0, min(5.0, pred))
                predictions.append(pred)
            except Exception as e:
                predictions.append(0.0)
        
        candidates['predicted_rating'] = predictions
        
        # Filtruj po minimalnej ocenie
        if min_rating:
            candidates = candidates[candidates['predicted_rating'] >= min_rating]
        
        # Sortuj i zwróć top N
        recommendations = candidates.nlargest(n, 'predicted_rating')
        
        # Wybierz kolumny do wyświetlenia
        result = recommendations[[
            'tmdb_id', 'title', 'year', 'tmdb_type',
            'predicted_rating', 'tmdb_rating', 'tmdb_popularity',
            'genres', 'director', 'actors'
        ]].reset_index(drop=True)
        
        # Rename dla spójności
        result = result.rename(columns={'tmdb_type': 'type'})
        
        print(f"✅ Wygenerowano {len(result)} rekomendacji")
        
        return result
    
    def get_recommendations(
        self,
        watched_movie_ids: List[int],
        n: int = 10,
        min_rating: Optional[float] = None,
        genres: Optional[List[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        movie_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generuje rekomendacje filmów.
        
        Args:
            watched_movie_ids: Lista ID obejrzanych filmów (do wykluczenia)
            n: Liczba rekomendacji
            min_rating: Minimalna przewidywana ocena
            genres: Lista gatunków do filtrowania
            year_min: Minimalny rok produkcji
            year_max: Maksymalny rok produkcji
            movie_type: 'movie' lub 'tv'
            
        Returns:
            DataFrame z rekomendacjami (posortowane malejąco według przewidywanej oceny)
        """
        print(f"🎬 Generuję rekomendacje...")
        
        # Filtruj obejrzane filmy
        unwatched = self.movies_df[~self.movies_df['tmdb_id'].isin(watched_movie_ids)].copy()
        
        # Zastosuj filtry
        if genres:
            unwatched = unwatched[unwatched['genres'].apply(
                lambda x: any(g in eval(x) if isinstance(x, str) else [] for g in genres)
            )]
        
        if year_min:
            unwatched = unwatched[unwatched['year'] >= year_min]
        
        if year_max:
            unwatched = unwatched[unwatched['year'] <= year_max]
        
        if movie_type:
            unwatched = unwatched[unwatched['type'] == movie_type]
        
        print(f"   Kandydatów po filtrach: {len(unwatched)}")
        
        # Przewiduj oceny dla wszystkich kandydatów
        predictions = []
        for idx, row in unwatched.iterrows():
            try:
                features = self._create_features(row)
                with torch.no_grad():
                    X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    pred = self.model(X).item()
                predictions.append(pred)
            except Exception as e:
                print(f"   ⚠️  Błąd dla {row['title']}: {e}")
                predictions.append(0.0)
        
        unwatched['predicted_rating'] = predictions
        
        # Filtruj po minimalnej ocenie
        if min_rating:
            unwatched = unwatched[unwatched['predicted_rating'] >= min_rating]
        
        # Sortuj i zwróć top N
        recommendations = unwatched.nlargest(n, 'predicted_rating')
        
        # Wybierz kolumny do wyświetlenia
        result = recommendations[[
            'tmdb_id', 'title', 'year', 'type', 
            'predicted_rating', 'tmdb_rating', 'tmdb_popularity',
            'genres', 'director', 'actors'
        ]].reset_index(drop=True)
        
        print(f"✅ Wygenerowano {len(result)} rekomendacji")
        
        return result
    
    def get_similar_movies(
        self,
        movie_id: int,
        n: int = 10,
        watched_movie_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Znajduje filmy podobne do podanego w pełnej bazie TMDB (10k filmów).
        
        Args:
            movie_id: ID filmu referencyjnego
            n: Liczba podobnych filmów
            watched_movie_ids: Lista obejrzanych filmów do wykluczenia
            
        Returns:
            DataFrame z podobnymi filmami
        """
        import sqlite3
        
        # Znajdź film referencyjny (w filmach użytkownika lub bazie)
        reference = self.movies_df[self.movies_df['tmdb_id'] == movie_id]
        if reference.empty:
            ref_row = self._fetch_movie_from_db(movie_id)
            if ref_row is None:
                raise ValueError(f"Film {movie_id} nie został znaleziony")
        else:
            ref_row = reference.iloc[0]
        ref_genres = eval(ref_row['genres']) if isinstance(ref_row['genres'], str) else []
        ref_director = ref_row['director']
        ref_actors = eval(ref_row['actors']) if isinstance(ref_row['actors'], str) else []
        ref_year = ref_row.get('tmdb_year') if pd.notna(ref_row.get('tmdb_year')) else ref_row.get('year', 2000)
        ref_title = ref_row.get('tmdb_title') if pd.notna(ref_row.get('tmdb_title')) else ref_row.get('title', 'Unknown')
        
        print(f"Szukam filmow podobnych do: {ref_title} ({ref_year})")
        print(f"   Gatunki: {', '.join(ref_genres)}")
        print(f"   Reżyser: {ref_director}")
        
        # Pobierz kandydatów z bazy danych (wszystkie filmy z tymi samymi gatunkami)
        conn = sqlite3.connect(self.db_path)
        
        # Pobierz filmy z podobnymi gatunkami
        genre_filter = "', '".join(ref_genres) if ref_genres else ''
        
        query = f"""
        SELECT DISTINCT
            m.id as tmdb_id,
            m.title,
            m.year,
            m.rating as tmdb_rating,
            m.popularity as tmdb_popularity,
            m.type as tmdb_type,
            GROUP_CONCAT(DISTINCT g.name) as genres,
            (SELECT d.name 
             FROM movie_directors md2 
             JOIN directors d ON md2.director_id = d.id 
             WHERE md2.movie_id = m.id 
             LIMIT 1) as director,
            (SELECT GROUP_CONCAT(a.name)
             FROM movie_actors ma2
             JOIN actors a ON ma2.actor_id = a.id
             WHERE ma2.movie_id = m.id
             ORDER BY ma2.cast_order
             LIMIT 5) as actors
        FROM movies m
        LEFT JOIN movie_genres mg ON m.id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE m.id != ?
        {'AND g.name IN (\'' + genre_filter + '\')' if genre_filter else ''}
        GROUP BY m.id
        LIMIT 500
        """
        
        candidates = pd.read_sql_query(query, conn, params=(movie_id,))
        conn.close()
        
        # Konwertuj stringi na listy
        for idx, row in candidates.iterrows():
            if pd.notna(row['genres']):
                candidates.at[idx, 'genres'] = str(row['genres'].split(','))
            else:
                candidates.at[idx, 'genres'] = '[]'
            
            if pd.notna(row['actors']):
                candidates.at[idx, 'actors'] = str(row['actors'].split(','))
            else:
                candidates.at[idx, 'actors'] = '[]'
        
        if watched_movie_ids:
            candidates = candidates[~candidates['tmdb_id'].isin(watched_movie_ids)]
        
        print(f"   Kandydatów z bazy: {len(candidates)}")
        
        # Oblicz podobieństwo
        similarities = []
        for idx, row in candidates.iterrows():
            score = 0.0
            
            # Podobieństwo gatunków (40%)
            movie_genres = eval(row['genres']) if isinstance(row['genres'], str) else []
            if ref_genres and movie_genres:
                common_genres = len(set(ref_genres) & set(movie_genres))
                score += (common_genres / len(ref_genres)) * 0.4
            
            # Ten sam reżyser (30%)
            if pd.notna(ref_director) and pd.notna(row['director']):
                if ref_director == row['director']:
                    score += 0.3
            
            # Wspólni aktorzy (20%)
            movie_actors = eval(row['actors']) if isinstance(row['actors'], str) else []
            if ref_actors and movie_actors:
                common_actors = len(set(ref_actors) & set(movie_actors))
                score += (common_actors / min(len(ref_actors), 5)) * 0.2
            
            # Podobny rok (10%)
            movie_year = row.get('tmdb_year') if pd.notna(row.get('tmdb_year')) else row.get('year', 2000)
            if pd.isna(movie_year):
                movie_year = 2000
            year_diff = abs(float(movie_year) - float(ref_year))
            if year_diff <= 2:
                score += 0.1
            elif year_diff <= 5:
                score += 0.05
            
            similarities.append(score)
        
        candidates['similarity'] = similarities
        
        # Przewiduj oceny i kombinuj z podobieństwem
        predictions = []
        for idx, row in candidates.iterrows():
            try:
                features = self._create_features(row)
                with torch.no_grad():
                    X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    pred = self.model(X).item()
                predictions.append(pred)
            except:
                predictions.append(0.0)
        
        candidates['predicted_rating'] = predictions
        
        # Kombinuj podobieństwo (70%) i przewidywaną ocenę (30%)
        # Normalizuj predicted_rating z 0-5 do 0-1
        candidates['final_score'] = (
            candidates['similarity'] * 0.7 +
            (candidates['predicted_rating'] / 5.0) * 0.3
        )
        
        # Sortuj i zwróć top N
        similar = candidates.nlargest(n, 'final_score')
        
        # Wybierz kolumny - zawsze używamy nazw z bazy danych
        display_columns = [
            'tmdb_id', 'title', 'year', 'tmdb_type',
            'similarity', 'predicted_rating', 'final_score',
            'genres', 'director', 'actors'
        ]
        
        result = similar[display_columns].reset_index(drop=True)
        
        # Zmień nazwy kolumn dla spójności
        result = result.rename(columns={
            'tmdb_title': 'title',
            'tmdb_year': 'year',
            'tmdb_type': 'type'
        })
        
        print(f"✅ Znaleziono {len(result)} podobnych filmów")
        
        return result


def format_recommendations(df: pd.DataFrame, content_type: str = "FILMOW") -> None:
    """Wyświetla rekomendacje w czytelny sposób."""
    print("\n" + "="*100)
    print(f"REKOMENDACJE {content_type}")
    print("="*100)
    
    for idx, row in df.iterrows():
        print(f"\n{idx + 1}. {row['title']} ({row['year']}) [{row['type'].upper()}]")
        
        # Przewidywana ocena w skali 0-5
        pred_rating = row.get('predicted_rating', row.get('final_score', 0))
        print(f"   Przewidywana ocena: {pred_rating:.1f}/5.0")
        
        if 'tmdb_rating' in row:
            print(f"   TMDB: {row['tmdb_rating']:.1f}/10 | Popularnosc: {row['tmdb_popularity']:.1f}")
        
        genres = eval(row['genres']) if isinstance(row['genres'], str) else []
        print(f"   Gatunki: {', '.join(genres)}")
        
        if pd.notna(row['director']):
            print(f"   Rezyser: {row['director']}")
        
        actors = eval(row['actors']) if isinstance(row['actors'], str) else []
        if actors:
            print(f"   Obsada: {', '.join(actors[:3])}")
        
        if 'similarity' in row:
            print(f"   Podobienstwo: {row['similarity']*100:.0f}%")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    # Test systemu rekomendacji
    base_dir = Path(__file__).parent.parent.parent
    
    # Ścieżki
    model_path = base_dir / "checkpoints" / "best_model.pth"
    enriched_data_path = base_dir / "src" / "data" / "prepared" / "enriched_movies.csv"
    encoders_path = base_dir / "src" / "data" / "prepared" / "encoders.pkl"
    db_path = base_dir / "database" / "movies.db"
    
    print("=== Test systemu rekomendacji ===\n")
    print("Model przewiduje oceny w skali 0-5 (skala Letterboxd)\n")
    
    # Inicjalizacja
    recommender = MovieRecommender(
        model_path=str(model_path),
        enriched_data_path=str(enriched_data_path),
        encoders_path=str(encoders_path),
        db_path=str(db_path)
    )
    
    # Załaduj obejrzane filmy użytkownika
    user_ratings_path = base_dir / "src" / "data" / "matched_movies.csv"
    user_ratings = pd.read_csv(user_ratings_path)
    
    print(f"\nUzytkownik obejrzal {len(user_ratings)} filmow")
    print(f"   Najlepiej ocenione:")
    top_3 = user_ratings.nlargest(3, 'user_rating')[['tmdb_title', 'user_rating', 'tmdb_year']]
    for idx, row in top_3.iterrows():
        print(f"   - {row['tmdb_title']} ({row['tmdb_year']}): {row['user_rating']}/5.0")
    
    # Test 1: Przewidywanie dla obejrzanych filmów
    print("\n" + "="*100)
    print("TEST 1: Przewidywanie ocen dla obejrzanych filmów (sprawdzenie accuracy)")
    print("="*100)
    
    sample = user_ratings.sample(min(5, len(user_ratings)))
    
    print("\nFilm | Rzeczywista | Przewidywana | Różnica")
    print("-" * 80)
    
    for idx, row in sample.iterrows():
        try:
            pred = recommender.predict_rating(row['tmdb_id'])
            diff = abs(row['user_rating'] - pred)
            print(f"{row['tmdb_title'][:40]:40} | {row['user_rating']:5.1f} | {pred:5.1f} | {diff:5.2f}")
        except Exception as e:
            print(f"{row['tmdb_title'][:40]:40} | ERROR: {e}")
    
    # Test 2A: Top rekomendacje filmów
    print("\n" + "="*100)
    print("TEST 2A: Top 20 rekomendowanych FILMÓW")
    print("="*100)
    
    try:
        movie_recs = recommender.get_top_recommendations(
            watched_movie_ids=user_ratings['tmdb_id'].tolist(),
            n=20,
            min_rating=3.5,  # Minimum 3.5/5.0
            min_popularity=10.0,  # Popularne filmy
            movie_type='movie'  # Tylko filmy
        )
        
        if len(movie_recs) > 0:
            format_recommendations(movie_recs, "FILMOW")
        else:
            print("   Brak rekomendacji spełniających kryteria")
            
    except Exception as e:
        import traceback
        print(f"   ❌ Błąd: {e}")
        traceback.print_exc()
    
    # Test 2B: Top rekomendacje seriali
    print("\n" + "="*100)
    print("TEST 2B: Top 20 rekomendowanych SERIALI")
    print("="*100)
    
    try:
        tv_recs = recommender.get_top_recommendations(
            watched_movie_ids=user_ratings['tmdb_id'].tolist(),
            n=20,
            min_rating=3.5,  # Minimum 3.5/5.0
            min_popularity=10.0,  # Popularne seriale
            movie_type='tv'  # Tylko seriale
        )
        
        if len(tv_recs) > 0:
            format_recommendations(tv_recs, "SERIALI")
        else:
            print("   Brak rekomendacji spełniających kryteria")
            
    except Exception as e:
        import traceback
        print(f"   ❌ Błąd: {e}")
        traceback.print_exc()
    
    print("\n✅ Test zakończony!")


