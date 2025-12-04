"""
Przygotowanie danych treningowych dla modelu rekomendacji filmów.
Łączy dane z matched_movies.csv z metadanymi z bazy SQLite.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import pickle


class DataPreparer:
    """Przygotowuje dane treningowe."""
    
    def __init__(self, matched_csv: str, db_path: str):
        """
        Args:
            matched_csv: Ścieżka do matched_movies.csv
            db_path: Ścieżka do bazy SQLite
        """
        self.matched_csv = matched_csv
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Załaduj dopasowane filmy
        self.df = pd.read_csv(matched_csv)
        print(f"✅ Załadowano {len(self.df)} dopasowanych filmów")
    
    def get_movie_genres(self, movie_id: int) -> list:
        """Pobiera gatunki filmu."""
        query = """
            SELECT g.name
            FROM genres g
            JOIN movie_genres mg ON g.id = mg.genre_id
            WHERE mg.movie_id = ?
        """
        result = pd.read_sql_query(query, self.conn, params=(movie_id,))
        return result['name'].tolist()
    
    def get_all_genres_from_db(self) -> list:
        """Pobiera wszystkie unikalne gatunki z bazy danych."""
        query = "SELECT DISTINCT name FROM genres ORDER BY name"
        result = pd.read_sql_query(query, self.conn)
        return result['name'].tolist()
    
    def get_movie_director(self, movie_id: int) -> str:
        """Pobiera reżysera filmu."""
        query = """
            SELECT d.name
            FROM directors d
            JOIN movie_directors md ON d.id = md.director_id
            WHERE md.movie_id = ?
            LIMIT 1
        """
        result = pd.read_sql_query(query, self.conn, params=(movie_id,))
        return result['name'].iloc[0] if len(result) > 0 else None
    
    def get_movie_actors(self, movie_id: int, top_n: int = 5) -> list:
        """Pobiera top N aktorów."""
        query = """
            SELECT a.name
            FROM actors a
            JOIN movie_actors ma ON a.id = ma.actor_id
            WHERE ma.movie_id = ?
            ORDER BY ma.cast_order
            LIMIT ?
        """
        result = pd.read_sql_query(query, self.conn, params=(movie_id, top_n))
        return result['name'].tolist()
    
    def enrich_with_metadata(self):
        """Wzbogaca DataFrame o metadane z bazy."""
        print("\n🔍 Pobieram metadane z bazy...")
        
        genres_list = []
        directors_list = []
        actors_list = []
        
        for idx, row in self.df.iterrows():
            movie_id = row['tmdb_id']
            
            genres = self.get_movie_genres(movie_id)
            director = self.get_movie_director(movie_id)
            actors = self.get_movie_actors(movie_id, top_n=5)
            
            genres_list.append(genres)
            directors_list.append(director)
            actors_list.append(actors)
            
            if (idx + 1) % 100 == 0:
                print(f"   Przetworzono {idx + 1}/{len(self.df)} filmów")
        
        self.df['genres'] = genres_list
        self.df['director'] = directors_list
        self.df['actors'] = actors_list
        
        print(f"✅ Wzbogacono dane o metadane")
    
    def create_features(self):
        """Tworzy features dla modelu."""
        print("\n🔧 Tworzę features...")
        
        features = {}
        
        # 1. Numerical features
        features['tmdb_rating'] = self.df['tmdb_rating'].values
        features['tmdb_popularity'] = self.df['tmdb_popularity'].values
        features['year'] = self.df['tmdb_year'].values
        
        # 2. Gatunki (multi-hot encoding) - użyj WSZYSTKICH gatunków z bazy
        print("   Encoding gatunków...")
        all_genres = self.get_all_genres_from_db()
        mlb_genres = MultiLabelBinarizer(classes=all_genres)
        genres_encoded = mlb_genres.fit_transform(self.df['genres'])
        features['genres'] = genres_encoded
        
        print(f"   Znaleziono {len(mlb_genres.classes_)} unikalnych gatunków: {mlb_genres.classes_[:10]}...")
        
        # 3. Reżyserzy (one-hot dla najpopularniejszych, "other" dla reszty)
        print("   Encoding reżyserów...")
        director_counts = self.df['director'].value_counts()
        top_directors_list = list(director_counts.head(50).index)  # Top 50 reżyserów (lista dla kolejności)
        top_directors_set = set(top_directors_list)  # Set dla szybkiego wyszukiwania
        
        self.df['director_category'] = self.df['director'].apply(
            lambda x: x if x in top_directors_set else 'other'
        )
        directors_encoded = pd.get_dummies(self.df['director_category'], prefix='dir')
        features['directors'] = directors_encoded.values
        
        print(f"   Znaleziono {len(top_directors_list)} popularnych reżyserów + category 'other'")
        
        # 4. Aktorzy (binarny: czy dany aktor występuje)
        print("   Encoding aktorów...")
        # Zbierz wszystkich aktorów
        all_actors = []
        for actors_list in self.df['actors']:
            all_actors.extend(actors_list)
        
        actor_counts = pd.Series(all_actors).value_counts()
        top_actors = list(actor_counts.head(100).index)  # Top 100 aktorów (lista dla zachowania kolejności)
        
        actors_binary = np.zeros((len(self.df), len(top_actors)))
        actor_to_idx = {actor: i for i, actor in enumerate(top_actors)}
        
        for i, actors_list in enumerate(self.df['actors']):
            for actor in actors_list:
                if actor in actor_to_idx:
                    actors_binary[i, actor_to_idx[actor]] = 1
        
        features['actors'] = actors_binary
        
        print(f"   Znaleziono {len(top_actors)} popularnych aktorów")
        
        # 5. Type (movie vs tv) - ZAWSZE 2 kolumny niezależnie od danych
        type_encoded = pd.get_dummies(self.df['tmdb_type'], prefix='type')
        # Dodaj brakujące kolumny jeśli nie istnieją
        for col in ['type_movie', 'type_tv']:
            if col not in type_encoded.columns:
                type_encoded[col] = 0
        # Sortuj kolumny alfabetycznie dla spójności
        type_encoded = type_encoded[['type_movie', 'type_tv']]
        features['type'] = type_encoded.values
        
        # Połącz wszystkie features - wymuszenie float64 dla wszystkich
        X = np.concatenate([
            features['tmdb_rating'].reshape(-1, 1),
            features['tmdb_popularity'].reshape(-1, 1),
            features['year'].reshape(-1, 1),
            features['genres'],
            features['directors'],
            features['actors'],
            features['type']
        ], axis=1).astype(np.float64)
        
        # Target (user rating)
        y = self.df['user_rating'].values
        
        print(f"\n✅ Utworzono features:")
        print(f"   Shape: {X.shape}")
        print(f"   Features: {X.shape[1]} (numerical: 3, genres: {features['genres'].shape[1]}, "
              f"directors: {features['directors'].shape[1]}, actors: {features['actors'].shape[1]}, "
              f"type: {features['type'].shape[1]})")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Target range: {y.min():.1f} - {y.max():.1f}")
        
        # Zapisz encodery do późniejszego użycia
        self.mlb_genres = mlb_genres
        self.top_directors = top_directors_list  # Lista dla zachowania kolejności
        self.top_actors = top_actors  # Już jest listą
        self.actor_to_idx = actor_to_idx
        
        return X, y
    
    def normalize_features(self, X_train, X_test):
        """Normalizuje numerical features (pierwsze 3 kolumny)."""
        print("\n📊 Normalizuję numerical features...")
        
        scaler = StandardScaler()
        
        # Normalizuj tylko pierwsze 3 kolumny (numerical)
        X_train_numerical = X_train[:, :3]
        X_test_numerical = X_test[:, :3]
        
        X_train_numerical_scaled = scaler.fit_transform(X_train_numerical)
        X_test_numerical_scaled = scaler.transform(X_test_numerical)
        
        # Zastąp znormalizowane wartości
        X_train[:, :3] = X_train_numerical_scaled
        X_test[:, :3] = X_test_numerical_scaled
        
        self.scaler = scaler
        
        print(f"✅ Znormalizowano numerical features")
        
        return X_train, X_test
    
    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """Przygotowuje dane treningowe i testowe."""
        print("\n🎯 Przygotowuję dane treningowe...")
        
        # 1. Wzbogać o metadane
        self.enrich_with_metadata()
        
        # 2. Utwórz features
        X, y = self.create_features()
        
        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n✅ Train/test split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        # 4. Normalizacja
        X_train, X_test = self.normalize_features(X_train, X_test)
        
        return X_train, X_test, y_train, y_test
    
    def save_prepared_data(self, output_dir: str):
        """Zapisuje przygotowane dane."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Zapisuję dane do {output_dir}...")
        
        X_train, X_test, y_train, y_test = self.prepare_train_test_split()
        
        # Zapisz jako numpy arrays
        np.save(output_dir / 'X_train.npy', X_train)
        np.save(output_dir / 'X_test.npy', X_test)
        np.save(output_dir / 'y_train.npy', y_train)
        np.save(output_dir / 'y_test.npy', y_test)
        
        # Zapisz encodery i scalery
        with open(output_dir / 'encoders.pkl', 'wb') as f:
            pickle.dump({
                'mlb_genres': self.mlb_genres,
                'top_directors': self.top_directors,
                'top_actors': self.top_actors,
                'actor_to_idx': self.actor_to_idx,
                'scaler': self.scaler
            }, f)
        
        # Zapisz enriched DataFrame
        self.df.to_csv(output_dir / 'enriched_movies.csv', index=False)
        
        print(f"✅ Zapisano:")
        print(f"   - X_train.npy, X_test.npy")
        print(f"   - y_train.npy, y_test.npy")
        print(f"   - encoders.pkl")
        print(f"   - enriched_movies.csv")
    
    def close(self):
        """Zamyka połączenie z bazą."""
        self.conn.close()


if __name__ == "__main__":
    # Ścieżki
    base_dir = Path(__file__).parent.parent.parent
    matched_csv = base_dir / "src" / "data" / "matched_movies.csv"
    db_path = base_dir / "database" / "movies.db"
    output_dir = base_dir / "src" / "data" / "prepared"
    
    # Przygotuj dane
    preparer = DataPreparer(str(matched_csv), str(db_path))
    
    try:
        preparer.save_prepared_data(str(output_dir))
    finally:
        preparer.close()
    
    print("\n🎉 Gotowe!")
