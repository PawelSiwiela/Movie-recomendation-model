"""
Moduł do przygotowywania danych treningowych na podstawie dopasowanych filmów użytkownika.

1. Wczytuje dopasowane filmy użytkownika.
2. Wzbogaca je o szczegółowe metadane (gatunki, reżyserzy, aktorzy) z bazy TMDB.
3. Tworzy i zapisuje encodery (scaler, binarizery) specyficzne dla danych użytkownika.
4. Tworzy wektory cech (X) i etykiety (y).
5. Skaluje etykiety (oceny użytkownika) do zakresu [0, 1].
6. Dzieli dane na zbiór treningowy i walidacyjny.
7. Zapisuje gotowe dane w formacie .npy oraz .csv.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from collections import Counter

class DataPreparer:
    """Klasa do przygotowywania danych treningowych dla modelu."""
    
    def __init__(self, matched_movies_path: str, db_path: str):
        """
        Args:
            matched_movies_path: Ścieżka do matched_movies.csv
            db_path: Ścieżka do bazy TMDB (movies.db)
        """
        self.matched_movies_df = pd.read_csv(matched_movies_path)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.enriched_df = None
        self.encoders = {}
        print(f"✅ DataPreparer zainicjalizowany z {len(self.matched_movies_df)} filmami użytkownika.")

    def _fetch_movie_details(self, tmdb_id: int) -> dict:
        """Pobiera szczegóły jednego filmu z bazy."""
        
        # Pobieranie gatunków
        genres_q = "SELECT g.name FROM genres g JOIN movie_genres mg ON g.id = mg.genre_id WHERE mg.movie_id = ?"
        genres = pd.read_sql_query(genres_q, self.conn, params=(tmdb_id,)).name.tolist()

        # Pobieranie reżyserów
        directors_q = "SELECT d.name FROM directors d JOIN movie_directors md ON d.id = md.director_id WHERE md.movie_id = ?"
        directors = pd.read_sql_query(directors_q, self.conn, params=(tmdb_id,)).name.tolist()

        # Pobieranie aktorów
        actors_q = "SELECT a.name FROM actors a JOIN movie_actors ma ON a.id = ma.actor_id WHERE ma.movie_id = ? ORDER BY ma.cast_order"
        actors = pd.read_sql_query(actors_q, self.conn, params=(tmdb_id,)).name.tolist()
        
        return {'genres': genres, 'directors': directors, 'actors': actors}

    def enrich_data(self):
        """Wzbogaca dane o filmy użytkownika o szczegóły z bazy TMDB."""
        print(" enriquecendo os dados...")
        enriched_rows = []
        for _, row in self.matched_movies_df.iterrows():
            details = self._fetch_movie_details(row['tmdb_id'])
            new_row = {**row.to_dict(), **details}
            enriched_rows.append(new_row)
        
        self.enriched_df = pd.DataFrame(enriched_rows)
        # Konwersja pustych list na NaN, żeby dropna działało poprawnie
        self.enriched_df['genres'] = self.enriched_df['genres'].apply(lambda x: x if x else np.nan)
        self.enriched_df.dropna(subset=['genres'], inplace=True) # Usuwamy filmy bez gatunków
        print(f"✅ Wzbogacono dane. Liczba filmów po czyszczeniu: {len(self.enriched_df)}")

    def prepare_features_and_labels(self):
        """Tworzy wektory cech (X) i etykiety (y) oraz zapisuje encodery."""
        if self.enriched_df is None:
            self.enrich_data()
        
        print(" Przygotowywanie cech i etykiet...")
        
        # --- Tworzenie enkoderów ---
        # 1. Scaler dla cech numerycznych
        numerical_features = self.enriched_df[['tmdb_rating', 'tmdb_popularity', 'tmdb_year']]
        scaler = MinMaxScaler()
        scaler.fit(numerical_features.values)
        self.encoders['scaler'] = scaler

        # 2. MultiLabelBinarizer dla gatunków
        mlb_genres = MultiLabelBinarizer()
        mlb_genres.fit(self.enriched_df['genres'])
        self.encoders['mlb_genres'] = mlb_genres

        # 3. Enkoder dla reżyserów (top N)
        director_counts = Counter([d for directors in self.enriched_df['directors'] for d in directors])
        top_directors = [d for d, count in director_counts.most_common(50)] # Top 50 reżyserów
        self.encoders['top_directors'] = top_directors

        # 4. Enkoder dla aktorów (top N)
        actor_counts = Counter([a for actors in self.enriched_df['actors'] for a in actors])
        top_actors = [a for a, count in actor_counts.most_common(100)] # Top 100 aktorów
        self.encoders['top_actors'] = top_actors
        self.encoders['actor_to_idx'] = {actor: i for i, actor in enumerate(top_actors)}

        print(f"✅ Utworzono encodery (gatunki: {len(mlb_genres.classes_)}, reżyserzy: {len(top_directors)}, aktorzy: {len(top_actors)})")

        # --- Tworzenie wektorów cech (X) ---
        feature_vectors = []
        for _, row in self.enriched_df.iterrows():
            features = []
            
            # Cechy numeryczne
            num_vals = row[['tmdb_rating', 'tmdb_popularity', 'tmdb_year']].values.reshape(1, -1)
            scaled_num = self.encoders['scaler'].transform(num_vals)[0]
            features.extend(scaled_num)
            
            # Gatunki
            genre_vec = self.encoders['mlb_genres'].transform([row['genres']])[0]
            features.extend(genre_vec)
            
            # Reżyserzy
            director_vec = np.zeros(len(self.encoders['top_directors']) + 1)
            director_found = False
            for director in row['directors']:
                if director in self.encoders['top_directors']:
                    idx = self.encoders['top_directors'].index(director)
                    director_vec[idx] = 1
                    director_found = True
                    break # Bierzemy tylko pierwszego pasującego reżysera
            if not director_found:
                director_vec[-1] = 1 # Kategoria "other"
            features.extend(director_vec)

            # Aktorzy
            actor_vec = np.zeros(len(self.encoders['top_actors']))
            for actor in row['actors']:
                if actor in self.encoders['actor_to_idx']:
                    idx = self.encoders['actor_to_idx'][actor]
                    actor_vec[idx] = 1
            features.extend(actor_vec)

            # Typ (movie/tv)
            type_movie = 1 if row['tmdb_type'] == 'movie' else 0
            type_tv = 1 if row['tmdb_type'] == 'tv' else 0
            features.extend([type_movie, type_tv])
            
            feature_vectors.append(features)
        
        X = np.array(feature_vectors, dtype=np.float32)

        # --- Przygotowanie etykiet (y) ---
        # Skalowanie ocen użytkownika [0.5, 5.0] do zakresu [0, 1]
        user_ratings = self.enriched_df['user_rating'].values.reshape(-1, 1)
        rating_scaler = MinMaxScaler(feature_range=(0, 1))
        # Skaler jest dopasowywany do stałego, teoretycznego zakresu ocen Letterboxd
        rating_scaler.fit(np.array([[0.5], [5.0]]))
        y = rating_scaler.transform(user_ratings).flatten()
        self.encoders['rating_scaler'] = rating_scaler # Zapisujemy też ten skaler
        
        print(f"✅ Utworzono wektory cech (X) i etykiety (y). Kształt X: {X.shape}")
        
        return X, y

    def save_prepared_data(self, output_dir: str):
        """Orkiestruje całym procesem i zapisuje wyniki."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        X, y = self.prepare_features_and_labels()
        
        # Zapisz wzbogacone dane
        self.enriched_df.to_csv(output_path / "enriched_movies.csv", index=False)
        
        # Zapisz encodery
        with open(output_path / "encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Podziel na zbiór treningowy i walidacyjny
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=(self.enriched_df['user_rating'] > 3.5)
        )
        print(f"✅ Podzielono dane: {len(X_train)} treningowych, {len(X_test)} walidacyjnych")

        # Zapisz dane .npy
        np.save(output_path / "X_train.npy", X_train)
        np.save(output_path / "X_test.npy", X_test)
        np.save(output_path / "y_train.npy", y_train)
        np.save(output_path / "y_test.npy", y_test)

        print(f"✅ Wszystkie pliki wynikowe zapisano w folderze: {output_path}")

    def close(self):
        """Zamyka połączenie z bazą danych."""
        self.conn.close()

if __name__ == '__main__':
    # Test działania skryptu
    print("🧪 Testowanie DataPreparer...")
    project_root = Path(__file__).parent.parent.parent
    
    # Załóżmy, że match_movies.py został już uruchomiony i stworzył ten plik
    matched_csv_path = project_root / "src/data/matched_movies.csv"
    if not matched_csv_path.exists():
        print(f"❌ Plik {matched_csv_path} nie istnieje. Uruchom najpierw match_movies.py")
    else:
        db_path = project_root / "database/movies.db"
        output_dir = project_root / "src/data/prepared"
        
        preparer = DataPreparer(str(matched_csv_path), str(db_path))
        try:
            preparer.save_prepared_data(str(output_dir))
        finally:
            preparer.close()
        
        print("\n🎉 Test zakończony pomyślnie!")
        # Weryfikacja plików
        print("Sprawdzam zapisane pliki:")
        for f in ["enriched_movies.csv", "encoders.pkl", "X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]:
            p = Path(output_dir) / f
            status = "✅" if p.exists() else "❌"
            print(f"  {status} {p.name}")
