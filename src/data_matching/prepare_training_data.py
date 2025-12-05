"""
Moduł do przygotowywania danych treningowych na podstawie dopasowanych filmów użytkownika.

Wersja "On-Demand":
1. Wczytuje dopasowane filmy użytkownika.
2. Wzbogaca je o szczegółowe metadane (gatunki, reżyserzy, aktorzy) POBIERANE NA ŻYWO Z API TMDB.
3. Tworzy i zapisuje encodery (scaler, binarizery) specyficzne dla danych użytkownika.
4. Tworzy wektory cech (X) i etykiety (y).
5. Skaluje etykiety (oceny użytkownika) do zakresu [0, 1].
6. Dzieli dane na zbiór treningowy i walidacyjny.
7. Zapisuje gotowe dane w formacie .npy oraz .csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from collections import Counter
import sys

# Dodaj ścieżkę, aby import TMDBClient działał
sys.path.append(str(Path(__file__).parent.parent.parent / "database"))
from tmdb_client import TMDBClient

class DataPreparer:
    """Klasa do przygotowywania danych treningowych dla modelu w trybie On-Demand."""
    
    def __init__(self, matched_movies_path: str, tmdb_client: TMDBClient):
        """
        Args:
            matched_movies_path: Ścieżka do matched_movies.csv
            tmdb_client: Instancja klienta TMDB API.
        """
        self.matched_movies_df = pd.read_csv(matched_movies_path)
        self.client = tmdb_client
        self.enriched_df = None
        self.encoders = {}
        print(f"✅ DataPreparer (On-Demand) zainicjalizowany z {len(self.matched_movies_df)} filmami użytkownika.")

    def _fetch_movie_details_api(self, row: pd.Series) -> dict:
        """Pobiera szczegóły jednego filmu/serialu z API TMDB."""
        tmdb_id = row['tmdb_movie_id']  # Używamy prawdziwego ID z TMDB
        media_type = row['tmdb_type']
        
        details, credits = None, None
        
        if media_type == 'movie':
            details = self.client.get_movie_details(tmdb_id)
            credits = self.client.get_movie_credits(tmdb_id)
        elif media_type == 'tv':
            details = self.client.get_tv_details(tmdb_id)
            credits = self.client.get_tv_credits(tmdb_id)

        if not details or not credits:
            return {'genres': [], 'directors': [], 'actors': []}

        genres = [g['name'] for g in details.get('genres', [])]
        directors = [c['name'] for c in credits.get('crew', []) if c.get('job') == 'Director']
        actors = [c['name'] for c in credits.get('cast', [])]
        
        return {'genres': genres, 'directors': directors, 'actors': actors}

    def enrich_data(self):
        """Wzbogaca dane o filmy użytkownika o szczegóły z API TMDB."""
        print(" Wzbogacanie danych z API TMDB (może potrwać 2-3 minuty)...")
        enriched_rows = []
        total_movies = len(self.matched_movies_df)
        start_time = time.time()

        for i, (idx, row) in enumerate(self.matched_movies_df.iterrows()):
            details = self._fetch_movie_details_api(row)
            new_row = {**row.to_dict(), **details}
            enriched_rows.append(new_row)
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = total_movies - (i + 1)
                eta = remaining / rate if rate > 0 else 0
                print(f"   [{i+1}/{total_movies}] Wzbogacono... (ETA: {eta:.0f}s)")

        self.enriched_df = pd.DataFrame(enriched_rows)
        self.enriched_df['genres'] = self.enriched_df['genres'].apply(lambda x: x if x else np.nan)
        self.enriched_df.dropna(subset=['genres'], inplace=True)
        print(f"✅ Wzbogacono dane. Liczba filmów po czyszczeniu: {len(self.enriched_df)}")

    def prepare_features_and_labels(self):
        """Tworzy wektory cech (X) i etykiety (y) oraz zapisuje encodery."""
        if self.enriched_df is None:
            self.enrich_data()
        
        print(" Przygotowywanie cech i etykiet...")
        
        numerical_features = self.enriched_df[['tmdb_rating', 'tmdb_popularity', 'tmdb_year']]
        scaler = MinMaxScaler()
        scaler.fit(numerical_features.values)
        self.encoders['scaler'] = scaler

        mlb_genres = MultiLabelBinarizer()
        mlb_genres.fit(self.enriched_df['genres'])
        self.encoders['mlb_genres'] = mlb_genres

        director_counts = Counter([d for directors in self.enriched_df['directors'] if directors for d in directors])
        top_directors = [d for d, count in director_counts.most_common(50)]
        self.encoders['top_directors'] = top_directors

        actor_counts = Counter([a for actors in self.enriched_df['actors'] if actors for a in actors])
        top_actors = [a for a, count in actor_counts.most_common(100)]
        self.encoders['top_actors'] = top_actors
        self.encoders['actor_to_idx'] = {actor: i for i, actor in enumerate(top_actors)}

        print(f"✅ Utworzono encodery (gatunki: {len(mlb_genres.classes_)}, reżyserzy: {len(top_directors)}, aktorzy: {len(top_actors)})")

        feature_vectors = []
        for _, row in self.enriched_df.iterrows():
            features = []
            num_vals = row[['tmdb_rating', 'tmdb_popularity', 'tmdb_year']].values.reshape(1, -1)
            features.extend(scaler.transform(num_vals)[0])
            features.extend(mlb_genres.transform([row['genres']])[0])

            director_vec = np.zeros(len(top_directors) + 1)
            director_found = False
            for director in row.get('directors', []):
                if director in top_directors:
                    director_vec[top_directors.index(director)] = 1
                    director_found = True
                    break
            if not director_found:
                director_vec[-1] = 1
            features.extend(director_vec)

            actor_vec = np.zeros(len(top_actors))
            actor_to_idx = self.encoders['actor_to_idx']
            for actor in row.get('actors', []):
                if actor in actor_to_idx:
                    actor_vec[actor_to_idx[actor]] = 1
            features.extend(actor_vec)

            features.extend([1 if row['tmdb_type'] == 'movie' else 0, 1 if row['tmdb_type'] == 'tv' else 0])
            feature_vectors.append(features)
        
        X = np.array(feature_vectors, dtype=np.float32)

        user_ratings = self.enriched_df['user_rating'].values.reshape(-1, 1)
        rating_scaler = MinMaxScaler(feature_range=(0, 1))
        rating_scaler.fit(np.array([[0.5], [5.0]]))
        y = rating_scaler.transform(user_ratings).flatten()
        self.encoders['rating_scaler'] = rating_scaler
        
        print(f"✅ Utworzono wektory cech (X) i etykiety (y). Kształt X: {X.shape}")
        return X, y

    def save_prepared_data(self, output_dir: str):
        """Orkiestruje całym procesem i zapisuje wyniki."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        X, y = self.prepare_features_and_labels()
        
        self.enriched_df.to_csv(output_path / "enriched_movies.csv", index=False)
        
        with open(output_path / "encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=(self.enriched_df['user_rating'] > 3.5)
        )
        print(f"✅ Podzielono dane: {len(X_train)} treningowych, {len(X_test)} walidacyjnych")

        np.save(output_path / "X_train.npy", X_train)
        np.save(output_path / "X_test.npy", X_test)
        np.save(output_path / "y_train.npy", y_train)
        np.save(output_path / "y_test.npy", y_test)

        print(f"✅ Wszystkie pliki wynikowe zapisano w folderze: {output_path}")

    def close(self):
        """Placeholder - nie ma już połączenia do zamknięcia."""
        pass

if __name__ == '__main__':
    print("🧪 Testowanie DataPreparer w trybie On-Demand...")
    import os
    from dotenv import load_dotenv
    
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / '.env')
    api_key = os.getenv("TMDB_API_KEY")

    if not api_key:
        print("❌ Brak klucza API! Ustaw TMDB_API_KEY w pliku .env")
    else:
        matched_csv_path = project_root / "src/data/matched_movies.csv"
        if not matched_csv_path.exists():
            print(f"❌ Plik {matched_csv_path} nie istnieje. Uruchom najpierw match_movies.py")
        else:
            output_dir = project_root / "src/data/prepared"
            client = TMDBClient(api_key)
            preparer = DataPreparer(str(matched_csv_path), client)
            preparer.save_prepared_data(str(output_dir))
            
            print("\n🎉 Test zakończony pomyślnie!")