"""
System rekomendacji wykorzystujący wytrenowany model (wersja On-Demand).

- Ładuje wytrenowany model i encodery specyficzne dla użytkownika.
- Generuje kandydatów do rekomendacji z wielu źródeł API TMDB (popularne, oceniane, podobne).
- Dla każdego kandydata pobiera na żywo jego pełne cechy.
- Tworzy wektory cech i przepuszcza je przez model w celu uzyskania "Wyniku dopasowania".
- Zwraca posortowaną listę najlepszych rekomendacji.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Optional, Set
import sys
import time

# Dodaj ścieżki do importów
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "database"))
from model import create_model
from tmdb_client import TMDBClient


class MovieRecommender:
    """System rekomendacji filmów w trybie On-Demand."""
    
    def __init__(
        self,
        model_path: str,
        user_movies_path: str,
        encoders_path: str,
        tmdb_client: TMDBClient,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.client = tmdb_client
        
        print("Laduje dane i encodery użytkownika...")
        self.user_movies_df = pd.read_csv(user_movies_path)
        
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
        
        print("Laduje model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'input_dim' not in checkpoint:
            raise ValueError("Checkpoint modelu jest niekompatybilny (brak 'input_dim').")
        
        input_dim = checkpoint['input_dim']
        print(f"   Wykryto wymiar modelu: {input_dim} (z metadanych)")
        
        self.model = create_model(input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"OK Recommender gotowy!")

    def _create_features_from_api(self, api_details: dict, api_credits: dict) -> Optional[np.ndarray]:
        """Tworzy wektor cech dla filmu na podstawie danych z API."""
        try:
            features = []
            
            # 1. Cechy numeryczne
            rating = api_details.get('vote_average', 0)
            popularity = api_details.get('popularity', 0)
            year_str = api_details.get('release_date') or api_details.get('first_air_date', '')
            year = int(year_str[:4]) if year_str else 2000
            
            num_vals = np.array([[rating, popularity, year]])
            features.extend(self.scaler.transform(num_vals)[0])
            
            # 2. Gatunki
            genres = [g['name'] for g in api_details.get('genres', [])]
            known_genres = [g for g in genres if g in self.genre_encoder.classes_]
            features.extend(self.genre_encoder.transform([known_genres])[0])
            
            # 3. Reżyser
            directors = [c['name'] for c in api_credits.get('crew', []) if c.get('job') == 'Director']
            director_vec = np.zeros(self.n_directors)
            director_found = False
            for director in directors:
                if director in self.top_directors:
                    director_vec[self.top_directors.index(director)] = 1
                    director_found = True
                    break
            if not director_found:
                director_vec[-1] = 1
            features.extend(director_vec)
            
            # 4. Aktorzy
            actors = [c['name'] for c in api_credits.get('cast', [])]
            actor_vec = np.zeros(self.n_actors)
            for actor in actors[:10]: # Bierzemy top 10 aktorów z obsady filmu
                if actor in self.actor_to_idx:
                    actor_vec[self.actor_to_idx[actor]] = 1
            features.extend(actor_vec)
            
            # 5. Typ
            is_movie = 1 if 'release_date' in api_details else 0
            features.extend([is_movie, 1 - is_movie])
            
            return np.array(features, dtype=np.float32)

        except Exception as e:
            # print(f"Błąd przy tworzeniu cech dla filmu: {e}")
            return None

    def _get_candidate_ids(self, watched_movie_ids: Set[int], media_type: str) -> Set[int]:
        """Pobiera ID kandydatów do rekomendacji z różnych źródeł API."""
        candidate_ids = set()
        
        print(f"\nPobieram kandydatów dla '{media_type}'...")
        
        # Źródło 1: Filmy popularne
        print("   -> Źródło 1/3: Filmy popularne...")
        discover_func = self.client.discover_movies if media_type == 'movie' else self.client.discover_tv
        for page in range(1, 4): # 3 strony = 60 filmów
            res = discover_func(page=page, sort_by='popularity.desc')
            candidate_ids.update(r['id'] for r in res.get('results', []))
        
        # Źródło 2: Filmy najwyżej oceniane
        print("   -> Źródło 2/3: Filmy najwyżej oceniane...")
        for page in range(1, 10):
            res = discover_func(page=page, sort_by='vote_average.desc', min_vote_count=500)
            candidate_ids.update(r['id'] for r in res.get('results', []))

        # Źródło 3: Filmy podobne do ulubionych
        print("   -> Źródło 3/3: Filmy podobne do ulubionych...")
        favorite_movies = self.user_movies_df[self.user_movies_df['tmdb_type'] == media_type].nlargest(5, 'user_rating')
        similar_func = self.client.get_movie_similar if media_type == 'movie' else self.client.get_tv_similar
        
        for _, row in favorite_movies.iterrows():
            res = similar_func(row['tmdb_movie_id'])
            candidate_ids.update(r['id'] for r in res.get('results', [])[:5]) # Top 5 podobnych

        print(f"   Zebrano {len(candidate_ids)} unikalnych kandydatów.")
        
        # Odrzuć już obejrzane
        final_candidates = candidate_ids - watched_movie_ids
        print(f"   Po odrzuceniu obejrzanych, zostało {len(final_candidates)} kandydatów.")
        return final_candidates

    def get_top_recommendations(
        self,
        watched_movie_ids: List[int],
        n: int = 10,
        movie_type: str = 'movie'
    ) -> pd.DataFrame:
        
        candidate_ids = self._get_candidate_ids(set(watched_movie_ids), movie_type)
        
        print(f"\nPrzetwarzam {len(candidate_ids)} kandydatów, aby wygenerować top {n} rekomendacji...")
        
        recommendations_data = []
        total_candidates = len(candidate_ids)
        start_time = time.time()

        details_func = self.client.get_movie_details if movie_type == 'movie' else self.client.get_tv_details
        credits_func = self.client.get_movie_credits if movie_type == 'movie' else self.client.get_tv_credits

        for i, tmdb_id in enumerate(candidate_ids):
            details = details_func(tmdb_id)
            credits = credits_func(tmdb_id)
            
            if not details or not credits: continue

            features = self._create_features_from_api(details, credits)
            if features is None: continue
            
            with torch.no_grad():
                X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                score = self.model(X).item()
            
            year_str = details.get('release_date') or details.get('first_air_date', '')
            recommendations_data.append({
                'tmdb_id': tmdb_id,
                'title': details.get('title') or details.get('name'),
                'year': int(year_str[:4]) if year_str else 'N/A',
                'type': movie_type,
                'match_score': score,
                'tmdb_rating': details.get('vote_average', 0),
                'tmdb_popularity': details.get('popularity', 0),
                'genres': [g['name'] for g in details.get('genres', [])],
                'director': next((c['name'] for c in credits.get('crew', []) if c.get('job') == 'Director'), None),
                'actors': [c['name'] for c in credits.get('cast', [])[:5]]
            })

            if (i + 1) % 50 == 0:
                 elapsed = time.time() - start_time
                 rate = (i + 1) / elapsed
                 remaining = total_candidates - (i + 1)
                 eta = remaining / rate if rate > 0 else 0
                 print(f"   [{i+1}/{total_candidates}] Oceniono... (ETA: {eta:.0f}s)")
        
        if not recommendations_data:
            print("   Nie udało się wygenerować żadnych rekomendacji.")
            return pd.DataFrame()
            
        recommendations_df = pd.DataFrame(recommendations_data)
        result_df = recommendations_df.nlargest(n, 'match_score')
        
        print(f"✅ Wygenerowano {len(result_df)} rekomendacji")
        return result_df.reset_index(drop=True)

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
        
        genres = row.get('genres', [])
        print(f"   Gatunki: {', '.join(genres)}")
        
        if pd.notna(row['director']):
            print(f"   Rezyser: {row['director']}")
        
        actors = row.get('actors', [])
        if actors:
            print(f"   Obsada: {', '.join(actors)}")
    
    print("\n" + "="*100)
