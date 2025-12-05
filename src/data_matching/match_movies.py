"""
Skrypt dopasowujący filmy użytkownika z Letterboxd do ID w TMDB (wersja On-Demand).

1. Wczytuje oceny z pliku ratings.csv.
2. Dla każdego filmu/serialu, wyszukuje go w API TMDB po tytule i roku.
3. Zapisuje mapowanie filmu użytkownika na znalezione ID w TMDB.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import time

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "database_user"))
sys.path.append(str(project_root / "database"))

from letterboxd_parser import LetterboxdParser
from tmdb_client import TMDBClient

def match_user_movies_to_tmdb(
    letterboxd_folder: str,
    tmdb_client: TMDBClient
):
    """
    Dopasowuje filmy użytkownika z Letterboxd do TMDB używając API.
    """
    print("🔍 Rozpoczynam dopasowywanie filmów (On-Demand)...")
    
    parser = LetterboxdParser(letterboxd_folder)
    user_ratings = parser.load_ratings()
    
    print(f"✅ Załadowano {len(user_ratings)} ocen użytkownika. Rozpoczynam wyszukiwanie w API...")
    
    matched_movies = []
    unmatched_movies = []
    start_time = time.time()

    for i, (_, row) in enumerate(user_ratings.iterrows()):
        user_title = row['Name']
        user_year = row['Year'] if pd.notna(row['Year']) else None
        
        # Heurystyka: Najpierw szukaj jako film, potem jako serial
        found_match = None
        
        # Wyszukaj jako film
        search_results_movie = tmdb_client.search_movie(query=user_title, year=user_year)
        if search_results_movie.get('results'):
            found_match = search_results_movie['results'][0]
            media_type = 'movie'
        
        # Jeśli nie znaleziono filmu, spróbuj jako serial
        if not found_match:
            search_results_tv = tmdb_client.search_tv(query=user_title, year=user_year)
            if search_results_tv.get('results'):
                found_match = search_results_tv['results'][0]
                media_type = 'tv'

        if found_match:
            matched_movies.append({
                'user_title': user_title,
                'user_year': user_year,
                'user_rating': row['Rating'],
                'tmdb_movie_id': found_match['id'],
                'tmdb_title': found_match.get('title') or found_match.get('name'),
                'tmdb_year': int((found_match.get('release_date') or found_match.get('first_air_date', '0'))[:4]),
                'tmdb_rating': found_match.get('vote_average'),
                'tmdb_popularity': found_match.get('popularity'),
                'tmdb_type': media_type,
            })
        else:
            unmatched_movies.append({'user_title': user_title, 'user_year': user_year})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = len(user_ratings) - (i + 1)
            eta = remaining / rate if rate > 0 else 0
            print(f"   [{i+1}/{len(user_ratings)}] Dopasowano... (ETA: {eta:.0f}s)")

    matched_df = pd.DataFrame(matched_movies)
    unmatched_df = pd.DataFrame(unmatched_movies)
    
    print(f"\n📊 Wyniki dopasowania:")
    print(f"  ✅ Dopasowane: {len(matched_df)} filmów ({len(matched_df)/len(user_ratings)*100:.1f}%)")
    print(f"  ❌ Niedopasowane: {len(unmatched_df)} filmów ({len(unmatched_df)/len(user_ratings)*100:.1f}%)")
    
    return matched_df, unmatched_df

def save_matched_movies(matched_df: pd.DataFrame, output_path: str):
    """Zapisuje dopasowane filmy do CSV."""
    matched_df.to_csv(output_path, index=False)
    print(f"\n💾 Zapisano dopasowane filmy do: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match Letterboxd ratings to TMDB using the API.")
    parser.add_argument("letterboxd_folder", type=str, help="Path to the directory with Letterboxd CSV files.")
    
    args = parser.parse_args()
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    
    if not api_key:
        print("❌ Brak klucza API! Ustaw TMDB_API_KEY w pliku .env")
    else:
        client = TMDBClient(api_key)
        matched_df, unmatched_df = match_user_movies_to_tmdb(args.letterboxd_folder, client)
        
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        
        if not matched_df.empty:
            save_matched_movies(matched_df, str(output_dir / "matched_movies.csv"))
        if not unmatched_df.empty:
            unmatched_df.to_csv(str(output_dir / "unmatched_movies.csv"), index=False)