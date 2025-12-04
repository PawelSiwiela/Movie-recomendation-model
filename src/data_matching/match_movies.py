"""
Skrypt dopasowujący filmy użytkownika z Letterboxd do bazy TMDB.

Łączy oceny użytkownika z danymi filmów w bazie (po tytule + roku),
żeby model wiedział które filmy użytkownik lubi.
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
from difflib import SequenceMatcher
import argparse

# Dodaj ścieżki do importów
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "database_user"))

from letterboxd_parser import LetterboxdParser


def normalize_title(title: str) -> str:
    """
    Normalizuje tytuł do porównywania (lowercase, usunięcie znaków specjalnych).
    
    Args:
        title: Oryginalny tytuł
        
    Returns:
        Znormalizowany tytuł
    """
    # Lowercase
    title = title.lower()
    
    # Usuń przedimki "the", "a", "an" z początku
    for article in [' the ', ' a ', ' an ']:
        if title.startswith(article[1:]):
            title = title[len(article)-1:]
    
    # Usuń znaki specjalne i spacje dodatkowe
    title = ''.join(c for c in title if c.isalnum() or c.isspace())
    title = ' '.join(title.split())  # Usuń wielokrotne spacje
    
    return title


def calculate_similarity(title1: str, title2: str) -> float:
    """
    Oblicza podobieństwo dwóch tytułów (0-1).
    
    Args:
        title1: Pierwszy tytuł
        title2: Drugi tytuł
        
    Returns:
        Współczynnik podobieństwa (0-1)
    """
    title1_norm = normalize_title(title1)
    title2_norm = normalize_title(title2)
    
    return SequenceMatcher(None, title1_norm, title2_norm).ratio()


def get_movie_metadata(movie_id: int, conn: sqlite3.Connection) -> dict:
    """
    Pobiera metadane filmu (reżyser, top 5 aktorów).
    
    Args:
        movie_id: ID filmu w bazie
        conn: Połączenie SQLite
        
    Returns:
        Słownik z metadanymi
    """
    # Reżyser
    director_query = """
        SELECT d.name 
        FROM directors d
        JOIN movie_directors md ON d.id = md.director_id
        WHERE md.movie_id = ?
        LIMIT 1
    """
    director = pd.read_sql_query(director_query, conn, params=(movie_id,))
    director_name = director['name'].iloc[0] if len(director) > 0 else None
    
    # Top 5 aktorów
    actors_query = """
        SELECT a.name
        FROM actors a
        JOIN movie_actors ma ON a.id = ma.actor_id
        WHERE ma.movie_id = ?
        ORDER BY ma.cast_order
        LIMIT 5
    """
    actors = pd.read_sql_query(actors_query, conn, params=(movie_id,))
    actor_names = actors['name'].tolist() if len(actors) > 0 else []
    
    return {
        'director': director_name,
        'actors': actor_names
    }


def verify_match_by_metadata(
    movie1_id: int,
    movie2_id: int,
    conn: sqlite3.Connection
) -> float:
    """
    Weryfikuje czy dwa filmy to ten sam tytuł na podstawie metadanych.
    
    Args:
        movie1_id: ID pierwszego filmu
        movie2_id: ID drugiego filmu
        conn: Połączenie SQLite
        
    Returns:
        Confidence score (0-1) - jak bardzo pasują metadane
    """
    meta1 = get_movie_metadata(movie1_id, conn)
    meta2 = get_movie_metadata(movie2_id, conn)
    
    score = 0.0
    
    # Reżyser się zgadza? (+0.5)
    if meta1['director'] and meta2['director']:
        if meta1['director'] == meta2['director']:
            score += 0.5
    
    # Aktorzy się pokrywają? (+0.1 za każdego, max 0.5)
    if meta1['actors'] and meta2['actors']:
        common_actors = set(meta1['actors']) & set(meta2['actors'])
        score += min(len(common_actors) * 0.1, 0.5)
    
    return score


def match_user_movies_to_tmdb(
    letterboxd_folder: str,
    tmdb_db_path: str,
    min_similarity: float = 0.85,
    metadata_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Dopasowuje filmy użytkownika z Letterboxd do bazy TMDB.
    
    Args:
        letterboxd_folder: Ścieżka do folderu z eksportem Letterboxd
        tmdb_db_path: Ścieżka do bazy TMDB
        min_similarity: Minimalny próg podobieństwa tytułów (0-1)
        metadata_threshold: Minimalny próg podobieństwa metadanych (0-1)
        
    Returns:
        DataFrame z dopasowanymi filmami
    """
    print("🔍 Rozpoczynam dopasowywanie filmów...")
    
    # 1. Załaduj oceny użytkownika z Letterboxd
    parser = LetterboxdParser(letterboxd_folder)
    user_ratings = parser.load_ratings()
    
    print(f"✅ Załadowano {len(user_ratings)} ocen użytkownika")
    
    # 2. Załaduj filmy i seriale z bazy TMDB
    conn = sqlite3.connect(tmdb_db_path)
    tmdb_movies = pd.read_sql_query("""
        SELECT id, tmdb_id, title, original_title, year, rating, popularity, type
        FROM movies
    """, conn)
    
    print(f"✅ Załadowano {len(tmdb_movies)} filmów/seriali z bazy TMDB")
    
    # 3. Dopasuj filmy
    matched_movies = []
    unmatched_movies = []
    
    for idx, user_movie in user_ratings.iterrows():
        user_title = user_movie['Name']
        user_year = user_movie['Year']
        user_rating = user_movie['Rating']
        
        # Filtruj filmy z tego samego roku (±1 rok tolerancji)
        year_candidates = tmdb_movies[
            (tmdb_movies['year'] >= user_year - 1) & 
            (tmdb_movies['year'] <= user_year + 1)
        ]
        
        if len(year_candidates) == 0:
            unmatched_movies.append({
                'user_title': user_title,
                'user_year': user_year,
                'reason': 'No year match'
            })
            continue
        
        # Znajdź najlepsze dopasowanie po tytule
        best_match = None
        best_similarity = 0
        
        for _, tmdb_movie in year_candidates.iterrows():
            # Sprawdź similarity z title i original_title
            sim1 = calculate_similarity(user_title, tmdb_movie['title'])
            sim2 = calculate_similarity(user_title, tmdb_movie['original_title'])
            similarity = max(sim1, sim2)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = tmdb_movie
        
        # Jeśli similarity >= threshold, uznaj za match
        if best_similarity >= min_similarity and best_match is not None:
            matched_movies.append({
                'user_title': user_title,
                'user_year': user_year,
                'user_rating': user_rating,
                'tmdb_id': best_match['id'],
                'tmdb_movie_id': best_match['tmdb_id'],
                'tmdb_title': best_match['title'],
                'tmdb_year': best_match['year'],
                'tmdb_rating': best_match['rating'],
                'tmdb_popularity': best_match['popularity'],
                'tmdb_type': best_match['type'],
                'similarity': best_similarity,
                'match_method': 'title'
            })
        # Jeśli similarity niska (0.5-0.85), użyj zaawansowanych metod
        elif 0.5 <= best_similarity < min_similarity and best_match is not None:
            # Najpierw filtruj kandydatów po similarity >= 0.5, potem weź top 5 po popularności
            filtered_candidates = []
            for _, candidate in year_candidates.iterrows():
                title_sim = max(
                    calculate_similarity(user_title, candidate['title']),
                    calculate_similarity(user_title, candidate['original_title'])
                )
                if title_sim >= 0.5:
                    candidate_copy = candidate.copy()
                    candidate_copy['title_sim'] = title_sim
                    filtered_candidates.append(candidate_copy)
            
            if not filtered_candidates:
                unmatched_movies.append({
                    'user_title': user_title,
                    'user_year': user_year,
                    'best_match': best_match['title'] if best_match is not None else 'None',
                    'similarity': best_similarity,
                    'reason': 'No candidates with similarity >= 0.5'
                })
                continue
            
            # Sortuj po popularności
            filtered_df = pd.DataFrame(filtered_candidates)
            top_candidates = filtered_df.nlargest(5, 'popularity')
            
            metadata_match = None
            best_metadata_confidence = 0
            match_reason = None
            
            for _, candidate in top_candidates.iterrows():
                # title_sim już jest obliczone
                title_sim = candidate.get('title_sim', 0)
                
                # METODA 1: Substring matching
                # Jeśli krótszy tytuł jest zawarty w dłuższym (np. "Glass Onion" w "Glass Onion: A Knives Out Mystery")
                user_normalized = normalize_title(user_title)
                candidate_normalized = normalize_title(candidate['title'])
                candidate_original_normalized = normalize_title(candidate['original_title'])
                
                is_substring = (
                    (len(user_normalized) < len(candidate_normalized) and user_normalized in candidate_normalized) or
                    (len(user_normalized) < len(candidate_original_normalized) and user_normalized in candidate_original_normalized) or
                    (len(candidate_normalized) < len(user_normalized) and candidate_normalized in user_normalized) or
                    (len(candidate_original_normalized) < len(user_normalized) and candidate_original_normalized in user_normalized)
                )
                
                if is_substring and candidate['year'] == user_year:
                    # Substring match + dokładny rok = bardzo pewny match
                    metadata_match = candidate
                    best_metadata_confidence = 0.95
                    match_reason = 'substring'
                    break
                
                # METODA 2: Weryfikacja po metadanych
                try:
                    metadata = get_movie_metadata(candidate['id'], conn)
                    
                    # Jeśli brak metadanych, pomiń tę metodę
                    if not metadata['director'] and not metadata['actors']:
                        continue
                    
                    # Sprawdź czy reżyser występuje w tytule użytkownika
                    director_in_title = False
                    if metadata['director']:
                        director_parts = metadata['director'].split()
                        if director_parts:
                            last_name = director_parts[-1].lower()
                            if last_name in user_title.lower():
                                director_in_title = True
                    
                    # Oblicz confidence score
                    confidence = 0
                    
                    # 1. Bazowa similarity (+0.3-0.4)
                    confidence += title_sim * 0.4
                    
                    # 2. Dokładny rok (+0.3), rok ±1 (+0.2)
                    if candidate['year'] == user_year:
                        confidence += 0.3
                    elif abs(candidate['year'] - user_year) == 1:
                        confidence += 0.2
                    
                    # 3. Reżyser w tytule (+0.3)
                    if director_in_title:
                        confidence += 0.3
                    
                    # Jeśli confidence >= threshold, uznajemy za match
                    if confidence >= metadata_threshold and confidence > best_metadata_confidence:
                        metadata_match = candidate
                        best_metadata_confidence = confidence
                        match_reason = 'metadata'
                
                except Exception as e:
                    continue
            
            if metadata_match is not None:
                matched_movies.append({
                    'user_title': user_title,
                    'user_year': user_year,
                    'user_rating': user_rating,
                    'tmdb_id': metadata_match['id'],
                    'tmdb_movie_id': metadata_match['tmdb_id'],
                    'tmdb_title': metadata_match['title'],
                    'tmdb_year': metadata_match['year'],
                    'tmdb_rating': metadata_match['rating'],
                    'tmdb_popularity': metadata_match['popularity'],
                    'tmdb_type': metadata_match['type'],
                    'similarity': best_metadata_confidence,
                    'match_method': match_reason
                })
            else:
                unmatched_movies.append({
                    'user_title': user_title,
                    'user_year': user_year,
                    'best_match': best_match['title'] if best_match is not None else 'None',
                    'similarity': best_similarity,
                    'reason': 'Low similarity + metadata check failed'
                })
        else:
            unmatched_movies.append({
                'user_title': user_title,
                'user_year': user_year,
                'best_match': best_match['title'] if best_match is not None else 'None',
                'similarity': best_similarity,
                'reason': 'Low similarity'
            })
    
    # 4. Zamknij połączenie
    conn.close()
    
    # 5. Podsumowanie
    matched_df = pd.DataFrame(matched_movies)
    unmatched_df = pd.DataFrame(unmatched_movies)
    
    print(f"\n📊 Wyniki dopasowania:")
    print(f"  ✅ Dopasowane: {len(matched_df)} filmów ({len(matched_df)/len(user_ratings)*100:.1f}%)")
    print(f"  ❌ Niedopasowane: {len(unmatched_df)} filmów ({len(unmatched_df)/len(user_ratings)*100:.1f}%)")
    if len(matched_df) > 0:
        print(f"\n📈 Statystyki dopasowanych:")
        print(f"  Średnia similarity: {matched_df['similarity'].mean():.3f}")
        print(f"  Minimalna similarity: {matched_df['similarity'].min():.3f}")
        print(f"  Filmy: {len(matched_df[matched_df['tmdb_type'] == 'movie'])}")
        print(f"  Seriale: {len(matched_df[matched_df['tmdb_type'] == 'tv'])}")
        print(f"  Dopasowane po tytule: {len(matched_df[matched_df['match_method'] == 'title'])}")
        print(f"  Dopasowane po metadanych: {len(matched_df[matched_df['match_method'] == 'metadata'])}")
    
    # Pokaż przykłady niedopasowanych
    if len(unmatched_df) > 0:
        print(f"\n❌ Przykłady niedopasowanych filmów:")
        print(unmatched_df.head(10))
    
    return matched_df, unmatched_df


def save_matched_movies(matched_df: pd.DataFrame, output_path: str):
    """
    Zapisuje dopasowane filmy do CSV.
    
    Args:
        matched_df: DataFrame z dopasowanymi filmami
        output_path: Ścieżka do pliku CSV
    """
    matched_df.to_csv(output_path, index=False)
    print(f"\n💾 Zapisano dopasowane filmy do: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match Letterboxd ratings to a local TMDB database.")
    parser.add_argument("letterboxd_folder", type=str, help="Path to the directory with Letterboxd CSV files.")
    parser.add_argument("--db_path", type=str, default="database/movies.db", help="Path to the TMDB SQLite database file.")
    parser.add_argument("--output_matched", type=str, default="src/data/matched_movies.csv", help="Output path for matched movies CSV.")
    parser.add_argument("--output_unmatched", type=str, default="src/data/unmatched_movies.csv", help="Output path for unmatched movies CSV.")
    parser.add_argument("--min_similarity", type=float, default=0.85, help="Minimum title similarity for a match (0.0 to 1.0).")

    args = parser.parse_args()

    # Dopasuj filmy
    matched_df, unmatched_df = match_user_movies_to_tmdb(
        letterboxd_folder=args.letterboxd_folder,
        tmdb_db_path=args.db_path,
        min_similarity=args.min_similarity
    )

    # Zapisz wyniki
    if len(matched_df) > 0:
        save_matched_movies(matched_df, args.output_matched)

        print(f"\n🎬 Przykłady dopasowanych filmów:")
        print(matched_df[['user_title', 'tmdb_title', 'user_rating', 'similarity']].head(10))

    # Opcjonalnie zapisz niedopasowane
    if len(unmatched_df) > 0:
        unmatched_df.to_csv(args.output_unmatched, index=False)
        print(f"\n💾 Niedopasowane filmy zapisane do: {args.output_unmatched}")
