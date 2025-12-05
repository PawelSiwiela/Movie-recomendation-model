"""
Główny pipeline do generowania rekomendacji w 100% w architekturze "On-Demand".

Przepływ:
1. Wczytanie klucza API i wybór użytkownika.
2. Czyszczenie starych plików tymczasowych.
3. Krok 1: Dopasowanie filmów z Letterboxd do ID z API TMDB.
4. Krok 2: Przygotowanie danych treningowych (pobieranie cech z API TMDB).
5. Krok 3: Trenowanie spersonalizowanego modelu.
6. Krok 4: Generowanie rekomendacji (pobieranie kandydatów i ich cech z API TMDB).
7. Opcjonalne czyszczenie plików po zakończeniu.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv

# Dodaj ścieżki do importów
project_root = Path(__file__).parent
sys.path.append(str(project_root / "database"))
sys.path.append(str(project_root / "database_user"))
sys.path.append(str(project_root / "src" / "data_matching"))
sys.path.append(str(project_root / "src" / "model"))

from tmdb_client import TMDBClient

def get_available_users(database_user_dir: Path) -> list[str]:
    """Pobiera listę dostępnych folderów użytkowników Letterboxd."""
    users = [d.name for d in database_user_dir.iterdir() if d.is_dir() and d.name.startswith("letterboxd-")]
    return sorted(users)

def select_user_interactive(database_user_dir: Path) -> str:
    """Interaktywny wybór użytkownika z listy dostępnych folderów."""
    users = get_available_users(database_user_dir)
    if not users:
        print("❌ Nie znaleziono żadnych folderów użytkowników w database_user/")
        sys.exit(1)
    
    print("\n📂 WYBÓR UŻYTKOWNIKA\n" + "="*50)
    for idx, user in enumerate(users, 1):
        print(f"  {idx}. {user}")
    print("="*50 + "\n")
    
    while True:
        try:
            choice = input("Wybierz numer użytkownika (lub 'q' aby wyjść): ").strip()
            if choice.lower() == 'q': sys.exit(0)
            choice_num = int(choice)
            if 1 <= choice_num <= len(users):
                selected = users[choice_num - 1]
                print(f"\n✅ Wybrano: {selected}")
                return selected
            else:
                print(f"❌ Wybierz numer od 1 do {len(users)}")
        except (ValueError, EOFError, KeyboardInterrupt):
            sys.exit("\n👋 Przerwano.")

def step0_cleanup_temp_files(data_dir: Path, checkpoint_dir: Path, runs_dir: Path, skip_cleanup: bool):
    if skip_cleanup:
        print("\n⏭️  Pomijam czyszczenie plików tymczasowych.")
        return True
    
    print("\n" + "="*50 + "\nKROK 0: Czyszczenie plików tymczasowych\n" + "="*50)
    files_to_remove = list((data_dir / "prepared").glob("*"))
    files_to_remove.extend(list(checkpoint_dir.glob("*.pth")))
    files_to_remove.extend([data_dir / "matched_movies.csv", data_dir / "unmatched_movies.csv"])
    
    removed_count = 0
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
            removed_count += 1
    
    if runs_dir.exists():
        import shutil
        shutil.rmtree(runs_dir)
        removed_count += 1
        
    print(f"✅ Wyczyszczono {removed_count} plików/folderów.")
    return True

def step1_match_movies(user_folder: str, tmdb_client: TMDBClient, output_csv: str) -> bool:
    print("\n" + "="*50 + "\nKROK 1: Dopasowanie filmów (On-Demand)\n" + "="*50)
    from match_movies import match_user_movies_to_tmdb, save_matched_movies
    try:
        matched_df, unmatched_df = match_user_movies_to_tmdb(letterboxd_folder=user_folder, tmdb_client=tmdb_client)
        if matched_df.empty:
            print("❌ Nie udało się dopasować żadnych filmów!")
            return False
        save_matched_movies(matched_df, output_csv)
        if not unmatched_df.empty:
            unmatched_path = str(Path(output_csv).parent / "unmatched_movies.csv")
            unmatched_df.to_csv(unmatched_path, index=False)
        print(f"✅ Krok 1 zakończony.")
        return True
    except Exception as e:
        import traceback
        print(f"❌ Błąd w kroku 1: {e}")
        traceback.print_exc()
        return False

def step2_prepare_training_data(matched_csv: str, tmdb_client: TMDBClient, output_dir: str) -> bool:
    print("\n" + "="*50 + "\nKROK 2: Przygotowanie danych (On-Demand)\n" + "="*50)
    from prepare_training_data import DataPreparer
    try:
        preparer = DataPreparer(matched_csv, tmdb_client)
        preparer.save_prepared_data(output_dir)
        print(f"✅ Krok 2 zakończony pomyślnie.")
        return True
    except Exception as e:
        import traceback
        print(f"❌ Błąd w kroku 2: {e}")
        traceback.print_exc()
        return False

def step3_train_model(data_dir: str, checkpoint_dir: str, num_epochs: int) -> bool:
    print("\n" + "="*50 + f"\nKROK 3: Trenowanie modelu ({num_epochs} epok)\n" + "="*50)
    from training import MovieRatingTrainer, create_dataloaders
    from model import create_model
    try:
        data_path = Path(data_dir)
        X_train, y_train = np.load(data_path / "X_train.npy"), np.load(data_path / "y_train.npy")
        X_test, y_test = np.load(data_path / "X_test.npy"), np.load(data_path / "y_test.npy")
        
        print(f"   Zbiór treningowy: {len(X_train)} próbek, walidacyjny: {len(X_test)} próbek")
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_test, y_test)
        
        model = create_model(input_dim=X_train.shape[1])
        trainer = MovieRatingTrainer(model, input_dim=X_train.shape[1])
        
        runs_dir = Path(checkpoint_dir).parent / "runs" / f"training_{int(time.time())}"
        trainer.train(train_loader, val_loader, num_epochs, checkpoint_dir=checkpoint_dir, tensorboard_dir=str(runs_dir))
        
        print(f"✅ Krok 3 zakończony pomyślnie.")
        return True
    except Exception as e:
        import traceback
        print(f"❌ Błąd w kroku 3: {e}")
        traceback.print_exc()
        return False

def step4_generate_recommendations(model_path: str, user_movies_path: str, encoders_path: str, tmdb_client: TMDBClient, matched_csv: str, n_recs: int) -> bool:
    print("\n" + "="*50 + f"\nKROK 4: Generowanie {n_recs} rekomendacji\n" + "="*50)
    from recommender import MovieRecommender, format_recommendations
    try:
        recommender = MovieRecommender(
            model_path=str(model_path),
            user_movies_path=str(user_movies_path),
            encoders_path=str(encoders_path),
            tmdb_client=tmdb_client
        )
        
        user_ratings = pd.read_csv(matched_csv)
        # Używamy tmdb_movie_id, bo to jest prawdziwe ID z TMDB
        watched_ids = user_ratings['tmdb_movie_id'].tolist()
        
        print(f"\nUżytkownik obejrzał {len(user_ratings)} filmów/seriali.")
        
        for media_type in ['movie', 'tv']:
            print(f"\n--- Generuję rekomendacje dla: {media_type.upper()} ---")
            recs = recommender.get_top_recommendations(watched_movie_ids=watched_ids, n=n_recs, movie_type=media_type)
            if not recs.empty:
                format_recommendations(recs, f"{media_type.upper()}S")
            else:
                print(f"   Brak rekomendacji dla {media_type.upper()}.")

        print(f"✅ Krok 4 zakończony pomyślnie.")
        return True
    except Exception as e:
        import traceback
        print(f"❌ Błąd w kroku 4: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Pipeline rekomendacji filmów w 100% w trybie On-Demand.")
    parser.add_argument('--user', help='Nazwa folderu użytkownika w database_user/')
    parser.add_argument('--epochs', type=int, default=100, help='Liczba epok treningu')
    parser.add_argument('--n', type=int, default=20, help='Liczba rekomendacji')
    parser.add_argument('--skip-cleanup', action='store_true', help='Pomiń czyszczenie plików tymczasowych')
    args = parser.parse_args()

    # --- Konfiguracja ---
    load_dotenv(project_root / '.env')
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        print("❌ Brak klucza API! Ustaw TMDB_API_KEY w pliku .env")
        return 1

    tmdb_client = TMDBClient(api_key)
    database_user_dir = project_root / "database_user"
    user_folder_name = args.user or select_user_interactive(database_user_dir)
    user_folder_path = database_user_dir / user_folder_name
    
    if not user_folder_path.exists():
        print(f"❌ Folder użytkownika nie istnieje: {user_folder_path}")
        return 1

    data_dir = project_root / "src" / "data"
    prepared_dir = data_dir / "prepared"
    checkpoint_dir = project_root / "checkpoints"
    runs_dir = project_root / "runs"
    
    matched_csv = data_dir / "matched_movies.csv"
    model_path = checkpoint_dir / "best_model.pth"
    user_movies_path = prepared_dir / "enriched_movies.csv"
    encoders_path = prepared_dir / "encoders.pkl"
    
    print("\n" + "="*50 + "\n🎬 START PIPELINE (100% On-Demand)\n" + "="*50)
    print(f"👤 Użytkownik: {user_folder_name}\n" + "="*50)

    # --- Uruchomienie Kroków ---
    if not step0_cleanup_temp_files(data_dir, checkpoint_dir, runs_dir, args.skip_cleanup): return 1
    if not step1_match_movies(str(user_folder_path), tmdb_client, str(matched_csv)): return 1
    if not step2_prepare_training_data(str(matched_csv), tmdb_client, str(prepared_dir)): return 1
    if not step3_train_model(str(prepared_dir), str(checkpoint_dir), args.epochs): return 1
    if not step4_generate_recommendations(str(model_path), str(user_movies_path), str(encoders_path), tmdb_client, str(matched_csv), args.n): return 1

    print("\n" + "="*50 + "\n🎉 PIPELINE ZAKOŃCZONY POMYŚLNIE!\n" + "="*50)
    
    if not args.skip_cleanup:
        step0_cleanup_temp_files(data_dir, checkpoint_dir, runs_dir, False)
        print("\n✅ Pliki tymczasowe zostały usunięte.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())