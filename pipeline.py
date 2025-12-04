"""
Pipeline do pełnego przetwarzania danych użytkownika i generowania rekomendacji.

Automatycznie wykonuje:
1. Dopasowanie filmów użytkownika z Letterboxd do bazy TMDB
2. Przygotowanie danych treningowych (wzbogacenie metadanymi)
3. Opcjonalnie: Trenowanie modelu (jeśli --train)
4. Generowanie rekomendacji filmów i seriali

Usage:
    # Tylko rekomendacje (używa istniejącego modelu):
    python pipeline.py --user letterboxd-plisiu-2025-12-04-11-19-utc
    
    # Z treningiem modelu:
    python pipeline.py --user letterboxd-plisiu-2025-12-04-11-19-utc --train
    
    # Zmiana liczby rekomendacji:
    python pipeline.py --user letterboxd-plisiu-2025-12-04-11-19-utc --n 30
"""

import sys
import argparse
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import pickle
import time

# Dodaj ścieżki do importów
project_root = Path(__file__).parent
sys.path.append(str(project_root / "database_user"))
sys.path.append(str(project_root / "src" / "data_matching"))
sys.path.append(str(project_root / "src" / "model"))


def check_database_exists(db_path: Path) -> bool:
    """Sprawdza czy baza danych TMDB istnieje."""
    if not db_path.exists():
        print(f"❌ Baza danych nie istnieje: {db_path}")
        print(f"💡 Uruchom najpierw: python database/daily_export_fetcher.py")
        return False
    return True


def check_model_exists(model_path: Path) -> bool:
    """Sprawdza czy wytrenowany model istnieje."""
    return model_path.exists()


def step0_cleanup_temp_files(data_dir: Path, skip_cleanup: bool = False) -> bool:
    """
    Krok 0: Czyszczenie plików tymczasowych z poprzedniego użytkownika.
    
    Usuwa:
    - matched_movies.csv
    - unmatched_movies.csv
    - prepared/* (wszystkie pliki treningowe)
    
    Args:
        data_dir: Katalog src/data
        skip_cleanup: Czy pominąć czyszczenie
        
    Returns:
        True jeśli sukces
    """
    if skip_cleanup:
        print("\n⏭️  Pomijam czyszczenie plików tymczasowych (--skip-cleanup)")
        return True
    
    print("\n" + "="*100)
    print("KROK 0: Czyszczenie plików tymczasowych")
    print("="*100)
    
    files_to_remove = [
        data_dir / "matched_movies.csv",
        data_dir / "unmatched_movies.csv",
    ]
    
    prepared_dir = data_dir / "prepared"
    if prepared_dir.exists():
        files_to_remove.extend([
            prepared_dir / "enriched_movies.csv",
            prepared_dir / "X_train.npy",
            prepared_dir / "X_test.npy",
            prepared_dir / "y_train.npy",
            prepared_dir / "y_test.npy",
            prepared_dir / "encoders.pkl",
        ])
    
    # WAŻNE: Usuń też stary model bo enkodery się nie zgadzają!
    checkpoint_dir = data_dir.parent.parent / "checkpoints"
    if checkpoint_dir.exists():
        model_files = [
            checkpoint_dir / "best_model.pth",
            checkpoint_dir / "latest_model.pth",
        ]
        files_to_remove.extend([f for f in model_files if f.exists()])
    
    removed = 0
    for file_path in files_to_remove:
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"   ✅ Usunięto: {file_path.name}")
                removed += 1
            except Exception as e:
                print(f"   ⚠️  Nie można usunąć {file_path.name}: {e}")
    
    if removed == 0:
        print("   ℹ️  Brak plików do usunięcia (czysty start)")
    else:
        print(f"\n✅ Wyczyszczono {removed} plików")
    
    return True


def step1_match_movies(user_folder: str, db_path: str, output_csv: str) -> bool:
    """
    Krok 1: Dopasowanie filmów użytkownika do bazy TMDB.
    
    Args:
        user_folder: Folder z eksportem Letterboxd
        db_path: Ścieżka do bazy TMDB
        output_csv: Plik wyjściowy matched_movies.csv
        
    Returns:
        True jeśli sukces
    """
    print("\n" + "="*100)
    print("KROK 1: Dopasowanie filmów użytkownika do bazy TMDB")
    print("="*100)
    
    try:
        from letterboxd_parser import LetterboxdParser
        from match_movies import match_user_movies_to_tmdb, save_matched_movies
        
        # Dopasuj filmy
        matched_df, unmatched_df = match_user_movies_to_tmdb(
            letterboxd_folder=user_folder,
            tmdb_db_path=db_path,
            min_similarity=0.85
        )
        
        if len(matched_df) == 0:
            print("❌ Nie udało się dopasować żadnych filmów!")
            return False
        
        # Zapisz wyniki
        save_matched_movies(matched_df, output_csv)
        
        # Zapisz niedopasowane
        if len(unmatched_df) > 0:
            unmatched_path = str(Path(output_csv).parent / "unmatched_movies.csv")
            unmatched_df.to_csv(unmatched_path, index=False)
            print(f"\n💾 Niedopasowane filmy zapisane do: {unmatched_path}")
        
        print(f"\n✅ Krok 1 zakończony pomyślnie!")
        print(f"   Dopasowano: {len(matched_df)} filmów")
        print(f"   Niedopasowane: {len(unmatched_df)} filmów")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Błąd w kroku 1: {e}")
        import traceback
        traceback.print_exc()
        return False


def step2_prepare_training_data(matched_csv: str, db_path: str, output_dir: str) -> bool:
    """
    Krok 2: Przygotowanie danych treningowych.
    
    Args:
        matched_csv: Plik matched_movies.csv
        db_path: Ścieżka do bazy TMDB
        output_dir: Folder wyjściowy dla danych treningowych
        
    Returns:
        True jeśli sukces
    """
    print("\n" + "="*100)
    print("KROK 2: Przygotowanie danych treningowych")
    print("="*100)
    
    try:
        from prepare_training_data import DataPreparer
        
        # Przygotuj dane
        preparer = DataPreparer(matched_csv, db_path)
        
        try:
            preparer.save_prepared_data(output_dir)
            print(f"\n✅ Krok 2 zakończony pomyślnie!")
            return True
        finally:
            preparer.close()
            
    except Exception as e:
        print(f"\n❌ Błąd w kroku 2: {e}")
        import traceback
        traceback.print_exc()
        return False


def step3_train_model(data_dir: str, checkpoint_dir: str, num_epochs: int = 100, model_type: str = 'standard') -> bool:
    """
    Krok 3: Trenowanie modelu (opcjonalny).
    
    Args:
        data_dir: Folder z danymi treningowymi
        checkpoint_dir: Folder na checkpointy
        num_epochs: Liczba epok
        
    Returns:
        True jeśli sukces
    """
    print("\n" + "="*100)
    print("KROK 3: Trenowanie modelu")
    print("="*100)
    
    try:
        import torch
        from training import MovieRatingTrainer, create_dataloaders
        from model import create_model
        
        data_dir = Path(data_dir)
        
        # Załaduj dane
        print("📂 Ładuję dane...")
        X_train = np.load(data_dir / "X_train.npy")
        X_test = np.load(data_dir / "X_test.npy")
        y_train = np.load(data_dir / "y_train.npy")
        y_test = np.load(data_dir / "y_test.npy")
        
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Utwórz DataLoadery
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=32
        )
        
        # Utwórz model
        input_dim = X_train.shape[1]
        print(f"   Architektura: {model_type}")
        model = create_model(input_dim, model_type=model_type)
        
        # Utwórz trainera
        trainer = MovieRatingTrainer(model, learning_rate=0.001)
        
        # Trening
        print(f"\n🚀 Trening ({num_epochs} epok)...\n")
        
        tensorboard_dir = Path(checkpoint_dir).parent / "runs" / f"training_{int(time.time())}"
        
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            early_stopping_patience=15,
            checkpoint_dir=checkpoint_dir,
            tensorboard_dir=str(tensorboard_dir)
        )
        
        print(f"\n✅ Krok 3 zakończony pomyślnie!")
        print(f"📊 TensorBoard: tensorboard --logdir={tensorboard_dir.parent}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Błąd w kroku 3: {e}")
        import traceback
        traceback.print_exc()
        return False


def step4_generate_recommendations(
    model_path: str,
    enriched_data_path: str,
    encoders_path: str,
    db_path: str,
    matched_csv: str,
    n_recommendations: int = 20
) -> bool:
    """
    Krok 4: Generowanie rekomendacji.
    
    Args:
        model_path: Ścieżka do wytrenowanego modelu
        enriched_data_path: Ścieżka do enriched_movies.csv
        encoders_path: Ścieżka do encoders.pkl
        db_path: Ścieżka do bazy TMDB
        matched_csv: Plik matched_movies.csv (do pobrania obejrzanych filmów)
        n_recommendations: Liczba rekomendacji
        
    Returns:
        True jeśli sukces
    """
    print("\n" + "="*100)
    print("KROK 4: Generowanie rekomendacji")
    print("="*100)
    
    try:
        from recommender import MovieRecommender, format_recommendations
        
        # Inicjalizacja recommender
        recommender = MovieRecommender(
            model_path=model_path,
            enriched_data_path=enriched_data_path,
            encoders_path=encoders_path,
            db_path=db_path
        )
        
        # Załaduj obejrzane filmy
        user_ratings = pd.read_csv(matched_csv)
        watched_ids = user_ratings['tmdb_id'].tolist()
        
        print(f"\nUżytkownik obejrzał {len(user_ratings)} filmów/seriali")
        
        # Top 3 najlepiej ocenione
        top_3 = user_ratings.nlargest(3, 'user_rating')[['tmdb_title', 'user_rating', 'tmdb_year']]
        print(f"   Najlepiej ocenione:")
        for idx, row in top_3.iterrows():
            print(f"   - {row['tmdb_title']} ({row['tmdb_year']}): {row['user_rating']}/5.0")
        
        # Rekomendacje filmów
        print("\n" + "="*100)
        print(f"Generuję {n_recommendations} rekomendacji FILMÓW...")
        print("="*100)
        
        try:
            movie_recs = recommender.get_top_recommendations(
                watched_movie_ids=watched_ids,
                n=n_recommendations,
                min_rating=None,  # Brak filtrowania po minimalnej ocenie - pokaż najlepsze dostępne
                min_popularity=10.0,
                movie_type='movie'
            )
            
            if len(movie_recs) > 0:
                format_recommendations(movie_recs, "FILMOW")
            else:
                print("   Brak rekomendacji filmów")
                
        except Exception as e:
            print(f"   ❌ Błąd przy rekomendacjach filmów: {e}")
        
        # Rekomendacje seriali
        print("\n" + "="*100)
        print(f"Generuję {n_recommendations} rekomendacji SERIALI...")
        print("="*100)
        
        try:
            tv_recs = recommender.get_top_recommendations(
                watched_movie_ids=watched_ids,
                n=n_recommendations,
                min_rating=None,  # Brak filtrowania po minimalnej ocenie - pokaż najlepsze dostępne
                min_popularity=10.0,
                movie_type='tv'
            )
            
            if len(tv_recs) > 0:
                format_recommendations(tv_recs, "SERIALI")
            else:
                print("   Brak rekomendacji seriali")
                
        except Exception as e:
            print(f"   ❌ Błąd przy rekomendacjach seriali: {e}")
        
        print(f"\n✅ Krok 4 zakończony pomyślnie!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Błąd w kroku 4: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline do generowania rekomendacji filmów dla użytkownika Letterboxd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Tylko rekomendacje (używa istniejącego modelu):
  python pipeline.py --user letterboxd-plisiu-2025-12-04-11-19-utc
  
  # Z treningiem modelu:
  python pipeline.py --user letterboxd-plisiu-2025-12-04-11-19-utc --train
  
  # Zmiana liczby rekomendacji:
  python pipeline.py --user letterboxd-plisiu-2025-12-04-11-19-utc --n 30
  
  # Pełny pipeline z treningiem (100 epok):
  python pipeline.py --user letterboxd-plisiu-2025-12-04-11-19-utc --train --epochs 100
        """
    )
    
    parser.add_argument(
        '--user',
        required=True,
        help='Nazwa folderu użytkownika w database_user/ (np. letterboxd-plisiu-2025-12-04-11-19-utc)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Czy trenować model od nowa (domyślnie używa istniejącego modelu)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Liczba epok treningu (domyślnie 100)'
    )
    
    parser.add_argument(
        '--n',
        type=int,
        default=20,
        help='Liczba rekomendacji do wygenerowania (domyślnie 20)'
    )
    
    parser.add_argument(
        '--skip-match',
        action='store_true',
        help='Pomiń krok dopasowania filmów (użyj istniejącego matched_movies.csv)'
    )
    
    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Pomiń krok przygotowania danych (użyj istniejących danych treningowych)'
    )
    
    parser.add_argument(
        '--skip-cleanup',
        action='store_true',
        help='Pomiń czyszczenie plików tymczasowych (może spowodować konflikty między użytkownikami!)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='advanced',
        choices=['standard', 'deep', 'advanced'],
        help='Architektura modelu: standard (szybka, ~86k), deep (~340k), advanced (najlepsza, ~530k)'
    )
    
    args = parser.parse_args()
    
    # Ścieżki
    base_dir = Path(__file__).parent
    user_folder = base_dir / "database_user" / args.user
    db_path = base_dir / "database" / "movies.db"
    matched_csv = base_dir / "src" / "data" / "matched_movies.csv"
    prepared_dir = base_dir / "src" / "data" / "prepared"
    checkpoint_dir = base_dir / "checkpoints"
    model_path = checkpoint_dir / "best_model.pth"
    enriched_data_path = prepared_dir / "enriched_movies.csv"
    encoders_path = prepared_dir / "encoders.pkl"
    
    # Walidacja
    if not user_folder.exists():
        print(f"❌ Folder użytkownika nie istnieje: {user_folder}")
        print(f"💡 Dostępne foldery w database_user/:")
        for folder in (base_dir / "database_user").iterdir():
            if folder.is_dir() and folder.name.startswith("letterboxd-"):
                print(f"   - {folder.name}")
        return 1
    
    if not check_database_exists(db_path):
        return 1
    
    print("\n" + "="*100)
    print("🎬 PIPELINE REKOMENDACJI FILMÓW")
    print("="*100)
    print(f"Użytkownik: {args.user}")
    print(f"Architektura: {args.model}")
    print(f"Trening modelu: {'TAK' if args.train else 'NIE'}")
    print(f"Liczba rekomendacji: {args.n}")
    print("="*100)
    
    # KROK 0: Czyszczenie plików tymczasowych
    # Wykonaj tylko jeśli nie pomijamy wszystkich kroków przetwarzania
    data_dir = base_dir / "src" / "data"
    should_cleanup = not args.skip_cleanup and not (args.skip_match and args.skip_prepare)
    if not step0_cleanup_temp_files(data_dir, not should_cleanup):
        print("\n❌ Pipeline przerwany na kroku 0")
        return 1
    
    # KROK 1: Dopasowanie filmów
    
    # KROK 1: Dopasowanie filmów
    if not args.skip_match:
        if not step1_match_movies(str(user_folder), str(db_path), str(matched_csv)):
            print("\n❌ Pipeline przerwany na kroku 1")
            return 1
    else:
        print("\n⏭️  Pomijam krok 1 (dopasowanie filmów)")
        if not matched_csv.exists():
            print(f"❌ Plik {matched_csv} nie istnieje!")
            return 1
    
    # KROK 2: Przygotowanie danych treningowych
    if not args.skip_prepare:
        if not step2_prepare_training_data(str(matched_csv), str(db_path), str(prepared_dir)):
            print("\n❌ Pipeline przerwany na kroku 2")
            return 1
    else:
        print("\n⏭️  Pomijam krok 2 (przygotowanie danych)")
        if not enriched_data_path.exists():
            print(f"❌ Plik {enriched_data_path} nie istnieje!")
            return 1
    
    # KROK 3: Trenowanie modelu
    # WAŻNE: Zawsze trenujemy model dla nowych danych użytkownika!
    # Model musi być zgodny z enkoderami użytkownika (różne liczby gatunków/aktorów/reżyserów)
    if args.train or not args.skip_match or not args.skip_prepare:
        # Jeśli przetwarzaliśmy dane, trenuj model
        print(f"\n💡 Model będzie wytrenowany dla danych użytkownika (architektura: {args.model})...")
        if not step3_train_model(str(prepared_dir), str(checkpoint_dir), args.epochs, args.model):
            print("\n❌ Pipeline przerwany na kroku 3")
            return 1
    else:
        # Tylko jeśli jawnie pomijamy wszystko
        print("\n⏭️  Pomijam krok 3 (trening modelu)")
        if not check_model_exists(model_path):
            print(f"❌ Model nie istnieje: {model_path}")
            print(f"💡 Usuń flagi --skip-* aby wytrenować model")
            return 1
    
    # KROK 4: Generowanie rekomendacji
    if not step4_generate_recommendations(
        str(model_path),
        str(enriched_data_path),
        str(encoders_path),
        str(db_path),
        str(matched_csv),
        args.n
    ):
        print("\n❌ Pipeline przerwany na kroku 4")
        return 1
    
    print("\n" + "="*100)
    print("🎉 PIPELINE ZAKOŃCZONY POMYŚLNIE!")
    print("="*100)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
