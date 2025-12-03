"""Parser dla danych eksportowanych z Letterboxd."""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

class LetterboxdParser:
    """Klasa do parsowania eksportów CSV z Letterboxd."""
    
    def __init__(self, export_folder: str):
        """
        Inicjalizacja parsera.
        
        Args:
            export_folder: Ścieżka do folderu z eksportem Letterboxd
        """
        self.export_folder = Path(export_folder)
        self.ratings_df: Optional[pd.DataFrame] = None
        self.watched_df: Optional[pd.DataFrame] = None
        self.diary_df: Optional[pd.DataFrame] = None
        
    def load_ratings(self) -> pd.DataFrame:
        """
        Ładuje plik ratings.csv z ocenami użytkownika.
        
        Returns:
            DataFrame z ocenami filmów
        """
        ratings_path = self.export_folder / "ratings.csv"
        
        if not ratings_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {ratings_path}")
        
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Normalizacja kolumn
        self.ratings_df.columns = self.ratings_df.columns.str.strip()
        
        # Konwersja typów
        self.ratings_df['Year'] = pd.to_numeric(self.ratings_df['Year'], errors='coerce')
        self.ratings_df['Rating'] = pd.to_numeric(self.ratings_df['Rating'], errors='coerce')
        self.ratings_df['Date'] = pd.to_datetime(self.ratings_df['Date'], errors='coerce')
        
        print(f"✅ Załadowano {len(self.ratings_df)} ocen z Letterboxd")
        return self.ratings_df
    
    def load_watched(self) -> pd.DataFrame:
        """
        Ładuje plik watched.csv z obejrzanymi filmami.
        
        Returns:
            DataFrame z obejrzanymi filmami
        """
        watched_path = self.export_folder / "watched.csv"
        
        if not watched_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {watched_path}")
        
        self.watched_df = pd.read_csv(watched_path)
        self.watched_df.columns = self.watched_df.columns.str.strip()
        self.watched_df['Year'] = pd.to_numeric(self.watched_df['Year'], errors='coerce')
        self.watched_df['Date'] = pd.to_datetime(self.watched_df['Date'], errors='coerce')
        
        print(f"✅ Załadowano {len(self.watched_df)} obejrzanych filmów")
        return self.watched_df
    
    def load_diary(self) -> pd.DataFrame:
        """
        Ładuje plik diary.csv z historią oglądania.
        
        Returns:
            DataFrame z diary
        """
        diary_path = self.export_folder / "diary.csv"
        
        if not diary_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {diary_path}")
        
        self.diary_df = pd.read_csv(diary_path)
        self.diary_df.columns = self.diary_df.columns.str.strip()
        self.diary_df['Year'] = pd.to_numeric(self.diary_df['Year'], errors='coerce')
        self.diary_df['Rating'] = pd.to_numeric(self.diary_df['Rating'], errors='coerce')
        self.diary_df['Date'] = pd.to_datetime(self.diary_df['Date'], errors='coerce')
        self.diary_df['Watched Date'] = pd.to_datetime(self.diary_df['Watched Date'], errors='coerce')
        
        print(f"✅ Załadowano {len(self.diary_df)} wpisów z diary")
        return self.diary_df
    
    def get_all_rated_movies(self) -> pd.DataFrame:
        """
        Zwraca wszystkie filmy z ocenami (z ratings.csv).
        
        Returns:
            DataFrame z ocenionymi filmami
        """
        if self.ratings_df is None:
            self.load_ratings()
        
        return self.ratings_df.copy()
    
    def get_movies_by_rating(self, min_rating: float, max_rating: float = 5.0) -> pd.DataFrame:
        """
        Filtruje filmy według zakresu ocen.
        
        Args:
            min_rating: Minimalna ocena
            max_rating: Maksymalna ocena
            
        Returns:
            DataFrame z filmami w zadanym zakresie ocen
        """
        if self.ratings_df is None:
            self.load_ratings()
        
        filtered = self.ratings_df[
            (self.ratings_df['Rating'] >= min_rating) & 
            (self.ratings_df['Rating'] <= max_rating)
        ]
        
        return filtered.copy()
    
    def get_favorite_movies(self, min_rating: float = 4.0) -> pd.DataFrame:
        """
        Zwraca ulubione filmy (ocena >= min_rating).
        
        Args:
            min_rating: Minimalna ocena dla "ulubionych"
            
        Returns:
            DataFrame z ulubionymi filmami
        """
        return self.get_movies_by_rating(min_rating, 5.0)
    
    def get_statistics(self) -> Dict:
        """
        Zwraca statystyki dotyczące filmów użytkownika.
        
        Returns:
            Słownik ze statystykami
        """
        if self.ratings_df is None:
            self.load_ratings()
        
        stats = {
            'total_rated': len(self.ratings_df),
            'avg_rating': self.ratings_df['Rating'].mean(),
            'median_rating': self.ratings_df['Rating'].median(),
            'min_rating': self.ratings_df['Rating'].min(),
            'max_rating': self.ratings_df['Rating'].max(),
            'rating_distribution': self.ratings_df['Rating'].value_counts().sort_index().to_dict(),
            'movies_by_year': self.ratings_df['Year'].value_counts().sort_index().head(10).to_dict()
        }
        
        return stats
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Eksportuje wszystkie oceny do jednego DataFrame.
        
        Returns:
            Zunifikowany DataFrame z danymi użytkownika
        """
        if self.ratings_df is None:
            self.load_ratings()
        
        # Wybierz najważniejsze kolumny
        df = self.ratings_df[['Name', 'Year', 'Rating', 'Date']].copy()
        df.rename(columns={
            'Name': 'title',
            'Year': 'year',
            'Rating': 'user_rating',
            'Date': 'rating_date'
        }, inplace=True)
        
        return df


if __name__ == "__main__":
    # Test parsera
    parser = LetterboxdParser("letterboxd-paesielawa-2025-12-03-23-47-utc")
    
    # Załaduj dane
    parser.load_ratings()
    
    # Statystyki
    stats = parser.get_statistics()
    print(f"\n📊 Statystyki:")
    print(f"  Ocenionych filmów: {stats['total_rated']}")
    print(f"  Średnia ocena: {stats['avg_rating']:.2f}")
    print(f"  Mediana: {stats['median_rating']}")
    
    # Ulubione filmy
    favorites = parser.get_favorite_movies(min_rating=4.5)
    print(f"\n⭐ Filmy z oceną >= 4.5: {len(favorites)}")
    print(favorites[['Name', 'Year', 'Rating']].head(10))
