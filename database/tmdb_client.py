"""Klient do komunikacji z TMDB API."""

import requests
import time
from typing import Dict, List, Optional

class TMDBClient:
    """Klient dla The Movie Database API."""
    
    def __init__(self, api_key: str):
        """
        Inicjalizacja klienta TMDB.
        
        Args:
            api_key: Klucz API z TMDB
        """
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        self.request_count = 0
        self.last_request_time = time.time()
    
    def _rate_limit(self):
        """Kontrola limitu zapytań (40 requestów na 10 sekund)."""
        self.request_count += 1
        
        if self.request_count >= 35:  # Bezpieczny margines
            elapsed = time.time() - self.last_request_time
            if elapsed < 10:
                sleep_time = 10 - elapsed + 0.5
                print(f"⏳ Rate limit - czekam {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            
            self.request_count = 0
            self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Wykonuje zapytanie do API.
        
        Args:
            endpoint: Endpoint API
            params: Dodatkowe parametry zapytania
            
        Returns:
            Odpowiedź JSON z API
        """
        self._rate_limit()
        
        if params is None:
            params = {}
        
        params['api_key'] = self.api_key
        params['language'] = 'en-US'  # Możesz zmienić na 'pl-PL'
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Błąd zapytania: {e}")
            return {}
    
    def discover_movies(self, page: int = 1, min_vote_count: int = 100, 
                       min_year: int = 1980, sort_by: str = "vote_average.desc") -> Dict:
        """
        Pobiera listę filmów według kryteriów.
        
        Args:
            page: Numer strony (20 filmów na stronę)
            min_vote_count: Minimalna liczba głosów
            min_year: Minimalny rok produkcji
            sort_by: Sortowanie (popularity.desc, vote_average.desc, etc.)
            
        Returns:
            Dane z API zawierające listę filmów
        """
        params = {
            'page': page,
            'vote_count.gte': min_vote_count,
            'primary_release_date.gte': f'{min_year}-01-01',
            'sort_by': sort_by
        }
        
        return self._make_request('discover/movie', params)
    
    def discover_tv(self, page: int = 1, min_vote_count: int = 100,
                   min_year: int = 1980, sort_by: str = "popularity.desc") -> Dict:
        """
        Pobiera listę seriali według kryteriów.
        
        Args:
            page: Numer strony
            min_vote_count: Minimalna liczba głosów
            min_year: Minimalny rok produkcji
            sort_by: Sortowanie
            
        Returns:
            Dane z API zawierające listę seriali
        """
        params = {
            'page': page,
            'vote_count.gte': min_vote_count,
            'first_air_date.gte': f'{min_year}-01-01',
            'sort_by': sort_by
        }
        
        return self._make_request('discover/tv', params)

    def search_movie(self, query: str, year: Optional[int] = None) -> Dict:
        """
        Wyszukuje filmy po tytule.
        
        Args:
            query: Tytuł filmu
            year: Rok produkcji (opcjonalnie)
            
        Returns:
            Wyniki wyszukiwania z API
        """
        params = {'query': query}
        if year:
            params['year'] = year
        return self._make_request('search/movie', params)

    def search_tv(self, query: str, year: Optional[int] = None) -> Dict:
        """
        Wyszukuje seriale po tytule.
        
        Args:
            query: Tytuł serialu
            year: Rok pierwszej emisji (opcjonalnie)
            
        Returns:
            Wyniki wyszukiwania z API
        """
        params = {'query': query}
        if year:
            params['first_air_date_year'] = year
        return self._make_request('search/tv', params)
    
    def get_movie_details(self, movie_id: int) -> Dict:
        """
        Pobiera szczegóły filmu.
        
        Args:
            movie_id: ID filmu w TMDB
            
        Returns:
            Szczegółowe dane filmu
        """
        return self._make_request(f'movie/{movie_id}')
    
    def get_tv_details(self, tv_id: int) -> Dict:
        """
        Pobiera szczegóły serialu.
        
        Args:
            tv_id: ID serialu w TMDB
            
        Returns:
            Szczegółowe dane serialu
        """
        return self._make_request(f'tv/{tv_id}')
    
    def get_movie_credits(self, movie_id: int) -> Dict:
        """
        Pobiera obsadę i ekipę filmu.
        
        Args:
            movie_id: ID filmu w TMDB
            
        Returns:
            Dane o obsadzie i ekipie
        """
        return self._make_request(f'movie/{movie_id}/credits')
    
    def get_tv_credits(self, tv_id: int) -> Dict:
        """
        Pobiera obsadę i ekipę serialu.
        
        Args:
            tv_id: ID serialu w TMDB
            
        Returns:
            Dane o obsadzie i ekipie
        """
        return self._make_request(f'tv/{tv_id}/credits')

    def get_movie_similar(self, movie_id: int) -> Dict:
        """
        Pobiera listę podobnych filmów.
        
        Args:
            movie_id: ID filmu w TMDB
            
        Returns:
            Dane o podobnych filmach
        """
        return self._make_request(f'movie/{movie_id}/similar')

    def get_tv_similar(self, tv_id: int) -> Dict:
        """
        Pobiera listę podobnych seriali.
        
        Args:
            tv_id: ID serialu w TMDB
            
        Returns:
            Dane o podobnych serialach
        """
        return self._make_request(f'tv/{tv_id}/similar')
    
    def get_genre_list(self, media_type: str = 'movie') -> List[Dict]:
        """
        Pobiera listę dostępnych gatunków.
        
        Args:
            media_type: 'movie' lub 'tv'
            
        Returns:
            Lista gatunków
        """
        result = self._make_request(f'genre/{media_type}/list')
        return result.get('genres', [])
