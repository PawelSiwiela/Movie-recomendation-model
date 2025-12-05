# 🎬 Spersonalizowany Model Rekomendacji Filmów

Nowoczesny, w pełni zautomatyzowany system rekomendacji filmów, który trenuje spersonalizowany model w oparciu o Twoje oceny z serwisu Letterboxd.

Projekt działa w architekturze **"On-Demand"**, co oznacza, że **nie wymaga pobierania wielogigabajtowej bazy danych**. Wszystkie potrzebne informacje pobierane są na żywo z API TMDB podczas działania skryptu, dzięki czemu możesz uzyskać pierwsze rekomendacje w ciągu kilku minut od zera.

### Główne Cechy:
-   **🚀 Szybki start:** Brak potrzeby wielogodzinnej synchronizacji bazy danych.
-   **🤖 Osobisty model:** Dla każdego użytkownika trenowana jest od nowa sieć neuronowa (PyTorch), która uczy się jego unikalnego gustu.
-   **💡 Inteligentne rekomendacje:** System proponuje filmy, których nie widziałeś, bazując na hybrydowej strategii (biorąc pod uwagę filmy popularne, najwyżej oceniane i podobne do Twoich ulubionych).
-   **✨ Czytelne wyniki:** Rekomendacje prezentowane są jako procentowy "Wynik dopasowania", co jest bardziej intuicyjne niż symulowana ocena w gwiazdkach.
-   **🧹 W pełni zautomatyzowany:** Jeden skrypt (`pipeline.py`) zarządza całym procesem – od wczytania danych, przez trening, aż po wygenerowanie rekomendacji.

---

## 🚀 Uruchomienie

Wymagany jest Python 3.10+ oraz klucz API z [The Movie Database (TMDB)](https://www.themoviedb.org/signup).

### 1. Instalacja

```bash
# Sklonuj repozytorium
git clone https://github.com/PawelSiwiela/Movie-recomendation-model.git
cd Movie-recomendation-model

# Zainstaluj zależności
pip install -r requirements.txt
```

### 2. Konfiguracja

1.  Utwórz w głównym folderze projektu plik o nazwie `.env`.
2.  W pliku `.env` dodaj jedną linię, wklejając swój klucz API v3 z TMDB:
    ```
    TMDB_API_KEY="tutaj_wklej_swój_klucz_api"
    ```

### 3. Dane Użytkownika

1.  Pobierz swój eksport danych z [Letterboxd](https://letterboxd.com/settings/data/).
2.  Wypakuj pobrane archiwum `.zip`.
3.  Przenieś cały folder z danymi (np. `letterboxd-nazwa-2025-12-04...`) do katalogu `database_user/` w projekcie.

### 4. Generowanie Rekomendacji

Wszystko gotowe! Uruchom główny pipeline w terminalu:

```bash
python pipeline.py
```

Skrypt automatycznie wykryje dostępne dane użytkowników i poprosi Cię o wybór w interaktywnym menu.

Możesz również podać użytkownika bezpośrednio:
```bash
python pipeline.py --user nazwa_folderu_uzytkownika
```

Cały proces (dopasowanie filmów, pobranie ich danych, trening i rekomendacja) potrwa kilka-kilkanaście minut, w zależności od liczby ocenionych przez Ciebie filmów i obciążenia API TMDB.

---

## 🛠️ Struktura Projektu

```
.
├── .env                  # Plik z kluczem API (tworzony ręcznie)
├── .gitignore            # Pliki ignorowane przez Git
├── pipeline.py           # GŁÓWNY SKRYPT - wszystko uruchamia się stąd
├── requirements.txt      # Zależności projektu
├── README.md             # Ta dokumentacja
│
├── database/
│   └── tmdb_client.py    # Klient do komunikacji z API TMDB
│
├── database_user/
│   ├── letterboxd_parser.py    # Parser plików CSV z Letterboxd
│   └── letterboxd-user-1/...   # Folder z danymi użytkownika
│
└── src/
    ├── data_matching/
    │   └── match_movies.py     # Dopasowuje filmy z Letterboxd do ID z API TMDB
    │   └── prepare_training_data.py # Przygotowuje dane do treningu
    │
    └── model/
        ├── model.py            # Definicja architektury sieci neuronowej
        ├── training.py         # Logika treningu modelu
        └── recommender.py      # Generowanie rekomendacji z użyciem modelu
```
