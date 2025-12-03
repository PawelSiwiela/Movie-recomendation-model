# 🎬 Movie Recommendation Model

Spersonalizowany system rekomendacji filmów wykorzystujący deep learning (PyTorch) i dane z TMDB oraz Letterboxd.

## 📋 Opis projektu

Model uczący się na podstawie preferencji użytkownika (oceny z Letterboxd/Filmweb) i rekomendujący filmy/seriale z obszernej bazy TMDB.

### Funkcjonalności:

- 🎯 Personalizowane rekomendacje filmów
- 📊 Analiza preferencji użytkownika (gatunki, aktorzy, reżyserzy)
- 🧠 Sieć neuronowa (PyTorch) z embeddingami
- 🗄️ Baza danych SQLite z ~10 000 filmów i 1000 seriali

## 🏗️ Struktura projektu

```
├── src/
│   ├── database/              # Pobieranie i zarządzanie bazą danych
│   │   ├── database_setup.py      # Tworzenie tabel SQLite
│   │   ├── tmdb_client.py         # Klient TMDB API
│   │   └── database_fetcher.py    # Pobieranie danych z TMDB
│   │
│   ├── user_data/            # Parsowanie danych użytkownika
│   │   └── letterboxd_parser.py   # Parser eksportów Letterboxd
│   │
│   └── model/                # Model ML (PyTorch)
│       ├── model.py              # Architektury sieci neuronowych
│       ├── training.py           # Trenowanie modelu
│       ├── recommender.py        # System rekomendacji
│       └── utils.py              # Funkcje pomocnicze
│
├── user_data/                # Dane użytkownika (CSV z Letterboxd/Filmweb)
├── requirements.txt          # Zależności Python
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <https://github.com/PaeSielawa/Movie-recomendation-model>
   ```
2. Navigate to the project directory:
   ```
   cd movie-recommendation-system
   ```
3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Pobierz bazę filmów z TMDB:

```bash
python src/database/database_fetcher.py
```

## 📊 Źródła danych

- **TMDB (The Movie Database)**: ~10 000 filmów i 1 000 seriali

  - Tytuły, rok, gatunki, opisy
  - Obsada (top 10 aktorów)
  - Reżyserzy
  - Oceny i popularność

- **Letterboxd**: Eksport danych użytkownika (CSV)
  - Historia oglądania
  - Oceny filmów
  - Ulubione filmy

## 🧠 Model

### Architektura:

- **Embeddingi**: filmy, gatunki, aktorzy, reżyserzy
- **Feed-Forward Neural Network**: warstwy ukryte z dropout i batch normalization
- **Output**: przewidywana ocena użytkownika (0-5)

### Technologie:

- PyTorch 2.0+
- Pandas, NumPy
- SQLite
- TensorBoard (monitoring treningu)

## 📝 TODO

- [ ] Pobrać pełną bazę danych z TMDB
- [ ] Dopasować filmy użytkownika do bazy TMDB
- [ ] Stworzyć profil użytkownika
- [ ] Wytrenować model
- [ ] Zaimplementować system rekomendacji
- [ ] Dodać wsparcie dla Filmweb CSV
- [ ] Stworzyć interfejs użytkownika (opcjonalnie)

## 📄 Licencja

Projekt edukacyjny - wykorzystuje dane z TMDB (https://www.themoviedb.org/)

## 👤 Autor

PaeSielawa
