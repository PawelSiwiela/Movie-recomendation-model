# 🎬 Personalized Movie Recommendation Model

A modern, fully-automated movie recommendation system that trains a personalized model based on your ratings from Letterboxd.

This project operates on an **"On-Demand" architecture**, which means it **does not require downloading a large, multi-gigabyte database**. All necessary movie information is fetched live from the TMDB API as the script runs, allowing you to get your first recommendations within minutes from a fresh start.

### ✨ Key Features
-   **🚀 Quick Start:** No need for a multi-hour database synchronization. Get up and running in minutes.
-   **🤖 Personal Model:** A new neural network (PyTorch) is trained from scratch for each user, learning their unique taste profile.
-   **💡 Smart Recommendations:** The system suggests movies you haven't seen, based on a hybrid candidate generation strategy (considering popular, top-rated, and movies similar to your favorites).
-   **✨ Interpretable Scores:** Recommendations are presented as a percentage-based "Match Score," which is more intuitive than a simulated star rating.
-   **🧹 Fully Automated:** A single script (`pipeline.py`) manages the entire process—from loading user data and training the model to generating final recommendations.

---

## 🚀 Getting Started

Requires Python 3.10+ and an API key from [The Movie Database (TMDB)](https://www.themoviedb.org/signup).

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/PawelSiwiela/Movie-Recommendation-Model.git
cd Movie-Recommendation-Model

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

1.  In the project's root directory, create a file named `.env`.
2.  Inside the `.env` file, add a single line, pasting your v3 API key from TMDB:
    ```
    TMDB_API_KEY="paste_your_api_key_here"
    ```

### 3. User Data

1.  Download your data export from [Letterboxd](https://letterboxd.com/settings/data/).
2.  Unzip the downloaded archive.
3.  Move the entire data folder (e.g., `letterboxd-username-2025-12-04...`) into the `database_user/` directory in the project.

### 4. Generate Recommendations

You're all set! Run the main pipeline in your terminal:

```bash
python pipeline.py
```

The script will automatically detect the available user data folders and will prompt you to choose one from an interactive menu.

Alternatively, you can specify the user directly:
```bash
python pipeline.py --user your_letterboxd_folder_name
```

The entire process (matching movies, fetching their data, training the model, and generating recommendations) will take several minutes, depending on the number of movies you've rated and the current TMDB API load.

---

## 🛠️ Project Structure

```
.
├── .env                  # Stores your TMDB API key (created manually)
├── .gitignore            # Files ignored by Git
├── pipeline.py           # MAIN SCRIPT - run this to start the process
├── requirements.txt      # Project dependencies
├── README.md             # This documentation file
│
├── database/
│   └── tmdb_client.py    # Client for communicating with the TMDB API
│
├── database_user/
│   ├── letterboxd_parser.py    # Parses CSV files from a Letterboxd export
│   └── letterboxd-user-1/...   # Folder containing a user's data
│
└── src/
    ├── data_matching/
    │   └── match_movies.py     # Matches Letterboxd movies to TMDB IDs via API
    │   └── prepare_training_data.py # Prepares data for model training
    │
    └── model/
        ├── model.py            # Neural network architecture definition
        ├── training.py         # Model training logic
        └── recommender.py      # Generates recommendations using the trained model
```
