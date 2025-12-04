import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('TMDB_API_KEY')

# Sprawdź popularność niedopasowanych filmów
test_movies = [
    ('Bogowie', 2014),
    ('Katyn', 2007),
    ('Amateurs', 2020),  # Amatorzy
    ('Hurricane', 2018),  # 303 Dywizjon
    ('Winter Brothers', 2017),  # Zimni bracia (duński film)
    ('Afterimage', 2016),  # Powidoki
    ('Silent Night', 2017),  # Cicha noc (polski)
    ('The Sisterhood', 2019),
    ('Williams', 2017),
    ('Megalopolis', 2024),
]

print("🔍 Sprawdzam popularność niedopasowanych filmów:\n")

for title, year in test_movies:
    try:
        r = requests.get(
            f'https://api.themoviedb.org/3/search/movie',
            params={'api_key': api_key, 'query': title, 'year': year}
        )
        results = r.json().get('results', [])
        
        if results:
            top = results[0]
            print(f"✅ '{title}' ({year})")
            print(f"   TMDB: {top['title']} | ID: {top['id']}")
            print(f"   Popularity: {top['popularity']:.2f} | Votes: {top['vote_count']}")
            
            if top['popularity'] < 0.5:
                print(f"   ⚠️  PONIŻEJ progu 0.5!")
            elif top['popularity'] < 1.0:
                print(f"   ⚠️  PONIŻEJ progu 1.0, ale >= 0.5 ✓")
            else:
                print(f"   ✓ Powyżej progu 1.0")
        else:
            print(f"❌ '{title}' ({year}) - nie znaleziono")
        print()
    except Exception as e:
        print(f"❌ Błąd dla '{title}': {e}\n")

print("\n💡 Wnioski:")
print("   min_popularity=0.5 → wyłapie wszystkie z popularity >= 0.5")
print("   min_popularity=1.0 → wyłapie wszystkie z popularity >= 1.0")
