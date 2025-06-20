import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import csv_utils

load_dotenv()
# === CONFIGURATION ===
API_KEY = os.getenv('TMDB_API_KEY')
BASE_URL = 'https://api.themoviedb.org/3'


# === 1. Récupération des genres pour les ID → noms ===
def get_genres():
    if csv_utils.csv_exists("genres"):
        return get_genres_from_csv()
    else:
       return get_genres_from_api()
    
def get_genres_from_csv():
    return csv_utils.get_data_from_csv("genres")

def get_genres_from_api():
    url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    genres = response.json()['genres']
    csv_utils.generate_csv("genres", genres)
    return {g['id']: g['name'] for g in genres}

# === 2. Récupération de films populaires ===
def get_movies(pages=5):  # Nombre de pages (chaque page ≈ 20 films)
    if csv_utils.csv_exists("movies"):
        return get_movies_from_csv()
    else:
       return get_movies_from_api(pages)

def get_movies_from_csv():
    return csv_utils.get_data_from_csv("movies")

def get_movies_from_api(pages):
    movies = []
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
        response = requests.get(url)
        data = response.json()['results']
        for movie in data:
            movies.append({
                'title': movie['title'],
                'rating': movie['vote_average'],
                'votes': movie['vote_count'],
                'genre_ids': movie['genre_ids'],
                'language': movie['original_language'],
                'release_date': movie['release_date']
            })
    csv_utils.generate_csv("movies", movies)
    return movies

def get_total_pages():
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('total_pages')
    else:
        print("Erreur lors de la récupération du nombre de pages.")
        return 0

# === 3. Traitement des données ===
def enrich_movies(movies, genre_map):
    df = pd.DataFrame(movies)
    df['genres'] = df['genre_ids'].apply(lambda ids: [genre_map.get(i) for i in ids])
    df = df.explode('genres')  # Pour séparer les films multi-genres
    df = df.drop(columns='genre_ids')
    csv_utils.generate_csv("main", df)
    return df

# === 4. Visualisation : note moyenne par genre ===
def plot_avg_rating_by_genre(df):
    genre_rating = df.groupby('genres')['rating'].mean().sort_values(ascending=False)
    genre_rating.plot(kind='bar', color='skyblue', figsize=(10,5))
    plt.title('Note moyenne par genre (films populaires)')
    plt.xlabel('Genre')
    plt.ylabel('Note moyenne')
    plt.tight_layout()
    plt.show()

# === MAIN ===
genre_map = get_genres()
movies = get_movies(50)  # 5 pages → environ 100 films
df = enrich_movies(movies, genre_map)

# Aperçu des données
print(df.head())

# Graphique
plot_avg_rating_by_genre(df)