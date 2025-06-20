import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import csv_utils
import time

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

def get_total_pages():
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('total_pages')
    else:
        print("Erreur lors de la récupération du nombre de pages.")
        return 0
    
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
        print(f"Fetching page {page}...")
        url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
        response = requests.get(url)
        data = response.json()['results']

        for movie in data:
            movie_id = movie.get('id')
            details_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
            details_response = requests.get(details_url)
            
            if details_response.status_code != 200:
                print(f"Erreur détails film {movie.get('title')} (id={movie_id})")
                continue

            details = details_response.json()

            movies.append({
                'title': movie.get('title'),
                'rating': movie.get('vote_average'),
                'votes': movie.get('vote_count'),
                'genre_ids': movie.get('genre_ids'),
                'language': movie.get('original_language'),
                'release_date': movie.get('release_date'),
                'runtime': details.get('runtime'),
                'budget': details.get('budget'),
                'revenue': details.get('revenue'),
            })

            time.sleep(0.25)  # Attendre pour éviter de se faire bloquer par l'API

    csv_utils.generate_csv("movies", movies)
    return movies

def get_tv_shows(pages=5):
    if csv_utils.csv_exists("tv_shows"):
        return csv_utils.get_data_from_csv("tv_shows")
    else:
        return get_tv_shows_from_api(pages)

def get_tv_shows_from_csv():
    return csv_utils.get_data_from_csv("tv_shows")

def get_tv_shows_from_api(pages):
    tv_shows = []
    for page in range(1, pages + 1):
        print(f"Fetching TV page {page}...")
        url = f"{BASE_URL}/tv/popular?api_key={API_KEY}&language=en-US&page={page}"
        response = requests.get(url)
        data = response.json()['results']

        for tv in data:
            tv_id = tv.get('id')
            details_url = f"{BASE_URL}/tv/{tv_id}?api_key={API_KEY}&language=en-US"
            details_response = requests.get(details_url)

            if details_response.status_code != 200:
                print(f"Erreur détails série {tv.get('name')} (id={tv_id})")
                continue

            details = details_response.json()

            tv_shows.append({
                'title': tv.get('name'),
                'rating': tv.get('vote_average'),
                'votes': tv.get('vote_count'),
                'genre_ids': tv.get('genre_ids'),
                'language': tv.get('original_language'),
                'release_date': tv.get('first_air_date'),
                'runtime': details.get('episode_run_time')[0] if details.get('episode_run_time') else None,
                'budget': None,  # Non disponible dans l’API TV
                'revenue': None,  # Non disponible non plus
                'number_of_seasons': details.get('number_of_seasons'),
                'number_of_episodes': details.get('number_of_episodes'),
            })

            time.sleep(0.25)  # Limite de requête

    csv_utils.generate_csv("tv_shows", tv_shows)
    return tv_shows

def merge_movies_and_tv_shows(movies, tv_shows):
    for movie in movies:
        movie['type'] = 'movie'
    all_content = movies + tv_shows
    df = pd.DataFrame(all_content)
    csv_utils.generate_csv("main", df)
    return df

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

def get_total_pages():
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('total_pages')
    else:
        print("Erreur lors de la récupération du nombre de pages.")
        return 0

# === MAIN ===
#genre_map = get_genres()
movies = get_movies(50)        # Récupère ~1000 films
tv_shows = get_tv_shows(50)    # Récupère ~1000 séries

print(f"Movies loaded: {type(movies)} with length {len(movies) if movies else 'None'}")
print(f"TV Shows loaded: {type(tv_shows)} with length {len(tv_shows) if tv_shows else 'None'}")

all_df = merge_movies_and_tv_shows(movies, tv_shows)
print(all_df.head())

# Aperçu des données
# print(df.head())

# Graphique
# plot_avg_rating_by_genre(df)

print(get_total_pages())