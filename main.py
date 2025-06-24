import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import csv_utils
import time
import ast

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
    movie_url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=en-US"
    tv_url = f"{BASE_URL}/genre/tv/list?api_key={API_KEY}&language=en-US"

    movie_genres = requests.get(movie_url).json().get('genres', [])
    tv_genres = requests.get(tv_url).json().get('genres', [])

    # Fusion sans doublons
    all_genres = {g['id']: g['name'] for g in movie_genres}
    all_genres.update({g['id']: g['name'] for g in tv_genres})

    genre_list = [{'id': k, 'name': v} for k, v in all_genres.items()]
    csv_utils.generate_csv("genres", genre_list)
    return genre_list

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

    # Assure que genre_ids est bien une liste
    df['genre_ids'] = df['genre_ids'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Map les ids vers les noms de genres
    df['genres'] = df['genre_ids'].apply(lambda ids: [genre_map.get(int(i)) for i in ids if genre_map.get(int(i))])

    df = df.explode('genres')  # Séparer les lignes multi-genres
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
    
def plot_runtime_trend(df):
    df = df.copy()
    # Nettoyage : on enlève les contenus sans runtime ou année
    df = df[df['runtime'].notnull()]
    df = df[df['release_date'].notnull()]
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df[df['year'].notnull()]
    df['year'] = df['year'].astype(int)

    # On filtre les années raisonnables
    df = df[(df['year'] >= 1980) & (df['year'] <= 2024)]

    # Moyenne par type et année
    grouped = df.groupby(['type', 'year'])['runtime'].mean().reset_index()

    # Tracé
    plt.figure(figsize=(12,6))
    for content_type in grouped['type'].unique():
        data = grouped[grouped['type'] == content_type]
        plt.plot(data['year'], data['runtime'], label=content_type.capitalize())

    plt.title("Évolution de la durée moyenne des films et séries (1980–2024)")
    plt.xlabel("Année de sortie")
    plt.ylabel("Durée moyenne (minutes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_movies_from_api(pages_per_year=10): ## Récupération des films par année (1980–2023)
    movies = []
    for year in range(1980, 2024):  # Modifie si tu veux d'autres années
        for page in range(1, pages_per_year + 1):
            print(f"[MOVIE] Year {year} - Page {page}")
            url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&language=en-US&sort_by=popularity.desc&primary_release_year={year}&page={page}"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Erreur page {page} pour l'année {year}")
                continue

            for movie in response.json().get('results', []):
                movie_id = movie.get('id')
                details_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
                details_response = requests.get(details_url)
                if details_response.status_code != 200:
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
    csv_utils.generate_csv("movies", movies)
    return movies

def get_tv_shows_from_api(pages_per_year=10): # Récupération des séries par année (1980–2023)
    tv_shows = []
    for year in range(1980, 2024):
        for page in range(1, pages_per_year + 1):
            print(f"[TV] Year {year} - Page {page}")
            url = f"{BASE_URL}/discover/tv?api_key={API_KEY}&language=en-US&sort_by=popularity.desc&first_air_date_year={year}&page={page}"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Erreur page {page} pour l'année {year}")
                continue

            for tv in response.json().get('results', []):
                tv_id = tv.get('id')
                details_url = f"{BASE_URL}/tv/{tv_id}?api_key={API_KEY}&language=en-US"
                details_response = requests.get(details_url)
                if details_response.status_code != 200:
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
                    'budget': None,
                    'revenue': None,
                    'number_of_seasons': details.get('number_of_seasons'),
                    'number_of_episodes': details.get('number_of_episodes'),
                })
    csv_utils.generate_csv("tv_shows", tv_shows)
    return tv_shows

# === MAIN ===
#genre_map = get_genres()
# movies = get_movies_from_api(5)
# tv_shows = get_tv_shows_from_api(5)

# all_df = merge_movies_and_tv_shows(movies, tv_shows)

def plot_budget_vs_rating(df):
    """
    Affiche un graphique de dispersion montrant la relation entre le budget et la note moyenne des films.

    Cette fonction filtre les données pour ne conserver que les films avec un budget positif 
    et une note valide, puis génère un nuage de points avec une échelle logarithmique sur l'axe X
    pour mieux visualiser la distribution des budgets qui peuvent varier énormément.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données de films avec au minimum les colonnes :
        - 'type' : type de contenu (doit contenir 'movie')
        - 'budget' : budget du film en USD
        - 'rating' : note moyenne du film

    Returns:
    -------
    None
        Affiche directement le graphique à l'écran

    Notes:
    -----
    - Seuls les films (type == 'movie') sont pris en compte
    - Les films avec un budget de 0 ou négatif sont exclus
    - Les films sans note (rating null) sont exclus  
    - L'échelle logarithmique permet de mieux visualiser les budgets très variables
    - Le graphique utilise une transparence (alpha=0.5) pour gérer les superpositions
    """
    df = df.copy()
    df = df[df['type'] == 'movie']
    df = df[(df['budget'] > 0) & (df['rating'].notnull())]

    plt.figure(figsize=(10, 6))
    plt.scatter(df['budget'], df['rating'], alpha=0.5, color='darkcyan')
    plt.title("Budget vs Note (Films)")
    plt.xlabel("Budget (USD)")
    plt.ylabel("Note moyenne")
    plt.xscale('log')  # Logarithmique car certains budgets explosent
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_genres_over_time(df):
    """
    Génère une heatmap visualisant l'évolution de la popularité des genres de films/séries au fil du temps.
    
    Cette fonction crée une carte de chaleur (heatmap) qui montre comment la fréquence d'apparition 
    des différents genres évolue entre 1980 et 2024. Les genres sont triés par ordre décroissant 
    de popularité totale sur la période analysée.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données de films/séries avec au minimum les colonnes :
        - 'release_date' : dates de sortie des films/séries
        - 'genres' : liste des genres pour chaque film/série
    
    Returns:
    --------
    None
        Affiche directement le graphique avec matplotlib
    
    Notes:
    ------
    - Filtre automatiquement les données entre 1980 et 2024
    - Exclut les entrées sans date de sortie
    - Les genres doivent être sous forme de liste pour permettre l'explosion des données
    - La heatmap utilise une colormap 'viridis' où les couleurs plus claires indiquent 
      une plus grande fréquence
    - Les genres sont ordonnés verticalement par popularité décroissante
    - Les années sont affichées horizontalement avec rotation à 90°
    """
    df = df.copy()
    df = df[df['release_date'].notnull()]
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df[df['year'].between(1980, 2024)]
    df = df.explode('genres')  # Assure-toi que cette colonne est bien listée

    genre_counts = df.groupby(['year', 'genres']).size().unstack(fill_value=0)

    # Trier les genres par leur total sur toutes les années
    genre_totals = genre_counts.sum(axis=0).sort_values(ascending=False)
    genre_counts = genre_counts[genre_totals.index]

    plt.figure(figsize=(14, 8))
    plt.imshow(genre_counts.T, aspect='auto', cmap='viridis')
    plt.title("Évolution de la popularité des genres dans le temps")
    plt.xlabel("Année")
    plt.ylabel("Genre")
    plt.xticks(ticks=range(len(genre_counts.index)), labels=genre_counts.index, rotation=90)
    plt.yticks(ticks=range(len(genre_counts.columns)), labels=genre_counts.columns)
    plt.colorbar(label='Nombre de films/séries')
    plt.tight_layout()
    plt.show()

def plot_revenue_vs_budget(df):
    """
    Affiche un graphique de dispersion montrant la relation entre le budget et les revenus des films.
    
    Cette fonction crée un scatter plot avec une échelle logarithmique pour visualiser
    la corrélation entre le budget investi et les revenus générés par les films.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données de films avec les colonnes requises :
        - 'type' : type de contenu (doit contenir 'movie')
        - 'budget' : budget du film en USD
        - 'revenue' : revenus du film en USD
    
    Returns:
    --------
    None
        Affiche directement le graphique à l'écran
    
    Notes:
    ------
    - Filtre automatiquement pour ne garder que les films (type == 'movie')
    - Exclut les entrées avec budget ou revenus <= 0
    - Utilise une échelle logarithmique sur les deux axes
    - Le graphique est affiché avec une grille et un style optimisé
    """
    df = df.copy()
    df = df[df['type'] == 'movie']
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

    plt.figure(figsize=(10, 6))
    plt.scatter(df['budget'], df['revenue'], alpha=0.5, color='darkgreen')
    plt.title("Budget vs Revenus (Films)")
    plt.xlabel("Budget (USD)")
    plt.ylabel("Revenus (USD)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_roi_distribution(df):
    """
    Affiche un histogramme de la distribution de la rentabilité (ROI) des films.
    
    Cette fonction calcule le retour sur investissement (ROI) comme le ratio revenus/budget
    pour les films ayant un budget et des revenus positifs, puis affiche la distribution
    sous forme d'histogramme.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données cinématographiques avec les colonnes :
        - 'type' : type de contenu (seuls les 'movie' sont considérés)
        - 'budget' : budget du film (doit être > 0)
        - 'revenue' : revenus du film (doit être > 0)
    
    Returns:
    --------
    None
        Affiche directement l'histogramme avec matplotlib
    
    Notes:
    ------
    - Seuls les films (type='movie') sont inclus dans l'analyse
    - Les films avec budget ou revenus nuls/négatifs sont exclus
    - L'histogramme est limité à un ROI entre 0 et 10 pour une meilleure lisibilité
    - Utilise 100 bins pour une granularité fine de la distribution
    """
    df = df.copy()
    df = df[df['type'] == 'movie']
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

    df['roi'] = df['revenue'] / df['budget']

    plt.figure(figsize=(10, 5))
    plt.hist(df['roi'], bins=100, color='coral', range=(0, 10))
    plt.title("Distribution de la rentabilité (Revenus / Budget)")
    plt.xlabel("ROI")
    plt.ylabel("Nombre de films")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_languages_distribution(df):
    """
    Affiche un graphique en barres des 10 langues les plus représentées dans le dataset.
    
    Cette fonction analyse la distribution des langues dans le DataFrame fourni et génère
    un graphique en barres horizontal montrant les 10 langues ayant le plus grand nombre
    de contenus associés.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant une colonne 'language' avec les langues des contenus
        
    Returns:
    --------
    None
        Affiche directement le graphique via matplotlib
        
    Notes:
    ------
    - Le graphique utilise une couleur violet moyen ('mediumpurple')
    - Les titres et labels sont en français
    - La fonction créé une copie du DataFrame pour éviter les modifications
    - Utilise plt.tight_layout() pour optimiser l'affichage
    """
    df = df.copy()
    top_langs = df['language'].value_counts().nlargest(10)

    plt.figure(figsize=(8, 5))
    top_langs.plot(kind='bar', color='mediumpurple')
    plt.title("Langues les plus représentées (Top 10)")
    plt.xlabel("Langue")
    plt.ylabel("Nombre de contenus")
    plt.tight_layout()
    plt.show()

def plot_release_volume(df):
    """
    Génère un graphique en barres empilées montrant le nombre de films et séries sortis par année.

    Cette fonction analyse les données de contenu audiovisuel pour créer une visualisation
    du volume de sorties par type de contenu (films/séries) entre 1980 et 2024.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données de contenu avec au minimum les colonnes :
        - 'release_date' : date de sortie du contenu
        - 'type' : type de contenu (film, série, etc.)

    Returns:
    -------
    None
        Affiche directement le graphique via matplotlib.

    Notes:
    -----
    - Filtre automatiquement les données pour ne conserver que les années entre 1980 et 2024
    - Ignore les entrées avec des dates de sortie nulles
    - Utilise un graphique en barres empilées avec la palette de couleurs 'tab20c'
    - Le graphique affiché a une taille de 14x6 pouces avec un titre en français
    """
    df = df.copy()
    df = df[df['release_date'].notnull()]
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df[df['year'].between(1980, 2024)]

    releases = df.groupby(['year', 'type']).size().unstack(fill_value=0)

    releases.plot(kind='bar', stacked=True, figsize=(14,6), colormap='tab20c')
    plt.title("Nombre de films et séries sortis par an")
    plt.xlabel("Année")
    plt.ylabel("Nombre de contenus")
    plt.tight_layout()
    plt.show()

def plot_genre_distribution_by_type(df):
    """
    Affiche un graphique en barres de la répartition des genres par type de contenu (Films vs Séries).
    
    Cette fonction crée un graphique en barres groupées qui compare la distribution des différents
    genres entre les films et les séries. Chaque genre est représenté sur l'axe X, et la hauteur
    des barres indique le nombre de contenus pour chaque type (film/série).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données avec au minimum les colonnes :
        - 'type' : Type de contenu (ex: 'Movie', 'TV Show')
        - 'genres' : Genre du contenu (ex: 'Action', 'Comedy', etc.)
    
    Returns:
    --------
    None
        Affiche directement le graphique avec matplotlib.
    
    Notes:
    ------
    - Le graphique utilise la palette de couleurs 'Set2' pour différencier les types
    - Les dimensions du graphique sont fixées à 14x6 pouces
    - Le titre, les labels des axes et la mise en page sont automatiquement configurés
    - La fonction fait une copie du DataFrame pour éviter de modifier l'original
    """
    df = df.copy()
    genre_counts = df.groupby(['type', 'genres']).size().unstack(fill_value=0)

    genre_counts.T.plot(kind='bar', figsize=(14,6), colormap='Set2')
    plt.title("Répartition des genres (Films vs Séries)")
    plt.xlabel("Genre")
    plt.ylabel("Nombre de contenus")
    plt.tight_layout()
    plt.show()

# Aperçu des données
# print(df.head())

# Graphique
# plot_avg_rating_by_genre(df)

genre_list = get_genres()
genre_map = genre_list

movies = pd.read_csv('CSV/movies.csv')
tv_shows = pd.read_csv('CSV/tv_shows.csv')

movies = enrich_movies(movies, genre_map)
tv_shows = enrich_movies(tv_shows, genre_map)

movies['type'] = 'movie'
tv_shows['type'] = 'tv'

all_df = pd.concat([movies, tv_shows], ignore_index=True)
csv_utils.generate_csv("main", all_df)

plot_budget_vs_rating(all_df)
plot_genres_over_time(all_df)
plot_revenue_vs_budget(all_df)
plot_roi_distribution(all_df)
plot_languages_distribution(all_df)
plot_release_volume(all_df)
plot_genre_distribution_by_type(all_df)