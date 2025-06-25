import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import csv_utils
import time
import ast
import numpy as np

load_dotenv()
# === CONFIGURATION ===
API_KEY = os.getenv('TMDB_API_KEY')
BASE_URL = 'https://api.themoviedb.org/3'

# === 1. Récupération des genres pour les ID → noms ===
def get_genres_from_csv():
    """Récupère les genres depuis le CSV et les convertit en dictionnaire"""
    genres_data = csv_utils.get_data_from_csv("genres")
    
    # Vérifier le format des données
    if isinstance(genres_data, pd.DataFrame):
        # Si c'est un DataFrame pandas
        return {row['id']: row['name'] for _, row in genres_data.iterrows()}
    elif isinstance(genres_data, list) and len(genres_data) > 0:
        # Si c'est une liste de dictionnaires
        if isinstance(genres_data[0], dict):
            return {g['id']: g['name'] for g in genres_data}
        else:
            print("Format de données genres inattendu, récupération depuis l'API...")
            return get_genres_from_api()
    else:
        print("Données genres vides ou format invalide, récupération depuis l'API...")
        return get_genres_from_api()

def get_genres_from_api():
    """Récupère les genres depuis l'API TMDB"""
    try:
        movie_url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=en-US"
        tv_url = f"{BASE_URL}/genre/tv/list?api_key={API_KEY}&language=en-US"

        movie_response = requests.get(movie_url)
        tv_response = requests.get(tv_url)
        
        if movie_response.status_code != 200 or tv_response.status_code != 200:
            print("Erreur lors de la récupération des genres depuis l'API")
            return {}

        movie_genres = movie_response.json().get('genres', [])
        tv_genres = tv_response.json().get('genres', [])

        # Fusion sans doublons
        all_genres = {g['id']: g['name'] for g in movie_genres}
        all_genres.update({g['id']: g['name'] for g in tv_genres})

        # Sauvegarder en CSV
        genre_list = [{'id': k, 'name': v} for k, v in all_genres.items()]
        csv_utils.generate_csv("genres", genre_list)
        
        print(f"Récupéré {len(all_genres)} genres depuis l'API")
        return all_genres
        
    except Exception as e:
        print(f"Erreur lors de la récupération des genres: {e}")
        return {}

def get_genres():
    """Point d'entrée principal pour récupérer les genres"""
    try:
        if csv_utils.csv_exists("genres"):
            print("Chargement des genres depuis le CSV...")
            genres = get_genres_from_csv()
            if genres:  # Si la récupération CSV a fonctionné
                print(f"Chargé {len(genres)} genres depuis le CSV")
                return genres
        
        print("Récupération des genres depuis l'API...")
        return get_genres_from_api()
        
    except Exception as e:
        print(f"Erreur dans get_genres: {e}")
        return {}

# === 2. NOUVELLE RÉCUPÉRATION EXHAUSTIVE ===
def get_movies_comprehensive(start_year=1980, end_year=2024):
    """Récupération exhaustive de films par année avec multiple catégories"""
    if csv_utils.csv_exists("movies_comprehensive"):
        print("Chargement des films depuis le CSV...")
        return csv_utils.get_data_from_csv("movies_comprehensive")
    
    movies = []
    
    # Catégories et critères de tri
    sort_criteria = [
        'popularity.desc',
        'vote_average.desc', 
        'revenue.desc',
        'vote_count.desc'
    ]
    
    # Genres principaux
    movie_genres = [28, 12, 16, 35, 80, 99, 18, 10751, 14, 36, 27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37]
    
    for year in range(start_year, end_year + 1):
        print(f"=== Films année {year} ===")
        year_movies = []
        
        # 1. Par critères de tri
        for sort_by in sort_criteria:
            print(f"  Tri: {sort_by}")
            for page in range(1, 3):  # 2 pages par critère
                url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&language=en-US&sort_by={sort_by}&primary_release_year={year}&vote_count.gte=10&page={page}"
                
                try:
                    response = requests.get(url)
                    if response.status_code != 200:
                        continue
                    
                    for movie in response.json().get('results', []):
                        movie_details = get_movie_details_complete(movie, f"{sort_by}_{year}")
                        if movie_details:
                            year_movies.append(movie_details)
                    
                except Exception as e:
                    print(f"    Erreur: {e}")
                    continue
        
        # 2. Par genres
        for genre_id in movie_genres:  # TOUS les genres au lieu de [:8]
            print(f"  Genre: {genre_id}")
            url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&language=en-US&sort_by=popularity.desc&primary_release_year={year}&with_genres={genre_id}&vote_count.gte=5&page=1"
            
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    continue
                
                for movie in response.json().get('results', [])[:3]:  # Réduire à 3 par genre pour éviter trop de données
                    movie_details = get_movie_details_complete(movie, f"genre_{genre_id}_{year}")
                    if movie_details:
                        year_movies.append(movie_details)
                
            except Exception as e:
                print(f"    Erreur genre: {e}")
                continue
        
        # Supprimer doublons année
        unique_year_movies = {movie['id']: movie for movie in year_movies if movie}.values()
        movies.extend(list(unique_year_movies))
        print(f"  → {len(unique_year_movies)} films uniques")
    
    # Supprimer doublons globaux
    unique_movies = {movie['id']: movie for movie in movies}.values()
    final_movies = list(unique_movies)
    
    csv_utils.generate_csv("movies_comprehensive", final_movies)
    print(f"TOTAL FILMS: {len(final_movies)}")
    return final_movies

def get_tv_shows_comprehensive(start_year=1980, end_year=2024):
    """Récupération exhaustive de séries par année"""
    if csv_utils.csv_exists("tv_shows_comprehensive"):
        print("Chargement des séries depuis le CSV...")
        return csv_utils.get_data_from_csv("tv_shows_comprehensive")
    
    tv_shows = []
    
    sort_criteria = [
        'popularity.desc',
        'vote_average.desc',
        'vote_count.desc'
    ]
    
    tv_genres = [10759, 16, 35, 80, 99, 18, 10751, 14, 27, 10762, 9648, 10763, 10764, 10765, 10766, 10767, 10768]
    
    for year in range(start_year, end_year + 1):
        print(f"=== Séries année {year} ===")
        year_shows = []
        
        # Par critères
        for sort_by in sort_criteria:
            print(f"  Tri: {sort_by}")
            for page in range(1, 2):  # 1 page par critère
                url = f"{BASE_URL}/discover/tv?api_key={API_KEY}&language=en-US&sort_by={sort_by}&first_air_date_year={year}&vote_count.gte=5&page={page}"
                
                try:
                    response = requests.get(url)
                    if response.status_code != 200:
                        continue
                    
                    for tv in response.json().get('results', []):
                        tv_details = get_tv_details_complete(tv, f"{sort_by}_{year}")
                        if tv_details:
                            year_shows.append(tv_details)
                    
                except Exception as e:
                    print(f"    Erreur: {e}")
                    continue
        
        # Par genres
        for genre_id in tv_genres[:6]:
            print(f"  Genre TV: {genre_id}")
            url = f"{BASE_URL}/discover/tv?api_key={API_KEY}&language=en-US&sort_by=popularity.desc&first_air_date_year={year}&with_genres={genre_id}&page=1"
            
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    continue
                
                for tv in response.json().get('results', [])[:3]:
                    tv_details = get_tv_details_complete(tv, f"genre_{genre_id}_{year}")
                    if tv_details:
                        year_shows.append(tv_details)
                
            except Exception as e:
                continue
        
        unique_year_shows = {show['id']: show for show in year_shows if show}.values()
        tv_shows.extend(list(unique_year_shows))
        print(f"  → {len(unique_year_shows)} séries uniques")
    
    unique_shows = {show['id']: show for show in tv_shows}.values()
    final_shows = list(unique_shows)
    
    csv_utils.generate_csv("tv_shows_comprehensive", final_shows)
    print(f"TOTAL SÉRIES: {len(final_shows)}")
    return final_shows

def get_movie_details_complete(movie_data, source_category):
    """Récupération complète des détails d'un film"""
    movie_id = movie_data.get('id')
    if not movie_id:
        return None
    
    try:
        # Détails du film
        details_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        details_response = requests.get(details_url)
        
        if details_response.status_code != 200:
            return None
        
        details = details_response.json()
        
        # Crédits (optionnel)
        credits = {}
        try:
            credits_url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={API_KEY}"
            credits_response = requests.get(credits_url)
            if credits_response.status_code == 200:
                credits = credits_response.json()
        except:
            pass
        
        return {
            'id': movie_id,
            'title': details.get('title'),
            'rating': details.get('vote_average'),
            'votes': details.get('vote_count'),
            'genre_ids': movie_data.get('genre_ids', []),
            'language': details.get('original_language'),
            'release_date': details.get('release_date'),
            'runtime': details.get('runtime'),
            'budget': details.get('budget', 0),
            'revenue': details.get('revenue', 0),
            'popularity': details.get('popularity'),
            'overview': details.get('overview', '')[:300],
            'production_countries': [c['name'] for c in details.get('production_countries', [])],
            'production_companies': [c['name'] for c in details.get('production_companies', [])][:3],
            'director': get_director_from_credits(credits),
            'main_actors': get_main_actors_from_credits(credits),
            'source_category': source_category,
            'type': 'movie'
        }
    except Exception as e:
        print(f"Erreur film {movie_id}: {e}")
        return None

def get_tv_details_complete(tv_data, source_category):
    """Récupération complète des détails d'une série"""
    tv_id = tv_data.get('id')
    if not tv_id:
        return None
    
    try:
        details_url = f"{BASE_URL}/tv/{tv_id}?api_key={API_KEY}&language=en-US"
        details_response = requests.get(details_url)
        
        if details_response.status_code != 200:
            return None
        
        details = details_response.json()
        
        credits = {}
        try:
            credits_url = f"{BASE_URL}/tv/{tv_id}/credits?api_key={API_KEY}"
            credits_response = requests.get(credits_url)
            if credits_response.status_code == 200:
                credits = credits_response.json()
        except:
            pass
        
        return {
            'id': tv_id,
            'title': details.get('name'),
            'rating': details.get('vote_average'),
            'votes': details.get('vote_count'),
            'genre_ids': tv_data.get('genre_ids', []),
            'language': details.get('original_language'),
            'release_date': details.get('first_air_date'),
            'runtime': details.get('episode_run_time')[0] if details.get('episode_run_time') else None,
            'budget': None,
            'revenue': None,
            'popularity': details.get('popularity'),
            'overview': details.get('overview', '')[:300],
            'number_of_seasons': details.get('number_of_seasons'),
            'number_of_episodes': details.get('number_of_episodes'),
            'production_countries': [c['name'] for c in details.get('production_countries', [])],
            'networks': [n['name'] for n in details.get('networks', [])][:3],
            'creators': [c['name'] for c in details.get('created_by', [])],
            'main_actors': get_main_actors_from_credits(credits),
            'source_category': source_category,
            'type': 'tv'
        }
    except Exception as e:
        print(f"Erreur série {tv_id}: {e}")
        return None

def get_director_from_credits(credits):
    """Extrait le réalisateur principal"""
    try:
        crew = credits.get('crew', [])
        directors = [person['name'] for person in crew if person.get('job') == 'Director']
        return directors[0] if directors else None
    except:
        return None

def get_main_actors_from_credits(credits):
    """Extrait les 5 acteurs principaux"""
    try:
        cast = credits.get('cast', [])
        return [actor['name'] for actor in cast[:5]]
    except:
        return []

# === 3. FONCTION PRINCIPALE D'ENRICHISSEMENT ===
def enrich_comprehensive_data(data, genre_map):
    """Enrichit les données avec les noms de genres"""
    df = pd.DataFrame(data)
    
    # Convertir genre_ids si nécessaire
    if 'genre_ids' in df.columns:
        df['genre_ids'] = df['genre_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
        )
        
        # Mapper vers les noms de genres
        df['genres'] = df['genre_ids'].apply(
            lambda ids: [genre_map.get(int(i)) for i in ids if genre_map.get(int(i))] if ids else []
        )
    
    return df

# === 4. FONCTION PRINCIPALE ===
def get_comprehensive_dataset():
    """Fonction principale pour récupérer un dataset exhaustif"""
    print("=== RÉCUPÉRATION DATASET EXHAUSTIF ===")
    
    # 1. Genres
    print("Récupération des genres...")
    genre_map = get_genres()
    
    # 2. Films exhaustifs
    print("Récupération des films...")
    movies = get_movies_comprehensive(1980, 2024)
    
    # 3. Séries exhaustives  
    print("Récupération des séries...")
    tv_shows = get_tv_shows_comprehensive(1980, 2024)
    
    # 4. Combinaison et enrichissement
    print("Enrichissement des données...")
    all_content = movies + tv_shows
    
    # Enrichir avec les genres
    for content in all_content:
        if 'genre_ids' in content and content['genre_ids']:
            content['genres'] = [genre_map.get(gid) for gid in content['genre_ids'] if genre_map.get(gid)]
        else:
            content['genres'] = []
    
    # Créer DataFrame final
    df = pd.DataFrame(all_content)
    
    # Exploser les genres pour les analyses
    df_exploded = df.explode('genres').reset_index(drop=True)
    df_exploded = df_exploded[df_exploded['genres'].notna()]
    
    # Sauvegarder
    csv_utils.generate_csv("comprehensive_dataset", df)
    csv_utils.generate_csv("comprehensive_dataset_exploded", df_exploded)
    
    print(f"Dataset final: {len(movies)} films + {len(tv_shows)} séries = {len(all_content)} contenus")
    print(f"Après explosion des genres: {len(df_exploded)} entrées")
    
    return df_exploded

# === TOUTES VOS FONCTIONS DE VISUALISATION (inchangées) ===
def plot_avg_rating_by_genre(df):
    genre_rating = df.groupby('genres')['rating'].mean().sort_values(ascending=False)
    genre_rating.plot(kind='bar', color='skyblue', figsize=(10,5))
    plt.title('Note moyenne par genre')
    plt.xlabel('Genre')
    plt.ylabel('Note moyenne')
    plt.tight_layout()
    plt.show()

def plot_budget_vs_rating(df):
    df = df.copy()
    df = df[df['type'] == 'movie']
    df = df[(df['budget'] > 0) & (df['rating'].notnull())]

    plt.figure(figsize=(10, 6))
    plt.scatter(df['budget'], df['rating'], alpha=0.5, color='darkcyan')
    plt.title("Budget vs Note (Films)")
    plt.xlabel("Budget (USD)")
    plt.ylabel("Note moyenne")
    plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_genres_over_time(df):
    df = df.copy()
    df = df[df['release_date'].notnull()]
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df[df['year'].between(1980, 2024)]

    genre_counts = df.groupby(['year', 'genres']).size().unstack(fill_value=0)
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
    df = df.copy()
    genre_counts = df.groupby(['type', 'genres']).size().unstack(fill_value=0)

    genre_counts.T.plot(kind='bar', figsize=(14,6), colormap='Set2')
    plt.title("Répartition des genres (Films vs Séries)")
    plt.xlabel("Genre")
    plt.ylabel("Nombre de contenus")
    plt.tight_layout()
    plt.show()

def plot_budget_evolution_over_time(df):
    df = df.copy()
    df = df[df['type'] == 'movie']
    df = df[df['budget'] > 0]
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df[df['year'].notnull()]
    
    budget_by_year = df.groupby('year')['budget'].mean()
    
    plt.figure(figsize=(14, 6))
    budget_by_year.plot(kind='line', color='gold', marker='o', markersize=4)
    plt.title("Évolution du budget moyen des films par année")
    plt.xlabel("Année")
    plt.ylabel("Budget moyen (USD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_most_profitable_genres(df):
    df = df.copy()
    df = df[df['type'] == 'movie']
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    df['roi'] = df['revenue'] / df['budget']
    
    genre_roi = df.groupby('genres')['roi'].mean().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    genre_roi.plot(kind='bar', color='lightcoral')
    plt.title("ROI moyen par genre (Top 10)")
    plt.xlabel("Genre")
    plt.ylabel("ROI moyen")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_rating_vs_popularity_by_genre(df):
    df = df.copy()
    df = df[df['rating'].notnull()]
    
    top_genres = df['genres'].value_counts().head(5).index
    df_top = df[df['genres'].isin(top_genres)]
    
    plt.figure(figsize=(12, 8))
    for genre in top_genres:
        genre_data = df_top[df_top['genres'] == genre]
        plt.scatter(genre_data['rating'], genre_data['votes'], 
                   alpha=0.6, label=genre, s=30)
    
    plt.xlabel("Note moyenne")
    plt.ylabel("Nombre de votes")
    plt.title("Relation Note vs Popularité par Genre")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_seasonal_releases(df):
    df = df.copy()
    df = df[df['release_date'].notnull()]
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['month'] = df['release_date'].dt.month
    
    def get_season(month):
        if month in [12, 1, 2]: return 'Hiver'
        elif month in [3, 4, 5]: return 'Printemps'
        elif month in [6, 7, 8]: return 'Été'
        else: return 'Automne'
    
    df['season'] = df['month'].apply(get_season)
    
    season_counts = df.groupby(['season', 'type']).size().unstack(fill_value=0)
    
    season_counts.plot(kind='bar', figsize=(10, 6), colormap='viridis')
    plt.title("Nombre de sorties par saison")
    plt.xlabel("Saison")
    plt.ylabel("Nombre de contenus")
    plt.xticks(rotation=45)
    plt.legend(title='Type')
    plt.tight_layout()
    plt.show()

def plot_most_profitable_genres_by_year(df):
    df = df.copy()
    df = df[df['type'] == 'movie']
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df[df['year'].notnull()]
    df = df[df['year'].between(1990, 2024)]
    
    df['roi'] = df['revenue'] / df['budget']
    
    genre_year_counts = df.groupby(['year', 'genres']).size()
    valid_combinations = genre_year_counts[genre_year_counts >= 3].index
    
    df_filtered = df.set_index(['year', 'genres']).loc[valid_combinations].reset_index()
    
    roi_stats = df_filtered.groupby(['year', 'genres']).agg({
        'roi': ['mean', 'median', 'count']
    }).round(2)
    
    roi_stats.columns = ['roi_mean', 'roi_median', 'count']
    roi_stats = roi_stats.reset_index()
    
    pivot_data = roi_stats.pivot(index='year', columns='genres', values='roi_median')
    pivot_data = pivot_data.fillna(0)
    
    genre_frequency = roi_stats['genres'].value_counts().head(8)
    top_genres = genre_frequency.index
    
    plt.figure(figsize=(16, 10))
    
    pivot_filtered = pivot_data[top_genres].fillna(0)
    
    for genre in top_genres:
        plt.plot(pivot_filtered.index, pivot_filtered[genre], 
                marker='o', linewidth=2, label=genre, alpha=0.8)
    
    plt.title("Évolution des genres les plus rentables par année (ROI médian)\n(Minimum 3 films par genre/année)", fontsize=16)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("ROI médian", fontsize=12)
    plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Nombre de films par genre (top 10):")
    print(df['genres'].value_counts().head(10))
    print(f"\nNombre total de combinaisons année/genre filtrées: {len(valid_combinations)}")

def plot_tv_rating_by_season(df, max_seasons=30):
    """Affiche l'évolution de la note moyenne par saison pour chaque série (simulation, valeurs aberrantes filtrées)"""
    tv_df = df[df['type'] == 'tv'].copy()
    if 'number_of_seasons' not in tv_df.columns or 'rating' not in tv_df.columns:
        print("Colonnes nécessaires manquantes.")
        return

    # Filtrer les séries avec un nombre de saisons raisonnable
    tv_df = tv_df[(tv_df['number_of_seasons'].notnull()) & 
                  (tv_df['number_of_seasons'] > 1) & 
                  (tv_df['number_of_seasons'] <= max_seasons)]

    for _, row in tv_df.iterrows():
        ratings = np.linspace(row['rating'], row['rating'] - np.random.uniform(0, 2), int(row['number_of_seasons']))
        plt.plot(range(1, int(row['number_of_seasons'])+1), ratings, alpha=0.3, color='blue')
    plt.xlabel("Saison")
    plt.ylabel("Note moyenne (simulée)")
    plt.title("Évolution de la note moyenne par saison (simulation)")
    plt.show()

def plot_tv_rating_by_season_viz(df, max_seasons=10):
    """Courbe de la moyenne simulée des notes par saison avec écart-type"""
    tv_df = df[df['type'] == 'tv'].copy()
    if 'number_of_seasons' not in tv_df.columns or 'rating' not in tv_df.columns:
        print("Colonnes nécessaires manquantes.")
        return

    tv_df = tv_df[(tv_df['number_of_seasons'].notnull()) & 
                  (tv_df['number_of_seasons'] > 1) & 
                  (tv_df['number_of_seasons'] <= max_seasons)]

    # Stocker toutes les notes simulées par saison
    ratings_by_season = {}
    for _, row in tv_df.iterrows():
        n_seasons = int(row['number_of_seasons'])
        ratings = np.linspace(row['rating'], row['rating'] - np.random.uniform(0, 2), n_seasons)
        for season in range(1, n_seasons + 1):
            ratings_by_season.setdefault(season, []).append(ratings[season-1])

    # Calculer moyenne et std par saison
    seasons = sorted(ratings_by_season.keys())
    means = [np.mean(ratings_by_season[s]) for s in seasons]
    stds = [np.std(ratings_by_season[s]) for s in seasons]

    plt.figure(figsize=(10, 6))
    plt.plot(seasons, means, color='royalblue', label='Note moyenne')
    plt.fill_between(seasons, np.array(means)-np.array(stds), np.array(means)+np.array(stds), 
                     color='royalblue', alpha=0.2, label='Écart-type')
    plt.title("Évolution moyenne de la note par saison (simulation)")
    plt.xlabel("Saison")
    plt.ylabel("Note moyenne")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_best_movie_duration_vs_rating(df):
    """Visualisation de la relation entre la durée des films et leur note"""
    df = df.copy()
    df = df[(df['type'] == 'movie') & (df['runtime'].notnull()) & (df['rating'].notnull())]
    df = df[(df['runtime'] > 40) & (df['runtime'] < 240)]  # Filtre les durées aberrantes

    plt.figure(figsize=(12, 7))
    plt.scatter(df['runtime'], df['rating'], alpha=0.3, color='teal', label='Films')
    
    # Courbe de tendance polynomiale (degré 2)
    z = np.polyfit(df['runtime'], df['rating'], 2)
    p = np.poly1d(z)
    x = np.linspace(df['runtime'].min(), df['runtime'].max(), 200)
    plt.plot(x, p(x), color='orange', linewidth=2, label='Tendance (poly2)')
    
    # Affiche la durée optimale estimée
    best_duration = -z[1] / (2 * z[0])
    plt.axvline(best_duration, color='red', linestyle='--', label=f'Durée optimale ≈ {int(best_duration)} min')
    plt.title("Quelle durée pour la meilleure note ?")
    plt.xlabel("Durée du film (minutes)")
    plt.ylabel("Note moyenne")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    print(f"Durée optimale estimée pour la meilleure note : {int(best_duration)} minutes")


def plot_best_tv_episode_duration_vs_rating(df):
    """Visualisation de la relation entre la durée des épisodes de séries et leur note"""
    df = df.copy()
    df = df[(df['type'] == 'tv') & (df['runtime'].notnull()) & (df['rating'].notnull())]
    df = df[(df['runtime'] > 10) & (df['runtime'] < 120)]  # Filtre les durées aberrantes

    plt.figure(figsize=(12, 7))
    plt.scatter(df['runtime'], df['rating'], alpha=0.3, color='purple', label='Séries')
    
    # Courbe de tendance polynomiale (degré 2)
    z = np.polyfit(df['runtime'], df['rating'], 2)
    p = np.poly1d(z)
    x = np.linspace(df['runtime'].min(), df['runtime'].max(), 200)
    plt.plot(x, p(x), color='orange', linewidth=2, label='Tendance (poly2)')
    
    # Vérification : s'assurer que la parabole a un maximum (coefficient a < 0)
    if z[0] >= 0:
        print("Attention: La courbe ne présente pas de maximum clair")
        best_duration = df['runtime'].median()  # Utiliser la médiane comme fallback
        plt.axvline(best_duration, color='red', linestyle='--', label=f'Durée médiane ≈ {int(best_duration)} min')
    else:
        best_duration = -z[1] / (2 * z[0])
        # Vérifier que la durée optimale est dans la plage des données
        if best_duration < df['runtime'].min() or best_duration > df['runtime'].max():
            print("Attention: La durée optimale calculée est hors de la plage des données")
            best_duration = df['runtime'].median()
            plt.axvline(best_duration, color='red', linestyle='--', label=f'Durée médiane ≈ {int(best_duration)} min')
        else:
            plt.axvline(best_duration, color='red', linestyle='--', label=f'Durée optimale ≈ {int(best_duration)} min')
    
    plt.title("Quelle durée d'épisode pour la meilleure note ? (Séries TV)")
    plt.xlabel("Durée moyenne d'un épisode (minutes)")
    plt.ylabel("Note moyenne")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_tv_seasons_vs_rating(df):
    """Visualisation de la relation entre le nombre de saisons et la note moyenne des séries"""
    df = df.copy()
    df = df[(df['type'] == 'tv') & (df['number_of_seasons'].notnull()) & (df['rating'].notnull())]
    df = df[(df['number_of_seasons'] > 1) & (df['number_of_seasons'] < 30)]  # Filtre les valeurs aberrantes

    plt.figure(figsize=(12, 7))
    plt.scatter(df['number_of_seasons'], df['rating'], alpha=0.3, color='darkorange', label='Séries')
    
    # Courbe de tendance polynomiale (degré 2)
    z = np.polyfit(df['number_of_seasons'], df['rating'], 2)
    p = np.poly1d(z)
    x = np.linspace(df['number_of_seasons'].min(), df['number_of_seasons'].max(), 200)
    plt.plot(x, p(x), color='blue', linewidth=2, label='Tendance (poly2)')
    
    # Affiche le nombre de saisons optimal estimé
    best_seasons = -z[1] / (2 * z[0])
    plt.axvline(best_seasons, color='red', linestyle='--', label=f'Nb optimal ≈ {int(best_seasons)} saisons')
    plt.title("Quel nombre de saisons pour la meilleure note ? (Séries TV)")
    plt.xlabel("Nombre de saisons")
    plt.ylabel("Note moyenne")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    print(f"Nombre de saisons optimal estimé pour la meilleure note : {int(best_seasons)}")

def plot_tv_season_total_duration_vs_rating(df):
    """Visualisation de la relation entre la durée totale d'une saison et la note des séries"""
    df = df.copy()
    df = df[(df['type'] == 'tv') & (df['runtime'].notnull()) & (df['rating'].notnull()) & (df['number_of_episodes'].notnull())]
    
    # Calculer la durée totale d'une saison (durée épisode × nombre d'épisodes / nombre de saisons)
    df['episodes_per_season'] = df['number_of_episodes'] / df['number_of_seasons'].fillna(1)
    df['season_total_duration'] = df['runtime'] * df['episodes_per_season']
    
    # Filtrer les valeurs aberrantes
    df = df[(df['season_total_duration'] > 200) & (df['season_total_duration'] < 2000)]
    df = df[df['season_total_duration'].notnull()]

    plt.figure(figsize=(12, 7))
    plt.scatter(df['season_total_duration'], df['rating'], alpha=0.3, color='mediumseagreen', label='Séries')
    
    # Courbe de tendance polynomiale (degré 2)
    z = np.polyfit(df['season_total_duration'], df['rating'], 2)
    p = np.poly1d(z)
    x = np.linspace(df['season_total_duration'].min(), df['season_total_duration'].max(), 200)
    plt.plot(x, p(x), color='orange', linewidth=2, label='Tendance (poly2)')
    
    # Vérification : s'assurer que la parabole a un maximum (coefficient a < 0)
    if z[0] >= 0:
        print("Attention: La courbe ne présente pas de maximum clair pour la durée de saison")
        best_duration = df['season_total_duration'].median()
        plt.axvline(best_duration, color='red', linestyle='--', label=f'Durée médiane ≈ {int(best_duration)} min')
    else:
        best_duration = -z[1] / (2 * z[0])
        # Vérifier que la durée optimale est dans la plage des données
        if best_duration < df['season_total_duration'].min() or best_duration > df['season_total_duration'].max():
            print("Attention: La durée optimale calculée est hors de la plage des données")
            best_duration = df['season_total_duration'].median()
            plt.axvline(best_duration, color='red', linestyle='--', label=f'Durée médiane ≈ {int(best_duration)} min')
        else:
            plt.axvline(best_duration, color='red', linestyle='--', label=f'Durée optimale ≈ {int(best_duration)} min')
    
    plt.title("Quelle durée totale de saison pour la meilleure note ? (Séries TV)")
    plt.xlabel("Durée totale d'une saison (minutes)")
    plt.ylabel("Note moyenne")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

# === EXÉCUTION PRINCIPALE ===
if __name__ == "__main__":
    # Récupération du dataset exhaustif
    all_df = pd.read_csv("CSV/comprehensive_dataset_exploded.csv")
    all_df_non_exploded = pd.read_csv("CSV/comprehensive_dataset.csv")

    # Visualisations
    print("\nGénération des visualisations...")
    
    # Décommenter les graphiques que vous voulez voir :
    plot_tv_season_total_duration_vs_rating(all_df_non_exploded)
    plot_best_tv_episode_duration_vs_rating(all_df_non_exploded)
    # plot_avg_rating_by_genre(all_df)
    # plot_budget_vs_rating(all_df) 
    plot_best_movie_duration_vs_rating(all_df_non_exploded)
    #plot_genres_over_time(all_df)
    # plot_revenue_vs_budget(all_df)
    # plot_roi_distribution(all_df)
    # plot_languages_distribution(all_df)
    # plot_release_volume(all_df)
    # plot_genre_distribution_by_type(all_df)
    # plot_budget_evolution_over_time(all_df)
    # plot_most_profitable_genres(all_df)
    # plot_rating_vs_popularity_by_genre(all_df)
    # plot_seasonal_releases(all_df)
    #plot_most_profitable_genres_by_year(all_df)
    plot_tv_rating_by_season_viz(all_df_non_exploded)