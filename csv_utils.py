import pandas as pd
import os
import ast

csv_names = {
    "genres" : "CSV/genres.csv", 
    "movies" : "CSV/movies.csv",
    "tv_shows" : "CSV/tv_shows.csv",
    "main" : "CSV/main.csv",
    "final" : "CSV/final.csv"
}

def generate_csv(key_file_name, data):
    file_name = csv_names[key_file_name]
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False, encoding='utf-8')
    print(f"CSV généré : {file_name}")

def csv_exists(key_file_name):
    file_name = csv_names[key_file_name]
    if os.path.exists(file_name):
        return True
    return False

def get_data_from_csv(key_file_name):
    file_name = csv_names[key_file_name]
    df = pd.read_csv(file_name)

    if key_file_name == "genres":
        return dict(zip(df["id"], df["name"]))
    
    if key_file_name == "movies":
        # Remet les champs de type liste (comme genre_ids) dans leur format original
        if "genre_ids" in df.columns:
            df["genre_ids"] = df["genre_ids"].apply(ast.literal_eval)
        return df.to_dict(orient="records")
    
    if key_file_name == "tv_shows":
        if "genre_ids" in df.columns:
            df["genre_ids"] = df["genre_ids"].apply(ast.literal_eval)
        return df.to_dict(orient="records")

    if key_file_name == "main":
        return df.to_dict(orient="records")

    return []