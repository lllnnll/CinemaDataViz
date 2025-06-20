import pandas as pd
import os
import ast

csv_names = {
    "genres" : "CSV/genres.csv", 
    "movies" : "CSV/movies.csv",
    "main" : "CSV/main.csv"
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
    if file_name == "CSV/genres.csv":
        return dict(zip(df["id"], df["name"]))
    if file_name == "CSV/movies.csv":
        df["genre_ids"] = df["genre_ids"].apply(ast.literal_eval) # remettre la chaine ["12", "45"] au format liste
        return df.to_dict(orient="records")
