import pandas as pd

def load_imdb_dataset(csv_file_path):
    return pd.read_csv(csv_file_path)