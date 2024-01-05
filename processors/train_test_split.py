import numpy as np

def train_test_split(df):
    X = np.array(df['review_token'].tolist())
    Y = np.array(df['movie_names_token'].tolist())
    output_size = Y.shape[1]

    return train_test_split(X, Y, test_size=0.2, random_state=42)

def output_size(df):
    Y = np.array(df['movie_names_token'].tolist())
    return Y.shape[1]