import numpy as np
from sklearn.model_selection import train_test_split


def data_split(df):
    X = np.array(df['review_token'].tolist())
    Y = np.array(df['movie_names_token'].tolist())

    return train_test_split(X, Y, test_size=0.2, random_state=42)

def get_output_size(df):
    Y = np.array(df['movie_names_token'].tolist())
    return Y.shape[1]

def get_input_shape(df):
    X = np.array(df['review_token'].tolist())
    return X.shape[1]