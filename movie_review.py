import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf
from data_loader.load_imdb import load_imdb_dataset
from processors.tokenizer import preprocess_review_data, preprocess_df
from training_strategy.distributed_training import grid_search
from models.model_builder import MultiHeadAttention
import os

def main():
    tf.keras.backend.clear_session()

    csv_file_path = './IMDB Dataset.csv'
    imdb_df = load_imdb_dataset(csv_file_path)

    movie_name_pattern = re.compile(r'"([^"]+)"')

    extracted_data = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for review in imdb_df['review']:
        data = preprocess_review_data(review, tokenizer, movie_name_pattern)
        if data:
            extracted_data.append(data)

    if not extracted_data:
        print("No valid data found. Exiting.")
        return

    max_review_length = max([len(data['review_token']) for data in extracted_data])
    max_movie_length = max([len(data['movie_names_token']) for data in extracted_data])
    print(f"max review length pad token: {max_review_length}, max movie name token: {max_movie_length}")
    extracted_df = preprocess_df(extracted_data, max_review_length, max_movie_length)
    # Split the data into features (X) and target (y)
    X = extracted_df[['review_token']]
    Y = extracted_df[['movie_names_token']]
    print(f"X input: {X.iloc[0]}")
    print(f"Y input: {Y.iloc[0]}")
    print(f"X is type of : {type(X)}, Y is type of : {type(Y)}")
    # Assuming 'review_token' is a list within each item of 'X'
    X = np.array([item[0] for item in extracted_df['review_token']], dtype=np.float32)

    # Assuming 'movie_names_token' is a list within each item of 'Y'
    Y = np.array([item[0] for item in extracted_df['movie_names_token']], dtype=np.int32)


    print(f"X input: {X.iloc[0]}")
    print(f"Y input: {Y.iloc[0]}")
    print(f"X is type of : {type(X)}, Y is type of : {type(Y)}")


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_sequences = X_train['review_token'].tolist()
    Y_train_sequences = y_train['movie_names_token'].tolist()
    X_test_sequences = X_test['review_token'].tolist()
    Y_test_sequences = y_test['movie_names_token'].tolist()

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(max_review_length,)),
        tf.keras.layers.Dense(1)  # Output layer with 1 neuron for regression task
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    mse = model.evaluate(X_test, y_test)
    print(f'Mean Squared Error on Test Set: {mse}')
    
if __name__ == "__main__":
    main()

