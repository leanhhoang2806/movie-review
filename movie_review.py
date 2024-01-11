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
from tensorflow import keras
from tensorflow.keras import layers

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
    extracted_df = preprocess_df(extracted_data, max_review_length, max_movie_length)
    # Split the data into features (X) and target (y)
    X = tf.constant(extracted_df['review_token'].tolist())
    Y = tf.constant(extracted_df['movie_names_token'].tolist())
    input_shape = len(extracted_df['review_token'].tolist()[0])
    output_shape = len(extracted_df['movie_names_token'].tolist()[0])

    print(f"X.shape: {len(extracted_df['review_token'].tolist()[0])}")
    print(f"Y.shape: {len(extracted_df['movie_names_token'].tolist()[0])}")
    # Define the neural network model
    model = keras.Sequential([
        layers.Dense(units=64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(units=output_shape)  # Output layer with 2 units for the target values
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(X, Y, epochs=1, verbose=1)

    # Make predictions
    predictions = model.predict(X)


    
if __name__ == "__main__":
    main()

