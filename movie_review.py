import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf
from data_loader.load_imdb import load_imdb_dataset
from processors.tokenizer import preprocess_review_data, preprocess_df
from training_strategy.distributed_training import grid_search
from tensorflow.keras.layers import LSTM, MultiHeadAttention
import os
from tensorflow import keras
from tensorflow.keras import layers

def main():
    tf.keras.backend.clear_session()

    csv_file_path = './IMDB Dataset.csv'
    imdb_df = load_imdb_dataset(csv_file_path)
    print(f"target for testing predictionis : {imdb_df.iloc[0]}")

    movie_name_pattern = re.compile(r'"([^"]+)"')

    extracted_data = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for review in imdb_df['review']:
        data = preprocess_review_data(review, tokenizer, movie_name_pattern)
        if data:
            extracted_data.append(data)

    max_review_length = max([len(data['review_token']) for data in extracted_data])
    max_movie_length = max([len(data['movie_names_token']) for data in extracted_data])
    extracted_df = preprocess_df(extracted_data, max_review_length, max_movie_length)
    
    test_review_token = extracted_df['review_token'].iloc[0]

    # Split the data into features (X) and target (y)
    X = tf.constant(extracted_df['review_token'].tolist())
    Y = tf.constant(extracted_df['movie_names_token'].tolist())
    input_shape = len(extracted_df['review_token'].tolist()[0])
    output_shape = len(extracted_df['movie_names_token'].tolist()[0])

    X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

    # Define the neural network model with an LSTM layer and MultiHeadAttention
    model = keras.Sequential([
        LSTM(units=64, activation='relu', input_shape=(input_shape, 1), return_sequences=True),
        MultiHeadAttention(num_heads=2, key_dim=64),  # Adjust num_heads and key_dim as needed
        layers.Dense(units=output_shape)  # Output layer with units matching target shape
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X, Y, epochs=1000, verbose=1)

    predictions = model.predict(X)
    decoded_predictions = tokenizer.decode(predictions[0], skip_special_tokens=True)
    print(f"the movie name prediction is {decoded_predictions}, \n the actual review is  {extracted_data[0]['review']} ")


    
if __name__ == "__main__":
    main()

