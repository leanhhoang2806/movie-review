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

    # Define the neural network model
    model = keras.Sequential([
        layers.Dense(units=64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(units=output_shape)  # Output layer with 2 units for the target values
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X, Y, epochs=1, verbose=1)

    # Make predictions
    print(f"test_review_token: {test_review_token}")
    predictions = model.predict(X)
    # decoded_predictions = tokenizer.decode(predictions[0], skip_special_tokens=True)
    # print(f"Given a review {test_review_token},  \n the movie name prediction is {decoded_predictions}, \n the actual movie name is  ")


    
if __name__ == "__main__":
    main()

