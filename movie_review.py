import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf
from data_loader.load_imdb import load_imdb_dataset
from processors.tokenizer import preprocess_review_data, preprocess_df
from training_strategy.distributed_training import grid_search
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM

# Multihead Attention layer
class SimpleModel(tf.keras.Model):
    def __init__(self, shape):
        super(SimpleModel, self).__init__()
        self.dense1 = Dense(64, activation='relu', input_shape=(shape,))
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(9, activation='relu')  # Assuming 9 as the output dimension based on Y.shape[1]

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output

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
    

    # Split the data into features (X) and target (y)
    X = tf.constant(extracted_df['review_token'].tolist())
    Y = tf.constant(extracted_df['movie_names_token'].tolist())
    print(f"X.shape : {X.shape}, and Y.shape: {Y.shape}")

    # Instantiate the model
    # model = SimpleModel(d_model=512, num_heads=8) 


    # # Compile the model
    # model.compile(optimizer='adam', loss='mean_squared_error')

    # # Train the model
    # model.fit({'query': X, 'key': X, 'value': X}, Y, epochs=10, verbose=1)

    # # Evaluate the model
    # predictions = model({'query': X, 'key': X, 'value': X})

    # print(predictions.numpy().tolist()[0])

    # word_list = [tokenizer.decode(int(val)) for val in predictions.numpy().tolist()[0][0]]


    # print(imdb_df['review'].iloc[0])
    # print(word_list)

    
if __name__ == "__main__":
    main()

