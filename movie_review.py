import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf
from data_loader.load_imdb import load_imdb_dataset
from processors.tokenizer import preprocess_review_data, preprocess_df
from training_strategy.distributed_training import grid_search
from tensorflow.keras.layers import LSTM, Input, Layer, Dense
from tensorflow.keras.models import Model

import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

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

    class MultiHeadAttention(Layer):
        def __init__(self, num_heads, key_dim, **kwargs):
            super(MultiHeadAttention, self).__init__(**kwargs)
            self.num_heads = num_heads
            self.key_dim = key_dim
            self.head_size = key_dim // num_heads

            # Linear projections for query, key, and value
            self.query_dense = Dense(units=key_dim)
            self.key_dense = Dense(units=key_dim)
            self.value_dense = Dense(units=key_dim)

            # Output dense layer
            self.output_dense = Dense(units=key_dim)

        def call(self, inputs):
            # Split the queries, keys, and values into multiple heads
            queries = tf.concat(tf.split(self.query_dense(inputs), self.num_heads, axis=-1), axis=0)
            keys = tf.concat(tf.split(self.key_dense(inputs), self.num_heads, axis=-1), axis=0)
            values = tf.concat(tf.split(self.value_dense(inputs), self.num_heads, axis=-1), axis=0)

            # Compute the attention scores
            attention_scores = tf.matmul(queries, tf.transpose(keys, perm=[0, 2, 1]))
            attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))

            # Apply softmax to get attention weights
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)

            # Apply attention weights to values
            output = tf.matmul(attention_weights, values)

            # Concatenate and project back to the original dimension
            output = tf.concat(tf.split(output, self.num_heads, axis=0), axis=-1)
            output = self.output_dense(output)

            return output

    # Input layer
    input_layer = Input(shape=(input_shape, 1))

    # MultiHead Attention layer manually built
    multihead_attention_layer = MultiHeadAttention(num_heads=2, key_dim=64)(input_layer)

    # Output layer
    output_layer = Dense(units=output_shape)(multihead_attention_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X, Y, epochs=1000, verbose=1)

    predictions = model.predict(X)
    decoded_predictions = tokenizer.decode(predictions[0], skip_special_tokens=True)
    print(f"the movie name prediction is {decoded_predictions}, \n the actual review is  {extracted_data[0]['review']} ")


    
if __name__ == "__main__":
    main()

