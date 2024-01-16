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
from tensorflow.keras.layers import Dense

# Multihead Attention layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = Dense(d_model)
        self.key_dense = Dense(d_model)
        self.value_dense = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]

        # Linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        scaled_attention = self.scaled_dot_product_attention(query, key, value)

        # Combine heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear layer
        output = self.dense(concat_attention)

        return output

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)

        return output

class MyModel(tf.keras.Model):
    def __init__(self, d_model, num_heads):
        super(MyModel, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(9)  # Assuming 9 as the output dimension based on Y.shape[1]
        ])

    def call(self, inputs):
        x = self.attention(inputs)
        x = self.ffn(x)
        return x


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
    model = MyModel(d_model=512, num_heads=8)  # Assuming d_model and num_heads based on your task


    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit({'query': X, 'key': X, 'value': X}, Y, epochs=10, verbose=1)

    # Evaluate the model
    predictions = model({'query': X, 'key': X, 'value': X})

    print(predictions.numpy().tolist()[:10])

    word_list = [tokenizer.decode(int(val)) for val in predictions[0].numpy()]

    print(word_list)

    # words_list = [tokenizer.decode(int(val)) for val in predictions.numpy().flatten()]
    
    # words_list = [print(word) for word in words_list]

    # word_list = [ ''.join(item) for item in words_list]
    # human_readable = ' '.join(word_list)


    # print(human_readable)
    

    
if __name__ == "__main__":
    main()

