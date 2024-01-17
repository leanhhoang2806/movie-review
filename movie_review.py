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
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class SimpleModel(tf.keras.Model):
    def __init__(self, in_shape, out_shape):
        super(SimpleModel, self).__init__()
        # rework this model for better accuracy
        self.dense1 = Dense(64, activation='relu', input_shape=(in_shape,))
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(out_shape, activation='relu')  # Assuming 9 as the output dimension based on Y.shape[1]

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

    extracted_df = pd.DataFrame(extracted_data)
    focus_data = extracted_df[['review', 'movie_names']]
    print(focus_data.head())

    training_data = [ {"answer": row['review'], "question": row['movie_names']} for _, row in focus_data.iterrows()]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = []
    for example in training_data:
        input_text = f"Q: {example['question']} A: {example['answer']}"
        encoded_text = tokenizer.encode(input_text, return_tensors='pt')
        input_ids.append(encoded_text)

    input_ids = torch.cat(input_ids, dim=0)
    
    


    
if __name__ == "__main__":
    main()

