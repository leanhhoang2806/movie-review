
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
from transformers import pipeline


def main():
    tf.keras.backend.clear_session()

    csv_file_path = './IMDB Dataset.csv'
    imdb_df = load_imdb_dataset(csv_file_path)
    review_text = imdb_df.iloc[0]['review']
    print(f"Text To predict : {review_text}")
    ner_pipeline = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english', tokenizer='dbmdz/bert-large-cased-finetuned-conll03-english')
    entities = ner_pipeline(review_text)
    for entity in entities:
        if entity['entity'] == 'ORG':
            print(entity['word'])



if __name__ == "__main__":
    main()


