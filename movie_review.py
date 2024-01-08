
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf
from data_loader.load_imdb import load_imdb_dataset
from processors.tokenizer import preprocess_review_data, preprocess_df
from training_strategy.distributed_training import grid_search
from predictions.predictor import predict_movie_name
import shutil
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

    extracted_df = preprocess_df(extracted_data, max_review_length, max_movie_length)


    X = np.array(extracted_df['review_token'].tolist())
    Y = np.array(extracted_df['movie_names_token'].tolist())
    output_size = Y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    param_grid = {
        'num_layers': [1],
        'layer_size': [256],
        'dropout_rate': [0.2],
        'num_heads': [4],
    }

    best_accuracy, best_params, best_model = grid_search(param_grid, X_train, y_train, X_test, y_test, X.shape[1], output_size)
    container_path = '/app/best_model.h5'
    directory_outside_container = '/Documents/work/movie-review/best_model.h5'
    best_model.save(container_path)
    os.makedirs(directory_outside_container, exist_ok=True)

    # Copy the model file to the host location
    shutil.copy(container_path, directory_outside_container)
    print(f'Best Model Accuracy: {best_accuracy} with best params: {best_params}')

if __name__ == "__main__":
    main()


