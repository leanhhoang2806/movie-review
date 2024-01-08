
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf
from data_loader.load_imdb import load_imdb_dataset
from processors.tokenizer import preprocess_review_data, preprocess_df
from training_strategy.distributed_training import grid_search
from predictions.predictor import predict_movie_name

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
        'num_layers': range(1, 3),
        'layer_size': [256 * i for i in range(1, 5)],
        'dropout_rate': [0.2 * i for i in range(1, 5)],
        'num_heads': [i * 2  for i in range(1, 3)],
    }

    best_accuracy, best_params, best_model = grid_search(param_grid, X_train, y_train, X_test, y_test, X.shape[1], output_size)

    print(f'Best Model Accuracy: {best_accuracy} with best params: {best_params}')

    predicted_df = predict_movie_name(extracted_df, best_model)
    print(predicted_df.head())
    predicted_df.to_csv('predicted_movie_names.csv', index=False)

if __name__ == "__main__":
    main()


