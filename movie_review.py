import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_imdb_dataset(file_path):
    imdb_df = pd.read_csv(file_path)
    return imdb_df

def preprocess_review_data(review, tokenizer, movie_name_pattern):
    review_tokens = tokenizer.encode(review, add_special_tokens=True)
    movie_names = re.findall(movie_name_pattern, review)
    if movie_names:
        movie_names_tokens = tokenizer.encode(movie_names[0], add_special_tokens=True)
        return {'review_token': review_tokens, 'movie_names_token': movie_names_tokens}
    else:
        return {'review_token': review_tokens, 'movie_names_token': []}

def preprocess_df(extracted_data, max_review_length):
    extracted_df = pd.DataFrame(extracted_data)
    extracted_df['review_token'] = pad_sequences(extracted_df['review_token'], maxlen=max_review_length, padding='post', truncating='post').tolist()
    return extracted_df

def build_model(max_review_length, max_movie_length):
    review_input = tf.keras.layers.Input(shape=(max_review_length,), name='review_token')
    movie_input = tf.keras.layers.Input(shape=(max_movie_length,), name='movie_names_token')

    review_embedding = tf.keras.layers.Embedding(input_dim=30522, output_dim=16)(review_input)
    movie_embedding = tf.keras.layers.Embedding(input_dim=30522, output_dim=16)(movie_input)

    lstm_review = tf.keras.layers.LSTM(32)(review_embedding)
    lstm_movie = tf.keras.layers.LSTM(32)(movie_embedding)

    merged = tf.keras.layers.concatenate([lstm_review, lstm_movie])
    dense1 = tf.keras.layers.Dense(8, activation='relu')(merged)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)

    model = tf.keras.Model(inputs=[review_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

    max_review_length = max([len(data['review_token']) for data in extracted_data])
    max_movie_length = max([len(data['movie_names_token']) for data in extracted_data])

    extracted_df = preprocess_df(extracted_data, max_review_length)
    token_df = extracted_df[['review_token', 'movie_names_token']]

    X_train_review, X_train_movie, X_test_review, X_test_movie, y_train, y_test = [], [], [], [], [], []

    for i, row in token_df.iterrows():
        review_tokens = row['review_token']
        movie_names_tokens = row['movie_names_token']

        if np.random.rand() < 0.8:  # 80% for training, 20% for testing
            X_train_review.append(review_tokens)
            X_train_movie.append(movie_names_tokens)
            y_train.append(1 if len(movie_names_tokens) > 0 else 0)
        else:
            X_test_review.append(review_tokens)
            X_test_movie.append(movie_names_tokens)
            y_test.append(1 if len(movie_names_tokens) > 0 else 0)

    X_train_review, X_train_movie, X_test_review, X_test_movie = np.array(X_train_review), np.array(X_train_movie), np.array(X_test_review), np.array(X_test_movie)

    model = build_model(max_review_length, max_movie_length)

    # Train the model
    model.fit({'review_token': X_train_review, 'movie_names_token': X_train_movie}, np.array(y_train), epochs=10, validation_data=({'review_token': X_test_review, 'movie_names_token': X_test_movie}, np.array(y_test)))

    # Evaluate the model
    loss, accuracy = model.evaluate({'review_token': X_test_review, 'movie_names_token': X_test_movie}, np.array(y_test))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
