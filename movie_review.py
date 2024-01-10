import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from sklearn.metrics import mean_squared_error

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

def preprocess_df(extracted_data, max_review_length, max_movie_length):
    extracted_df = pd.DataFrame(extracted_data)
    
    # Pad sequences with variable lengths
    extracted_df['review_token'] = pad_sequences(extracted_df['review_token'], maxlen=max_review_length, padding='post', truncating='post').tolist()
    extracted_df['movie_names_token'] = pad_sequences(extracted_df['movie_names_token'], maxlen=max_movie_length, padding='post', truncating='post').tolist()

    return extracted_df

def build_model(max_review_length, max_movie_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=30522, output_dim=16, input_length=max_review_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(max_movie_length, activation='softmax')  # Assuming max_movie_length is the vocabulary size
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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

    extracted_df = preprocess_df(extracted_data, max_review_length, max_movie_length)

    token_df = extracted_df[['review_token', 'movie_names_token']][:5]
    # Use tolist() to convert lists to NumPy arrays
    X = np.array([np.array(val) for val in token_df['review_token'].tolist()])
    Y = np.array([np.array(token_list) for token_list in token_df['movie_names_token']])



    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    Y_train = np.array([np.array(token_list) for token_list in y_train])
    Y_test = np.array([np.array(token_list) for token_list in y_test])

    # Check the shape of Y_train and Y_test
    print(f"Shape of Y_train before reshaping: {Y_train.shape}")
    print(f"Shape of Y_test before reshaping: {Y_test.shape}")

    # Reshape Y_train and Y_test
    Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1], 1))

    # Check the shape of Y_train and Y_test after reshaping
    print(f"Shape of Y_train after reshaping: {Y_train.shape}")
    print(f"Shape of Y_test after reshaping: {Y_test.shape}")

    # Build a simple neural network using TensorFlow's Keras API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(max_review_length,)),
        tf.keras.layers.Dense(max_movie_length, activation='softmax'),
        tf.keras.layers.Reshape((max_movie_length, 1))
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])




    # Train the model
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))


    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Flatten y_pred and y_test
    y_pred_flat = y_pred.reshape((y_pred.shape[0], -1))
    y_test_flat = Y_test.reshape((Y_test.shape[0], -1))

    # Evaluate the model
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    print(f'Mean Squared Error: {mse}')
        
if __name__ == "__main__":
    main()
