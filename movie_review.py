import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# # Create a sample pandas DataFrame with nested lists
# data = {'Features': [[1], [2], [3], [4], [5]],
#         'Target_Y': [[2], [4], [5], [4], [5]]}
# df = pd.DataFrame(data)

# # Convert the nested lists into separate columns
# df_expanded = pd.DataFrame(df['Features'].to_list(), columns=['Feature_X'])

# # Concatenate the expanded features with the target column
# df_processed = pd.concat([df_expanded, df['Target_Y']], axis=1)

# # Split the data into features (X) and target (y)
# X = df_processed[['Feature_X']]
# y = df_processed['Target_Y']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the input features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Convert the nested list target to a numpy array
# y_train_np = np.array(y_train.tolist())
# y_test_np = np.array(y_test.tolist())

# # Build a simple neural network model
# model = keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     layers.Dense(1)  # Output layer with 1 neuron for regression
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(X_train_scaled, y_train_np, epochs=50, batch_size=2, validation_data=(X_test_scaled, y_test_np))

# # Now, you can use the trained model to predict a new row with nested list input
# new_data = {'Feature_X': [[6]]}  # Nested list for the new row
# new_row = pd.DataFrame(new_data)

# # Convert the nested list to a NumPy array
# new_row_np = np.array(new_row['Feature_X'].tolist())

# # Make predictions on the new row
# prediction = model.predict(new_row_np)
# print(f'Predicted Target_Y for the new row: {prediction[0][0]}')








import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from sklearn.metrics import mean_squared_error
import string
from itertools import chain, product

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

def convert_tokens_to_words(tokenizer, tokens):
    words = tokenizer.convert_tokens_to_string(tokens)
    return " ".join(words)

def get_bert_token(word):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = tokenizer.encode(word, add_special_tokens=False, padding=True, truncation=True, max_length=512)
    return token_ids

def main():
    tf.keras.backend.clear_session()

    csv_file_path = './IMDB Dataset.csv'
    imdb_df = load_imdb_dataset(csv_file_path)
    movie_name_pattern = re.compile(r'"([^"]+)"')

    imdb_df = imdb_df[:10]

    extracted_data = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for review in imdb_df['review']:
    # Extract text between quotation marks using the defined pattern
        matches = re.findall(movie_name_pattern, review)

        # Filter out names that are not in capitalization format
        valid_names = [name for name in matches if name.istitle()]

        # Clean up punctuation at the beginning or end of each title
        cleaned_names = [name.strip(string.punctuation) for name in valid_names]

        # Append the review and cleaned text to the list if the list is not empty
        if cleaned_names and len(cleaned_names) == 1:
            corrected_name = cleaned_names[0].strip()
            if len(corrected_name.split()) == 2:
                # Get BERT token for movie name
                movie_name_tokens_nested = [get_bert_token(token) for token in corrected_name.split()]
                movie_name_tokens_flatten = list(chain(*movie_name_tokens_nested))

                # Get BERT token for review
                review_tokens = get_bert_token(review)
                extracted_data.append({'review_token': review_tokens, 'movie_names_token': movie_name_tokens_flatten,
                                    "review": review, "movie_names": corrected_name})

    extracted_df = pd.DataFrame(extracted_data)
    print(extracted_df.head())
    # print(extracted_df['review'].iloc[0])
    # print(extracted_df['movie_names'].iloc[0])

    # max_review_length = max([ len(i) for i in pd.Series(extracted_df['review_token'])])
    # max_movie_length = max([ len(i) for i in pd.Series(extracted_df['movie_names_token'])])
    # print(f"max_review length: {max_review_length}, max_movie_length: {max_movie_length}")

    # extracted_df = preprocess_df(extracted_data, max_review_length, max_movie_length)
    # review_token_series = pd.Series(extracted_df['review_token'])
    # all_length = set()
    # for i in review_token_series:
    #     all_length.add(len(i))
    # print(f"length of items in  review token: {all_length}")



    # token_df  = extracted_df[['review_token', 'movie_names_token']]
    # X = token_df[['review_token']]
    # Y = token_df['movie_names_token']

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # y_train_np = np.array(y_train.tolist())
    # y_test_np = np.array(y_test.to_list())
    # print(f"y_train_np_shape: {y_train_np.shape}")

    # model = keras.Sequential([
    #     layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    #     layers.Dense(y_train_np.shape[1])  # Output layer with the shape of y_train data
    # ])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(X_train, y_train_np, epochs=50, batch_size=32, validation_data=(X_test, y_test_np))

    # token_df = extracted_df[['review_token', 'movie_names_token']][:5]
    # # Use tolist() to convert lists to NumPy arrays
    # X = np.array([np.array(val) for val in token_df['review_token'].tolist()])
    # Y = np.array([np.array(token_list) for token_list in token_df['movie_names_token']])



    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Y_train = np.array([np.array(token_list) for token_list in y_train])
    # Y_test = np.array([np.array(token_list) for token_list in y_test])

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(input_dim=30522, output_dim=16, input_length=max_review_length),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.Dense(max_movie_length * 30522, activation='softmax'),  # Assuming max_movie_length is the vocabulary size
    #     tf.keras.layers.Reshape((max_movie_length, 30522))
    # ])


    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])






    # # Train the model
    # model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))


    # # Make predictions on the test set
    # y_pred = model.predict(X_test)

    # # Flatten y_pred and y_test
    # y_pred_flat = y_pred.reshape((y_pred.shape[0], -1))
    # y_test_flat = Y_test.reshape((Y_test.shape[0], -1))

    # # Evaluate the model
    # mse = mean_squared_error(y_test_flat, y_pred_flat)
    # print(f'Mean Squared Error: {mse}')

    # # Create a new DataFrame to store the results
    # results_df = pd.DataFrame(columns=['Original Review', 'Predicted Movie Names'])

    # # Iterate through each test example
    # for i in range(len(X_test)):
    #     original_review = imdb_df['review'].iloc[X_test[i].tolist()]
    #     predicted_movie_names_tokens = y_pred[i].argmax(axis=1)
    #     predicted_movie_names = convert_tokens_to_words(tokenizer, predicted_movie_names_tokens)

    #     # Append results to the DataFrame
    #     results_df = results_df.append({'Original Review': original_review, 'Predicted Movie Names': predicted_movie_names}, ignore_index=True)

    # print(results_df.head())

    
if __name__ == "__main__":
    main()
