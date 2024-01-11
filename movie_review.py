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
    print(f"max review length pad token: {max_review_length}, max movie name token: {max_movie_length}")
    extracted_df = preprocess_df(extracted_data, max_review_length, max_movie_length)
    # Split the data into features (X) and target (y)
    X = extracted_df[['review_token']]
    Y = extracted_df[['movie_names_token']]
    X = X.astype(np.float32)
    Y = Y.astype(np.int32)
    print(f"X shape: {X.shape} and Y shape: {Y.shape}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1, activation='linear')  # Linear activation for regression
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse']) 

    # Print the model summary
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    mse = model.evaluate(X_test, y_test)
    print(f'Mean Squared Error on Test Set: {mse}')
    
if __name__ == "__main__":
    main()

