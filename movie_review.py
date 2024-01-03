import pandas as pd
import re
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForTokenClassification
from transformers import TFBertForSequenceClassification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
from transformers import TFBertModel
from transformers import AdamW
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM

# Load the IMDb dataset
csv_file_path = './IMDB Dataset.csv'
imdb_df = pd.read_csv(csv_file_path)

# Define a regular expression pattern for extracting text between quotation marks
movie_name_pattern = re.compile(r'"([^"]+)"')

# Initialize an empty list to store extracted data
extracted_data = []

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_bert_token(word):
    token_ids = tokenizer.encode(word, add_special_tokens=False, padding=True, truncation=True, max_length=512)
    return token_ids

# Loop through each review and extract text between quotation marks
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
        if len(corrected_name.split())  == 2:
            # Get BERT token for movie name
            movie_name_tokens_nested = [get_bert_token(token) for token in corrected_name.split()]
            movie_name_tokens_flatten = list(chain(*movie_name_tokens_nested))

            # Get BERT token for review
            review_tokens = get_bert_token(review)
            extracted_data.append({'review_token': review_tokens, 'movie_names_token': movie_name_tokens_flatten, "review": review, "movie_names": corrected_name})

# Create a new DataFrame from the list of extracted data
extracted_df = pd.DataFrame(extracted_data)

print(extracted_df.head())
print(f"The percetange of extraction is : {len(extracted_df) * 100 / len(imdb_df)}")
print(f"Training data contains {len(extracted_df)} rows")
print(extracted_df[["review_token", "movie_names_token"]])

df = extracted_df[["review_token", "movie_names_token"]]

# Padding function
def pad_tokens(tokens_list, max_length):
    return pad_sequences([tokens_list], maxlen=max_length, padding='post', truncating='post')[0]

# Find the maximum lengths
max_review_length = max(df['review_token'].apply(len))
max_movie_length = max(df['movie_names_token'].apply(len))

# Apply padding to the DataFrame
df['review_token'] = df['review_token'].apply(lambda x: pad_tokens(x, max_review_length))
df['movie_names_token'] = df['movie_names_token'].apply(lambda x: pad_tokens(x, max_movie_length))
print(df.head())
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    np.array(df['review_token'].tolist()),
    np.array(df['movie_names_token'].tolist()),
    test_size=0.2,
    random_state=42
)
vocabulary_size = len(set(df['review_token'].sum())) + 1

# Split the data into training and testing sets
X = np.array(df['review_token'].tolist())
y = np.array(df['movie_names_token'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split

# Generate some random data
# Replace this with your actual data
num_samples = 1000
time_steps = 10
input_size = 4
output_size = 2

X = np.random.rand(num_samples, time_steps, input_size)
Y = np.random.rand(num_samples, output_size)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(units=50, activation='tanh', input_shape=(time_steps, input_size)))
model.add(Dense(units=output_size, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary for debugging
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Data: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Print some example predictions
for i in range(5):
    print(f"Example {i + 1}:")
    print("Actual:", y_test[i])
    print("Predicted:", predictions[i])
    print()



# ======== Working version, do not touch ===========

# # Assuming 'train_data' is your training dataset with reviews and extracted movie names
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# extracted_df['tokenized_reviews'] = extracted_df['review'].apply(lambda x: tokenizer(x, padding=True, truncation=True, return_tensors='tf', max_length=512))
# extracted_df["input_ids"] = extracted_df['tokenized_reviews'].apply(lambda x: np.array(x['input_ids']))
# extracted_df['input_ids'] = extracted_df['input_ids'].apply(lambda x: np.array(x[0]))

# # Pad or truncate the 'input_ids' to a fixed length (e.g., max_length=512)
# max_length = 512
# extracted_df['padded_input_ids'] = extracted_df['input_ids'].apply(lambda x: pad_sequences([x], maxlen=max_length, dtype="long", value=0, truncating="post")[0])

# # Encode movie names using LabelEncoder
# label_encoder = LabelEncoder()
# extracted_df['encoded_labels'] = label_encoder.fit_transform(extracted_df['movie_names'])

# # Shuffle the DataFrame
# extracted_df = shuffle(extracted_df, random_state=42)

# # Split the DataFrame into training and testing sets
# train_df, test_df = train_test_split(extracted_df, test_size=0.2, random_state=42)


# # Define the neural network model
# model = Sequential([
#     tf.keras.layers.Embedding(input_dim=30522, output_dim=32, input_length=max_length),  # 30522 is the vocabulary size for BERT
#     Flatten(),
#     Dense(units=len(set(extracted_df['encoded_labels'])), activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(np.vstack(train_df['padded_input_ids']), train_df['encoded_labels'], epochs=5, batch_size=16)

# # Evaluate the model on the test set
# test_loss, test_acc = model.evaluate(np.vstack(test_df['padded_input_ids']), test_df['encoded_labels'])
# print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

# # Save the model for later use
# model.save('movie_name_prediction_model')

# ==== testing =====

# loaded_model = load_model('movie_name_prediction_model')
# tokenized_reviews = imdb_df['review'].apply(lambda x: tokenizer(x, padding=True, truncation=True, return_tensors='tf', max_length=512))
# input_ids = tokenized_reviews.apply(lambda x: np.array(x['input_ids']))
# input_ids = input_ids.apply(lambda x: np.array(x[0]))
# padded_input_ids = input_ids.apply(lambda x: pad_sequences([x], maxlen=max_length, dtype="long", value=0, truncating="post")[0])

# predictions = loaded_model.predict(np.vstack(padded_input_ids))

# decoded_predictions = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# imdb_df['predicted_movie_names'] = decoded_predictions
# print(imdb_df[['review', 'predicted_movie_names']])