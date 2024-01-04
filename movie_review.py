import pandas as pd
import re
import string
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW

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
        if len(corrected_name.split()) == 2:
            # Get BERT token for movie name
            movie_name_tokens_nested = [get_bert_token(token) for token in corrected_name.split()]
            movie_name_tokens_flatten = list(chain(*movie_name_tokens_nested))

            # Get BERT token for review
            review_tokens = get_bert_token(review)
            extracted_data.append({'review_token': review_tokens, 'movie_names_token': movie_name_tokens_flatten, "review": review, "movie_names": corrected_name})

# Create a new DataFrame from the list of extracted data
extracted_df = pd.DataFrame(extracted_data)

# Determine the maximum lengths
max_review_length = max(extracted_df['review_token'].apply(len))
max_movie_length = max(extracted_df['movie_names_token'].apply(len))

# Apply padding to the DataFrame
extracted_df['review_token'] = extracted_df['review_token'].apply(lambda x: pad_sequences([x], maxlen=max_review_length, padding='post', truncating='post')[0])
extracted_df['movie_names_token'] = extracted_df['movie_names_token'].apply(lambda x: pad_sequences([x], maxlen=max_movie_length, padding='post', truncating='post')[0])
df = extracted_df



import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Input text
input_text = df['review'][0]

# Output text (target for training, for simplicity using the same text)
output_text = df['movie_names'][0]

# Tokenize the input and output texts
# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and preprocess the text
def preprocess_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return tokens

df['tokenized_review'] = df['review'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tokenized_review'], df['movie_name'], test_size=0.2, random_state=42)

# Pad sequences for equal length (assuming PyTorch tensors)
X_train = torch.nn.utils.rnn.pad_sequence(X_train, batch_first=True)
X_test = torch.nn.utils.rnn.pad_sequence(X_test, batch_first=True)

# Convert labels to PyTorch tensors
y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)

# Define the DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data['movie_name'].unique()))

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Make predictions on the test set
with torch.no_grad():
    model.eval()
    outputs = model(X_test)[0]
    _, predicted = torch.max(outputs, 1)
    predicted_movie = predicted.numpy()

# Evaluate the model
accuracy = accuracy_score(y_test, predicted_movie)
print(f'Model Accuracy: {accuracy}')

# Example usage
def predict_movie_name(review_text):
    processed_text = preprocess_text(review_text)
    with torch.no_grad():
        model.eval()
        outputs = model(processed_text)[0]
        _, predicted = torch.max(outputs, 1)
        predicted_movie = predicted.item()
    return predicted_movie

# Test the model with a sample review
sample_review = "I absolutely loved this movie! The acting was superb, and the storyline kept me engaged throughout."
predicted_movie = predict_movie_name(sample_review)
print(f'Predicted Movie: {predicted_movie}')


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