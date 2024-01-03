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


# Load the IMDb dataset
csv_file_path = './IMDB Dataset.csv'
imdb_df = pd.read_csv(csv_file_path)

# Define a regular expression pattern for extracting text between quotation marks
movie_name_pattern = re.compile(r'"([^"]+)"')

# Initialize an empty list to store extracted data
extracted_data = []

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
        if len(corrected_name) > 2:
            extracted_data.append({'review': review, 'movie_names': corrected_name})

# Create a new DataFrame from the list of extracted data
extracted_df = pd.DataFrame(extracted_data)
df =extracted_df
print(extracted_df.head())

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import pipeline
from torch.nn import Softmax


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

reviews = extracted_df['review']
movie_name = extracted_df["movie_names"]

model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
text = reviews[0]
result = movie_name[0]
# Tokenize the text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text, max_length=512, truncation=True)))

# Load pre-trained BERT model for token classification
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=4)  # 4 labels: B-LOC, I-LOC, B-MISC, I-MISC

# Convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Split the input into chunks of maximum 512 tokens
max_length = 512
token_ids_chunks = [token_ids[i:i + max_length] for i in range(0, len(token_ids), max_length)]

# Obtain predictions from the model for each chunk
predictions_list = []
with torch.no_grad():
    for chunk in token_ids_chunks:
        outputs = model(torch.tensor([chunk]))
        predictions_list.append(outputs.logits)

# Concatenate predictions from different chunks
predictions = torch.cat(predictions_list, dim=1)

# Apply softmax to get probabilities
softmax = Softmax(dim=2)
predictions = softmax(predictions)

# Get the predicted labels for each token
predicted_labels = torch.argmax(predictions, dim=2).numpy()[0]

# Map predicted labels to entities
id2label = {0: 'B-LOC', 1: 'I-LOC', 2: 'B-MISC', 3: 'I-MISC'}
predicted_entities = [id2label[label] for label in predicted_labels]

# Extract movie name combinations based on B-MISC and I-MISC entities
movie_name_combinations = []
current_entity = None
current_movie_name = []

for token, entity in zip(tokens, predicted_entities):
    if entity in ['B-MISC', 'I-MISC']:
        if current_entity is None:
            current_entity = entity
            current_movie_name = [token]
        else:
            current_movie_name.append(token)
    else:
        if current_entity is not None:
            movie_name_combinations.append(' '.join(current_movie_name))
            current_entity = None
            current_movie_name = []

# Print the recognized movie name combinations
print("Recognized Movie Name Combinations:", movie_name_combinations)
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