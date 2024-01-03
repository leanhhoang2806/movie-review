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

from torch.utils.data import TensorDataset, DataLoader

# Define the model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=2)
# Sample data (you need to replace this with your actual dataset)
texts = ["The movie Titanic is a classic.", "Inception is a mind-bending film."]

# Prepare the data
tokenized_texts = [tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text))) for text in texts]
input_ids = [tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
attention_masks = [[1] * len(input_id) for input_id in input_ids]
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = []
for i, tokenized_text in enumerate(tokenized_texts):
    # Convert labels to 1 for movie name tokens, 0 otherwise
    label = [1 if tokenized_text[j] in tokenizer.convert_ids_to_tokens(input_ids[i, j].item()) else 0 for j in range(len(tokenized_text))]
    labels.append(label)
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)

# Define data loader
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-5)

for epoch in range(3):  # Adjust the number of epochs as needed
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

# Inference
model.eval()
with torch.no_grad():
    example_text = "I enjoyed watching Titanic last night."
    tokenized_text = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(example_text)))
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
    attention_mask = torch.tensor([[1] * len(input_ids[0])])

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Extract movie names based on predictions
    movie_names = [token for token, label in zip(tokenized_text, predictions[0].tolist()) if label == 1]
    print("Extracted Movie Names:", movie_names)

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