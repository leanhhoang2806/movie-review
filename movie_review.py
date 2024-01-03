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
# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define a PyTorch dataset
class MyDataset(Dataset):
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.targets = targets

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return {'review_token': torch.tensor(self.reviews.iloc[idx]),
                'movie_names_token': torch.tensor(self.targets.iloc[idx])}

# Instantiate the dataset and DataLoader
train_dataset = MyDataset(train_df['review_token'], train_df['movie_names_token'])
test_dataset = MyDataset(test_df['review_token'], test_df['movie_names_token'])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Define the BERT model for sequence-to-sequence prediction
class SequencePredictionModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(SequencePredictionModel, self).__init__()
        self.bert_model = bert_model
        self.linear = torch.nn.Linear(bert_model.config.hidden_size, len(df['movie_names_token'][0]))

    def forward(self, input_ids):
        outputs = self.bert_model(input_ids=input_ids)
        logits = self.linear(outputs.last_hidden_state)
        return logits

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Instantiate the model and set up the training loop
model = SequencePredictionModel(bert_model)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 5  # Adjust based on your data

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids, targets = batch['review_token'], batch['movie_names_token']
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

# Evaluate on the test set
model.eval()
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating'):
        input_ids, targets = batch['review_token'], batch['movie_names_token']
        logits = model(input_ids)
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds)

all_preds = torch.cat(all_preds).numpy()
accuracy = accuracy_score(test_df['movie_names_token'].apply(lambda x: x[0]), all_preds)
print(f'Test Accuracy: {accuracy}')




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