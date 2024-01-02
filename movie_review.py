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


# Load the IMDb dataset
csv_file_path = './IMDB Dataset.csv'
imdb_df = pd.read_csv(csv_file_path)
os.environ["TRANSFORMERS_CACHE"] = "./"

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
df = extracted_df
cache_dir = "./"

# Load pre-trained BERT NER model and tokenizer with caching
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english', cache_dir=cache_dir)
model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english', num_labels=9, cache_dir=cache_dir)

# Your existing code...
# (Tokenization, Encoding, Training loop, Inference)

# Save or use the fine-tuned model for named entity recognition
model.save_pretrained('./fine_tuned_bert_ner_model')

# Tokenize and encode the DataFrame
def encode_data(df, tokenizer, max_length=512):
    tokenized_reviews = []
    token_labels = []

    for _, row in df.iterrows():
        review = row['review']
        movie_names = [row['movie_names']]

        # Tokenize the text
        encoding = tokenizer(review, max_length=max_length, truncation=True, return_tensors='pt', padding='max_length')

        # Convert extracted entities to token-level labels
        labels = torch.zeros(max_length, dtype=torch.long)  # Initialize labels with 'O' (Outside entity) index
        for entity in movie_names:
            entity_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(entity)))
            for i in range(len(encoding['input_ids'][0])):
                if encoding['input_ids'][0][i].item() == tokenizer.encode(entity_tokens[0])[0]:
                    labels[i] = 1  # 'B-PER' label
                    for j in range(1, len(entity_tokens)):
                        labels[i + j] = 2  # 'I-PER' label

        tokenized_reviews.append(encoding)
        token_labels.append(labels)

    return tokenized_reviews, token_labels

tokenized_reviews, token_labels = encode_data(df, tokenizer)

# Prepare training data
train_tokens, val_tokens, train_labels, val_labels = train_test_split(tokenized_reviews, token_labels, test_size=0.2, random_state=42)

# Convert labels to tensors
train_labels = torch.stack(train_labels)
val_labels = torch.stack(val_labels)

# Truncate the tensor if necessary to match the model's maximum length
max_length = min(val_labels.shape[1], 512)
val_labels = val_labels[:, :max_length]

# Set up optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)

# Fine-tune BERT for named entity recognition
num_epochs = 3
for epoch in range(num_epochs):
    for tokens, labels in zip(train_tokens, train_labels):
        # Ensure proper structure for input tokens
        tokens = {key: tokens[key].squeeze(0) for key in tokens}

        # Add batch dimension to labels tensor
        labels = labels.unsqueeze(0)

        # Flatten the labels tensor to match the model's expectations
        labels = labels.view(-1)
        
        # Make sure tokens are on the same device as the model
        tokens = {key: value.to(model.device) for key, value in tokens.items()}

        outputs = model(**tokens, labels=labels.unsqueeze(0))
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save or use the fine-tuned model for named entity recognition
model.save_pretrained('./fine_tuned_bert_ner_model')

# Example of using the fine-tuned model for inference
def extract_movie_names(review, model, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(review, max_length=max_length, truncation=True)))
    labels = model(**tokenizer(review, padding=True, return_tensors='pt', truncation=True), labels=None).logits.argmax(dim=2)
    movie_names = [tokens[i] for i in range(min(len(tokens), max_length)) if labels[0][i].item() in [5, 6, 7, 8]]  # Extract tokens labeled as 'B-PER' or 'I-PER'
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(movie_names))

# Example usage
sample_review = "I watched a great movie starring Tom Hanks yesterday."
predicted_movie_names = extract_movie_names(sample_review, model, tokenizer)
print(f"Predicted Movie Names: {predicted_movie_names}")

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