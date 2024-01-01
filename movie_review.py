import pandas as pd
import re
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

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

# Display the new DataFrame
print(extracted_df.head())

# Assuming 'train_data' is your training dataset with reviews and extracted movie names
train_data = extracted_df.sample(frac=0.8, random_state=42)  # Use 80% for training

# Encode the movie names with LabelEncoder
label_encoder = LabelEncoder()
train_data['encoded_labels'] = label_encoder.fit_transform(train_data['movie_names'])

# Tokenize the reviews using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_reviews = tokenizer(list(train_data['review']), padding=True, truncation=True, return_tensors='tf', max_length=512)

train_indices, val_indices = train_test_split(
    range(len(tokenized_reviews['input_ids'])), test_size=0.2, random_state=42
)

train_reviews = tokenized_reviews['input_ids'][train_indices]
val_reviews = tokenized_reviews['input_ids'][val_indices]

train_labels = train_data['encoded_labels'].iloc[train_indices]
val_labels = train_data['encoded_labels'].iloc[val_indices]


# Load the BERT model for sequence classification
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Compile the model
bert_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
bert_model.fit(train_reviews, train_labels, epochs=5, validation_data=(val_reviews, val_labels), batch_size=16)

# Save the trained model
bert_model.save('movie_name_extraction_model')
