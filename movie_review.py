import pandas as pd
import re
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification



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


# Assuming 'train_data' is your training dataset with reviews and extracted movie names
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

extracted_df['tokenized_reviews'] = extracted_df['review'].apply(lambda x: tokenizer(x, padding=True, truncation=True, return_tensors='tf', max_length=512))
extracted_df["input_ids"] = extracted_df['tokenized_reviews'].apply(lambda x: np.array(x['input_ids']))
extracted_df['input_ids'] = extracted_df['input_ids'].apply(lambda x: np.array(x[0]))

# Pad or truncate the 'input_ids' to a fixed length (e.g., max_length=512)
max_length = 512
extracted_df['padded_input_ids'] = extracted_df['input_ids'].apply(lambda x: pad_sequences([x], maxlen=max_length, dtype="long", value=0, truncating="post")[0])

# Encode movie names using LabelEncoder
label_encoder = LabelEncoder()
extracted_df['encoded_labels'] = label_encoder.fit_transform(extracted_df['movie_names'])

# Shuffle the DataFrame
extracted_df = shuffle(extracted_df, random_state=42)

train_data, test_data = train_test_split(extracted_df, test_size=0.2, random_state=42)

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Tokenize the input text
train_encodings = tokenizer(train_data['review'].tolist(), padding=True, truncation=True, return_tensors='tf')
test_encodings = tokenizer(test_data['review'].tolist(), padding=True, truncation=True, return_tensors='tf')

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_data['movie_names']
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_data['movie_names']
))

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset.shuffle(1000).batch(16), epochs=5, validation_data=test_dataset.shuffle(1000).batch(16))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset.batch(16))
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')