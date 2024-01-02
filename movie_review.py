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
from tensorflow.keras.models import load_model
from transformers import TFBertModel


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

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize and pad the input sequences
max_length = 512
input_ids = []
attention_masks = []

for review in extracted_df['review']:
    inputs = tokenizer(review, padding=True, truncation=True, return_tensors='tf', max_length=max_length)
    input_ids.append(tf.constant(inputs['input_ids'][0]))
    attention_masks.append(tf.constant(inputs['attention_mask'][0]))

# Convert lists to numpy arrays
input_ids = np.array(input_ids)
attention_masks = np.array(attention_masks)

# Encode movie names using LabelEncoder
label_encoder = LabelEncoder()
extracted_df['encoded_labels'] = label_encoder.fit_transform(extracted_df['movie_names'])

# Shuffle the DataFrame
extracted_df = shuffle(extracted_df, random_state=42)

# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(extracted_df, test_size=0.2, random_state=42)

# Build the BERT-based model
input_ids_input = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
attention_mask_input = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

# Get BERT output
bert_output = bert_model([input_ids_input, attention_mask_input])[0]  # Updated to [0] for last layer output

# Pooling layer to reduce dimensionality
pooled_output = tf.keras.layers.GlobalAveragePooling1D()(bert_output)

# Dense layer for classification
output = tf.keras.layers.Dense(units=len(set(extracted_df['encoded_labels'])), activation='softmax')(pooled_output)

# Build and compile the model
model = tf.keras.Model(inputs=[input_ids_input, attention_mask_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([input_ids, attention_masks], train_df['encoded_labels'], epochs=5, batch_size=16)




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