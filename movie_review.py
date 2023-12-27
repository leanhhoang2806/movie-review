import pandas as pd
import re
import string
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertConfig

# Check if a GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
device = "GPU" if len(physical_devices) > 0 else "CPU"
print(f"{device} is available.")

# csv_file_path = './IMDB Dataset.csv'
# imdb_df = pd.read_csv(csv_file_path)

# # Define a regular expression pattern for extracting text between quotation marks
# movie_name_pattern = re.compile(r'"([^"]+)"')

# # Initialize an empty list to store extracted data
# extracted_data = []

# # Loop through each review and extract text between quotation marks
# for review in imdb_df['review']:
#     # Extract text between quotation marks using the defined pattern
#     matches = re.findall(movie_name_pattern, review)
    
#     # Filter out names that are not in capitalization format
#     valid_names = [name for name in matches if name.istitle()]
    
#     # Clean up punctuation at the beginning or end of each title
#     cleaned_names = [name.strip(string.punctuation) for name in valid_names]
    
#     # Append the review and cleaned text to the list if the list is not empty
#     if cleaned_names and len(cleaned_names) == 1:
#         corrected_name = cleaned_names[0].strip()
#         if len(corrected_name) > 2: 
#             extracted_data.append({'review': review, 'movie_names': corrected_name})

# # Create a new DataFrame from the list of extracted data
# extracted_df = pd.DataFrame(extracted_data)

# # Percentage of extraction successfully
# percentage = (extracted_df.shape[0] / imdb_df.shape[0]) * 100
# print(f"The amount of extracted data is {percentage:.2f} %")

# # Display the new DataFrame
# print(extracted_df.head())

# # Assuming 'train_data' is your training dataset with reviews and extracted movie names
# # 'imdb_reviews' is your IMDb dataset with reviews
# train_data = extracted_df.sample(frac=0.8, random_state=42)  # Use 80% for training

# # Assuming 'train_data' is your DataFrame
# label_encoder = LabelEncoder()
# train_data['encoded_labels'] = label_encoder.fit_transform(train_data['movie_names'])

# # Tokenize the training data using BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenized_reviews = tokenizer(list(train_data['review']), padding=True, truncation=True, return_tensors='tf', max_length=512)

# # Create a TensorFlow Dataset
# dataset = tf.data.Dataset.from_tensor_slices(({
#     'input_ids': tokenized_reviews['input_ids'],
#     'attention_mask': tokenized_reviews['attention_mask']
# }, train_data['encoded_labels']))

# # Split the dataset into training and validation sets
# train_size = int(0.8 * len(train_data))
# val_size = len(train_data) - train_size
# train_dataset = dataset.take(train_size).shuffle(buffer_size=50000).batch(32)
# val_dataset = dataset.skip(train_size).batch(32)

# # Build the BERT-based model
# config = BertConfig.from_pretrained('bert-base-uncased', num_labels=len(set(train_data['movie_names'])))
# bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
# model = Sequential([
#     bert_model,
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(len(set(train_data['movie_names'])), activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Training loop
# num_epochs = 5
# model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)

# # Save the trained model
# model.save('path/to/saved_model')

# # Load the trained model for inference
# loaded_model = tf.keras.models.load_model('path/to/saved_model')

# # Evaluation on the test set
# val_predictions = loaded_model.predict(val_dataset)
# val_preds = tf.argmax(val_predictions, axis=1).numpy()

# # Compute confusion matrix
# conf_matrix = confusion_matrix(train_data['encoded_labels'][-val_size:], val_preds)

# # Print confusion matrix
# print("Confusion Matrix:")
# print(conf_matrix)

# # Optionally, you can also print a classification report
# class_report = classification_report(train_data)
