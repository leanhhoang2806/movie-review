import pandas as pd
import re
import string
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import TFBertModel

# Define the number of workers
num_workers = 2
gpus = tf.config.experimental.list_physical_devices('GPU')

# Initialize the distributed strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with strategy.scope():
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

    # Assuming 'train_data' is your DataFrame
    label_encoder = LabelEncoder()
    train_data['encoded_labels'] = label_encoder.fit_transform(train_data['movie_names'])
    print("Load BERT tokenizer and model")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', from_pt=True)

    # Tokenize reviews for training set
    train_tokenized_reviews = tokenizer(list(train_data['review']), padding=True, truncation=True,
                                         return_tensors='tf', max_length=512)

    # Filter out samples with inconsistent input sizes
    indices_to_keep = [i for i, input_ids in enumerate(train_tokenized_reviews['input_ids']) if len(input_ids) == 512]
    train_tokenized_reviews = {key: value[indices_to_keep] for key, value in train_tokenized_reviews.items()}

    # Convert movie names to numerical labels
    labels = label_encoder.fit_transform(train_data['movie_names'])

    # Split the dataset into training and test sets
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        train_tokenized_reviews, labels, test_size=0.2, random_state=42
    )

    # Further split the training set into training and validation sets
    train_reviews, val_reviews, train_labels, val_labels = train_test_split(
        train_reviews, train_labels, test_size=0.1, random_state=42
    )
    print("Build BERT-based model")
    model = Sequential([
        bert_model,
        Flatten(),
        Dense(units=len(set(labels)), activation='softmax')  # Assuming you have n classes for movie names
    ])

    print("Compile the model")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Train the model")
    model.fit(train_reviews, train_labels, epochs=5, validation_data=(val_reviews, val_labels), batch_size=16)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_reviews, test_labels)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

    # Save the model for later use
    model.save('movie_name_extraction_model')
