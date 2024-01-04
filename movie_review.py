import pandas as pd
import re
import string
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Flatten
from tensorflow.keras.models import Model
import itertools
from tqdm import tqdm

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
df = extracted_df[['review_token', 'movie_names_token']]

print(df.head())


import tensorflow as tf
X = np.array(df['review_token'].tolist())
Y = np.array(df['movie_names_token'].tolist())
output_size = Y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = Dense(d_model)
        self.key_dense = Dense(d_model)
        self.value_dense = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]

        # Linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concatenate heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear layer
        outputs = self.dense(concat_attention)

        return outputs

# Scaled Dot-Product Attention Layer
def scaled_dot_product_attention(query, key, value):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output

# Modify the model with Multi-Head Attention
def build_model(input_shape, output_size, num_layers, layer_size, dropout_rate, num_heads):
    inputs = Input(shape=(input_shape,))
    x = Dense(layer_size, activation='relu')(inputs)
    
    # Add Multi-Head Attention
    attention = MultiHeadAttention(d_model=layer_size, num_heads=num_heads)({
        'query': tf.expand_dims(x, 1),
        'key': tf.expand_dims(x, 1),
        'value': tf.expand_dims(x, 1)
    })
    x = LayerNormalization(epsilon=1e-6)(x + Flatten()(attention))
    
    for _ in range(num_layers - 1):
        x = Dense(layer_size, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    
    x = Dense(output_size, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

# Define a grid of hyperparameters to search over (including Multi-Head Attention parameters)
param_grid = {
    'num_layers': [1, 2, 3],
    'layer_size': [64, 128, 256],  # Increase the layer size
    'dropout_rate': [0.2, 0.5],
    'num_heads': [2, 4, 8],
}

# Set batch size and accumulation steps
batch_size = 32
accumulation_steps = 4  # Accumulate gradients over 4 mini-batches

# Perform a grid search with tqdm progress bar
best_accuracy = 0
best_model = None

# Wrap tqdm around itertools.product to show progress
for params in tqdm(itertools.product(*param_grid.values()), total=len(list(itertools.product(*param_grid.values()))), desc="Grid Search Progress"):
    model = build_model(X.shape[1], output_size, *params)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Split the data into batches
    for i in range(0, len(X_train), batch_size * accumulation_steps):
        X_batch = X_train[i:i + batch_size * accumulation_steps]
        y_batch = y_train[i:i + batch_size * accumulation_steps]

        # Train the model on the current batch
        model.train_on_batch(X_batch, y_batch)

    # Save the model parameters to disk
    model.save_weights(f'model_weights_{params}.h5')

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    tqdm.write(f'Model Accuracy for {params}: {accuracy}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Print the best model's summary
if best_model:
    print("\nBest Model Summary:")
    best_model.summary()


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