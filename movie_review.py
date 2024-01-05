import pandas as pd
import re
import string
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Flatten
from tensorflow.keras.models import Model
import itertools
from tqdm import tqdm
import tensorflow as tf
from transformers import TFBertModel

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

# ======== Multi-computer search ===========
def build_complex_model(input_shape, output_size, num_layers, layer_size, dropout_rate, num_heads):
    inputs = Input(shape=(input_shape,))
    x = Dense(layer_size // 2, activation='relu')(inputs)  # Reduce layer size
        
    # Use BERT model as embedding layer
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_output = bert_model(inputs)[0]
    # Apply pooling to BERT output
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(bert_output)

    x = Dense(layer_size // 2, activation='relu')(pooled_output)
    attention = MultiHeadAttention(d_model=layer_size // 2, num_heads=num_heads)({
        'query': tf.expand_dims(x, 1),
        'key': tf.expand_dims(x, 1),
        'value': tf.expand_dims(x, 1)
    })
    x = LayerNormalization(epsilon=1e-6)(x + Flatten()(attention))
    
    for _ in range(num_layers - 1):
        x = Dense(layer_size // 2, activation='relu')(x)  # Reduce layer size
        x = Dropout(dropout_rate)(x)
    
    x = Dense(layer_size // 2, activation='relu')(x)  # Reduce layer size
    x = Dropout(dropout_rate)(x)
    
    x = Dense(output_size, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


# Use OneDeviceStrategy for model parallelism in a single GPU environment
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Define a grid of hyperparameters to search over (including Multi-Head Attention parameters)
param_grid = {
    'num_layers': [i for i in range(1, 5)],
    'layer_size': [256*i for i in range(5, 10)],  # Increase the layer size
    'dropout_rate': [0.2 * 1 for i in range(1, 5)],
    'num_heads': [i*2 for i in range(1,3)],
}

# Perform a grid search with tqdm progress bar
best_accuracy = 0
best_model = None

# Wrap tqdm around itertools.product to show progress
for params in tqdm(itertools.product(*param_grid.values()), total=len(list(itertools.product(*param_grid.values()))), desc="Grid Search Progress"):
    with strategy.scope():
        model = build_complex_model(X.shape[1], output_size, *params)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=-1)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=-1)

    tqdm.write(f'Model Accuracy for {params}: {accuracy}')

    if accuracy > best_accuracy:
        best_accuracy = max(accuracy, best_accuracy)
        best_params = params

print(f'Best Model Accuracy: {best_accuracy} with best params: {best_params}')
# ======== Single computer search ===========
# # Modify the model with Multi-Head Attention
# def build_model(input_shape, output_size, num_layers, layer_size, dropout_rate, num_heads):
#     inputs = Input(shape=(input_shape,))
#     x = Dense(layer_size, activation='relu')(inputs)
    
#     # Add Multi-Head Attention
#     attention = MultiHeadAttention(d_model=layer_size, num_heads=num_heads)({
#         'query': tf.expand_dims(x, 1),
#         'key': tf.expand_dims(x, 1),
#         'value': tf.expand_dims(x, 1)
#     })
#     x = LayerNormalization(epsilon=1e-6)(x + Flatten()(attention))
    
#     for _ in range(num_layers - 1):
#         x = Dense(layer_size, activation='relu')(x)
#         x = Dropout(dropout_rate)(x)
    
#     x = Dense(output_size, activation='softmax')(x)

#     model = Model(inputs=inputs, outputs=x)
#     return model

# # Define a grid of hyperparameters to search over (including Multi-Head Attention parameters)
# param_grid = {
#     'num_layers': [1, 2, 3],
#     'layer_size': [256, 512, 1024, 2048, 4096],  # Increase the layer size
#     'dropout_rate': [0.2, 0.5],
#     'num_heads': [2, 4, 8],
# }

# # Set batch size and accumulation steps
# batch_size = 32
# accumulation_steps = 4  # Accumulate gradients over 4 mini-batches

# # Perform a grid search with tqdm progress bar
# best_accuracy = 0
# best_model = None

# # Wrap tqdm around itertools.product to show progress
# for params in tqdm(itertools.product(*param_grid.values()), total=len(list(itertools.product(*param_grid.values()))), desc="Grid Search Progress"):
#     model = build_model(X.shape[1], output_size, *params)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # Split the data into batches
#     for i in range(0, len(X_train), batch_size * accumulation_steps):
#         X_batch = X_train[i:i + batch_size * accumulation_steps]
#         y_batch = y_train[i:i + batch_size * accumulation_steps]

#         # Train the model on the current batch
#         model.train_on_batch(X_batch, y_batch)

#     # Save the model parameters to disk
#     model.save_weights(f'model_weights_{params}.h5')

#     # Evaluate the model
#     _, accuracy = model.evaluate(X_test, y_test, verbose=0)

#     tqdm.write(f'Model Accuracy for {params}: {accuracy}')

#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model = model

# # Print the best model's summary
# if best_model:
#     print("\nBest Model Summary:")
#     best_model.summary()
