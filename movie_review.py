import pandas as pd
import re
import string
import numpy as np
from itertools import product, chain
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, Flatten
from tensorflow.keras.models import Model
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

def scaled_dot_product_attention(query, key, value):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output

def load_imdb_dataset(csv_file_path='./IMDB Dataset.csv'):
    return pd.read_csv(csv_file_path)

def extract_movie_data(review):
    movie_name_pattern = re.compile(r'"([^"]+)"')
    matches = re.findall(movie_name_pattern, review)
    valid_names = [name for name in matches if name.istitle()]
    cleaned_names = [name.strip(string.punctuation) for name in valid_names]
    if cleaned_names and len(cleaned_names) == 1:
        corrected_name = cleaned_names[0].strip()
        if len(corrected_name.split()) == 2:
            return corrected_name
    return None

def get_bert_token(word, tokenizer):
    token_ids = tokenizer.encode(word, add_special_tokens=False, padding=True, truncation=True, max_length=512)
    return token_ids

def preprocess_review_data(review, tokenizer):
    movie_name = extract_movie_data(review)
    if movie_name:
        movie_name_tokens_nested = [get_bert_token(token, tokenizer) for token in movie_name.split()]
        movie_name_tokens_flatten = list(chain(*movie_name_tokens_nested))
        review_tokens = get_bert_token(review, tokenizer)
        return {'review_token': review_tokens, 'movie_names_token': movie_name_tokens_flatten,
                "review": review, "movie_names": movie_name}
    return None

def preprocess_df(extracted_data, max_review_length, max_movie_length):
    extracted_df = pd.DataFrame(extracted_data)
    extracted_df['review_token'] = extracted_df['review_token'].apply(
        lambda x: pad_sequences([x], maxlen=max_review_length, padding='post', truncating='post')[0])
    extracted_df['movie_names_token'] = extracted_df['movie_names_token'].apply(
        lambda x: pad_sequences([x], maxlen=max_movie_length, padding='post', truncating='post')[0])
    return extracted_df[['review_token', 'movie_names_token']]

def build_complex_model(input_shape, output_size, num_layers, layer_size, dropout_rate, num_heads):
    inputs = Input(shape=(input_shape,))
    x = Dense(layer_size // 2, activation='relu')(inputs)
    
    attention = MultiHeadAttention(d_model=layer_size // 2, num_heads=num_heads)({
        'query': tf.expand_dims(x, 1),
        'key': tf.expand_dims(x, 1),
        'value': tf.expand_dims(x, 1)
    })
    x = LayerNormalization(epsilon=1e-6)(x + Flatten()(attention))

    for _ in range(num_layers - 1):
        x = Dense(layer_size // 2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

    x = Dense(layer_size // 2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(output_size, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def grid_search(param_grid, X_train, y_train, X_test, y_test, input_shape, output_size):
    best_accuracy = 0
    best_params = None
    # Use MultiWorkerMirroredStrategy for model parallelism in a single GPU environment
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    for params in tqdm(product(*param_grid.values()), total=len(list(product(*param_grid.values()))), desc="Grid Search Progress"):
        with strategy.scope():
            model = build_complex_model(input_shape, output_size, *params)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

        _, accuracy = model.evaluate(X_test, y_test, verbose=0)

        tqdm.write(f'Model Accuracy for {params}: {accuracy}')

        if accuracy > best_accuracy:
            best_accuracy = max(accuracy, best_accuracy)
            best_params = params

    return best_accuracy, best_params

def main():
    # Reset TensorFlow session and graph
    tf.keras.backend.clear_session()

    # Load the IMDb dataset
    imdb_df = load_imdb_dataset()

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Extract and preprocess data
    extracted_data = [preprocess_review_data(review, tokenizer) for review in imdb_df['review']]

    # Determine the maximum lengths
    max_review_length = max([len(data['review_token']) for data in extracted_data])
    max_movie_length = max([len(data['movie_names_token']) for data in extracted_data])

    # Preprocess DataFrame
    extracted_df = preprocess_df(extracted_data, max_review_length, max_movie_length)

    # Split data
    X = np.array(extracted_df['review_token'].tolist())
    Y = np.array(extracted_df['movie_names_token'].tolist())
    output_size = Y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    # Define a grid of hyperparameters to search over (including Multi-Head Attention parameters)
    param_grid = {
        'num_layers': range(1),
        'layer_size': [256 * i for i in range(1)],  # Increase the layer size
        'dropout_rate': [0.2 * i for i in range(1)],
        'num_heads': [i * 2 for i in range(1)],
    }

    # Perform grid search
    best_accuracy, best_params = grid_search(param_grid, X_train, y_train, X_test, y_test, X.shape[1], output_size)

    print(f'Best Model Accuracy: {best_accuracy} with best params: {best_params}')

if __name__ == "__main__":
    main()




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
