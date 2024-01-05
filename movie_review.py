import pandas as pd
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, Flatten
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_imdb_dataset(csv_file_path):
    return pd.read_csv(csv_file_path)

def extract_data_from_reviews(imdb_df):
    extracted_data = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_bert_token(word):
        token_ids = tokenizer.encode(word, add_special_tokens=False, padding=True, truncation=True, max_length=512)
        return token_ids

    for review in tqdm(imdb_df['review'], desc="Extracting Data"):
        matches = re.findall(r'"([^"]+)"', review)
        valid_names = [name for name in matches if name.istitle()]
        cleaned_names = [name.strip(string.punctuation) for name in valid_names]

        if cleaned_names and len(cleaned_names) == 1 and len(cleaned_names[0].split()) == 2:
            corrected_name = cleaned_names[0].strip()

            movie_name_tokens_nested = [get_bert_token(token) for token in corrected_name.split()]
            movie_name_tokens_flatten = list(itertools.chain(*movie_name_tokens_nested))

            review_tokens = get_bert_token(review)

            extracted_data.append({'review_token': review_tokens, 'movie_names_token': movie_name_tokens_flatten,
                                   "review": review, "movie_names": corrected_name})

    return pd.DataFrame(extracted_data)

def preprocess_data(extracted_df):
    max_review_length = max(extracted_df['review_token'].apply(len))
    max_movie_length = max(extracted_df['movie_names_token'].apply(len))

    extracted_df['review_token'] = extracted_df['review_token'].apply(
        lambda x: pad_sequences([x], maxlen=max_review_length, padding='post', truncating='post')[0])
    extracted_df['movie_names_token'] = extracted_df['movie_names_token'].apply(
        lambda x: pad_sequences([x], maxlen=max_movie_length, padding='post', truncating='post')[0])

    return extracted_df[['review_token', 'movie_names_token']]


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
        scaled_attention = self._scaled_dot_product_attention(query, key, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concatenate heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear layer
        outputs = self.dense(concat_attention)

        return outputs

    def _scaled_dot_product_attention(query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output

def build_attention_model(input_shape, output_size, num_layers, layer_size, dropout_rate, num_heads):
    inputs = Input(shape=(input_shape,))
    x = Dense(layer_size // 2, activation='relu')(inputs)

    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_output = bert_model(inputs)[0]

    x = Dense(layer_size // 2, activation='relu')(bert_output[:, 0, :])
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

def perform_grid_search(param_grid, input_shape, output_size, X_train, y_train, X_test, y_test):
    best_accuracy = 0
    best_params = None

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    for params in tqdm(itertools.product(*param_grid.values()), total=len(list(itertools.product(*param_grid.values()))), desc="Grid Search Progress"):
        with strategy.scope():
            model = build_attention_model(input_shape, output_size, *params)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=-1)

        _, accuracy = model.evaluate(X_test, y_test, verbose=-1)

        tqdm.write(f'Model Accuracy for {params}: {accuracy}')

        if accuracy > best_accuracy:
            best_accuracy = max(accuracy, best_accuracy)
            best_params = params

    return best_accuracy, best_params

if __name__ == "__main__":
    csv_file_path = './IMDB Dataset.csv'
    imdb_df = load_imdb_dataset(csv_file_path)

    extracted_df = extract_data_from_reviews(imdb_df)
    preprocessed_df = preprocess_data(extracted_df)

    X = np.array(preprocessed_df['review_token'].tolist())
    Y = np.array(preprocessed_df['movie_names_token'].tolist())
    output_size = Y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    param_grid = {
        'num_layers': [i for i in range(1, 5)],
        'layer_size': [256 * i for i in range(5, 10)],
        'dropout_rate': [0.2 * i for i in range(1, 5)],
        'num_heads': [i * 2 for i in range(1, 3)],
    }

    best_accuracy, best_params = perform_grid_search(param_grid, X.shape[1], output_size, X_train, y_train, X_test, y_test)
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
