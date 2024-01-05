import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Flatten
import itertools
from tqdm import tqdm
import tensorflow as tf
from transformers import TFBertModel
from data_loader.load_imdb import DataLoader
from processors.tokenizer import TokenizedText
from processors.normalizer import Normalizer
from models.multiheads import MultiHeadAttention
from processors.train_test_split import data_split

# Load the IMDb dataset
csv_file_path = './IMDB Dataset.csv'
data_loader = DataLoader(csv_file_path)
imdb_df = data_loader.read_to_pandas()

# Create a new DataFrame from the list of extracted data
extracted_df = TokenizedText().tokenized(imdb_df)

df = Normalizer().normalize(extracted_df)



X = np.array(df['review_token'].tolist())
Y = np.array(df['movie_names_token'].tolist())
output_size = Y.shape[1]

X_train, X_test, y_train, y_test = data_split(df)


# ======== Multi-computer search ===========
def build_complex_model(input_shape, output_size, num_layers, layer_size, dropout_rate, num_heads):
    inputs = Input(shape=(input_shape,))
    x = Dense(layer_size // 2, activation='relu')(inputs)  # Reduce layer size
        
    # Use BERT model as embedding layer
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
