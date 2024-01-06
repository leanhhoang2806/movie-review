import tensorflow as tf
import tqdm
import itertools
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from models.multiheads import MultiHeadAttention
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
    'layer_size': [256, 512, 1024, 2048, 4096],  # Increase the layer size
    'dropout_rate': [0.2, 0.5],
    'num_heads': [2, 4, 8],
}

# Set batch size and accumulation steps
batch_size = 32
accumulation_steps = 4  # Accumulate gradients over 4 mini-batches

# Perform a grid search with tqdm progress bar
best_accuracy = 0
best_model = None

X = np.array() # fill in data for the array 
Y = np.array() # fill in data for the array
output_size = Y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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