import tensorflow as tf
import itertools
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Flatten
from transformers import TFBertModel
from models.multiheads import MultiHeadAttention



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

def distributed_training(param_grid, input_shape, output_size, X_train, y_train, X_test, y_test):
    best_accuracy = 0
    best_params = None

    # Use OneDeviceStrategy for model parallelism in a single GPU environment
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # Wrap tqdm around itertools.product to show progress
    for params in tqdm(itertools.product(*param_grid.values()), total=len(list(itertools.product(*param_grid.values()))), desc="Grid Search Progress"):
        with strategy.scope():
            model = build_complex_model(input_shape, output_size, *params)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=-1)

        # Evaluate the model
        _, accuracy = model.evaluate(X_test, y_test, verbose=-1)

        tqdm.write(f'Model Accuracy for {params}: {accuracy}')

        if accuracy > best_accuracy:
            best_accuracy = max(accuracy, best_accuracy)
            best_params = params
    
    return best_accuracy, best_params