
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, Flatten
from tensorflow.keras.models import Model
from models.multiheads import MultiHeadAttention

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