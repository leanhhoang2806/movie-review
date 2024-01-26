import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AdamW
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras import Input, Model

# Load the IMDb dataset (you can replace this with your own dataset)
from datasets import load_dataset

# Load the Hugging Face tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Create a strategy to distribute the training across multiple GPUs
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Load and preprocess the dataset
dataset = load_dataset("imdb")
encoded_data = tokenizer(
    dataset["train"]["text"],
    padding=True,
    truncation=True,
    return_tensors="tf",
)

# Use 80% of the data for training and 20% for validation
train_size = int(0.8 * len(dataset["train"]))
val_size = len(dataset["train"]) - train_size

train_dataset = tf.data.Dataset.from_tensor_slices(
    ({k: v[:train_size] for k, v in encoded_data.items()}, dataset["train"]["label"][:train_size])
)
val_dataset = tf.data.Dataset.from_tensor_slices(
    ({k: v[train_size:] for k, v in encoded_data.items()}, dataset["train"]["label"][train_size:])
)

batch_size = 8
learning_rate = 2e-5
epochs = 3

# Batch and shuffle the datasets
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# Use the strategy to create and compile the model
with strategy.scope():
    # Create the model inside the strategy scope
    inputs = {k: Input(shape=v.shape[1:], dtype=v.dtype, name=k) for k, v in encoded_data.items()}
    outputs = model(inputs)
    outputs = tf.keras.layers.Softmax()(outputs.logits)

    distributed_model = Model(inputs, outputs)

    # Compile the model
    distributed_model.compile(
        optimizer=AdamW(learning_rate=learning_rate),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()],
    )

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Train the model
    distributed_model.fit(train_dataset, epochs=1)

    # Evaluate on the validation set
    val_loss, val_accuracy = distributed_model.evaluate(val_dataset)

    # Print metrics
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
