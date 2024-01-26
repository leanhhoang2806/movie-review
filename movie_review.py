import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the IMDb dataset (you can replace this with your own dataset)
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("imdb")

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded_data = tokenizer(
    dataset["train"]["text"],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

# Prepare the DataLoader
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Use 80% of the data for training and 20% for validation
train_size = int(0.8 * len(dataset["train"]))
val_size = len(dataset["train"]) - train_size

train_dataset = MyDataset(
    {k: v[:train_size] for k, v in encoded_data.items()},
    dataset["train"]["label"][:train_size],
)

val_dataset = MyDataset(
    {k: v[train_size:] for k, v in encoded_data.items()},
    dataset["train"]["label"][train_size:],
)

# Load the pre-trained BERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training parameters
batch_size = 8
learning_rate = 2e-5
epochs = 3

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Set the model to training mode
    model.train()

    # Create a progress bar for the training data
    train_loader_with_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}, Training")

    for batch in train_loader_with_progress:
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Update the progress bar
        train_loader_with_progress.set_postfix(loss=loss.item())

    # Validation loop
    model.eval()

    # Create a progress bar for the validation data
    val_loader_with_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs}, Validation")

    for batch in val_loader_with_progress:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Update the progress bar
        val_loader_with_progress.set_postfix(loss=loss.item())

    # Close the progress bars after each epoch
    train_loader_with_progress.close()
    val_loader_with_progress.close()

    # Print metrics
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
