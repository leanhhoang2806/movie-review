import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the IMDb dataset (you can replace this with your own dataset)
from datasets import load_dataset

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
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            # Calculate accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Print metrics
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save the fine-tuned model
model.save_pretrained("path/to/save/model")
