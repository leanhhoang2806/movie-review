import pandas as pd
import re
import string
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from sklearn.preprocessing import LabelEncoder

# Check if a GPU is available and set device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

csv_file_path = './IMDB Dataset.csv'
imdb_df = pd.read_csv(csv_file_path)

# Assuming 'review' is the column containing the reviews
reviews = imdb_df['review']

# Define a regular expression pattern for extracting text between quotation marks
movie_name_pattern = re.compile(r'"([^"]+)"')

# Initialize an empty list to store extracted data
extracted_data = []

# Loop through each review and extract text between quotation marks
for review in reviews:
    # Extract text between quotation marks using the defined pattern
    matches = re.findall(movie_name_pattern, review)
    
    # Filter out names that are not in capitalization format
    valid_names = [name for name in matches if name.istitle()]
    
    # Clean up punctuation at the beginning or end of each title
    cleaned_names = [name.strip(string.punctuation) for name in valid_names]
    
    # Append the review and cleaned text to the list if the list is not empty
    if cleaned_names and len(cleaned_names) == 1:
        corrected_name = cleaned_names[0].strip()
        if len(corrected_name) > 2: 
            extracted_data.append({'review': review, 'movie_names': corrected_name})

# Create a new DataFrame from the list of extracted data
extracted_df = pd.DataFrame(extracted_data)

# percentage of extraction successfully
percentage = (extracted_df.shape[0] / imdb_df.shape[0]) * 100
print(f"The amount of extracted data is {percentage:.2f} %")

# Display the new DataFrame
print(extracted_df.head())

# Assuming 'train_data' is your training dataset with reviews and extracted movie names
# 'imdb_reviews' is your IMDb dataset with reviews
train_data = extracted_df.sample(frac=0.8, random_state=42)  # Use 80% for training

# Assuming 'train_data' is your DataFrame
label_encoder = LabelEncoder()
train_data['encoded_labels'] = label_encoder.fit_transform(train_data['movie_names'])

# Tokenize the training data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_reviews = tokenizer(train_data['review'].tolist(), padding=True, truncation=True, return_tensors='pt')
tokenized_reviews = {key: value.to(device) for key, value in tokenized_reviews.items()}  # Move to GPU
labels = torch.tensor(train_data['encoded_labels'].tolist(), device=device)

# Create a PyTorch Dataset
dataset = TensorDataset(tokenized_reviews['input_ids'], tokenized_reviews['attention_mask'], labels)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the transformer model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(train_data['movie_names']))).to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'labels': batch[2].to(device)}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the trained model
model.save_pretrained('path/to/saved_model')

# Load the trained model for inference
loaded_model = BertForSequenceClassification.from_pretrained('path/to/saved_model').to(device)

# Evaluation on the test set
loaded_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_dataloader:
        inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'labels': batch[2].to(device)}
        outputs = loaded_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        
        # Collect predictions and labels
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch[2].cpu().numpy())

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Optionally, you can also print a classification report
class_report = classification_report(all_labels, all_preds)
print("Classification Report:")
print(class_report)
