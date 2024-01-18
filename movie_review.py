import re
import pandas as pd
from transformers import BertTokenizer
import tensorflow as tf
from data_loader.load_imdb import load_imdb_dataset
from processors.tokenizer import preprocess_review_data
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch

def generate_response(prompt, model, tokenizer, max_length=100):
    input_text = f"Q: {prompt} A:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        input_text = f"question: {pair['question']} context: {pair['answer']}"
        input_ids = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )
        return {
            'input_ids': input_ids['input_ids'].squeeze(),
            'attention_mask': input_ids['attention_mask'].squeeze(),
            'start_positions': torch.tensor(pair['start_positions']),
            'end_positions': torch.tensor(pair['end_positions'])
        }

def main():
    tf.keras.backend.clear_session()

    csv_file_path = './IMDB Dataset.csv'
    imdb_df = load_imdb_dataset(csv_file_path)
    print(f"target for testing predictionis : {imdb_df.iloc[0]}")

    movie_name_pattern = re.compile(r'"([^"]+)"')

    extracted_data = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for review in imdb_df['review']:
        data = preprocess_review_data(review, tokenizer, movie_name_pattern)
        if data:
            extracted_data.append(data)

    extracted_df = pd.DataFrame(extracted_data)
    focus_data = extracted_df[['review', 'movie_names']]
    for _, row in focus_data.iterrows():
        print(row)
        print(re.search(row['movie_names'], row['review']).start() or 0)
        print(re.search(row['movie_names'], row['review']).end() or 0)
        break

    training_data = [ {
        "answer": row['review'], 
        "question": row['movie_names'], 
        "start_positions": re.search(re.escape(row['movie_names']), row['review']).start(),
        "end_positions": re.search(re.escape(row['movie_names']), row['review']).end()} 
        for _, row in focus_data.iterrows()]

    print(f"position of the answer {training_data[0]}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    tokenized_dataset = QADataset(training_data, tokenizer)

    # Load pre-trained DistilBERT model
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased")

    # Define optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    for epoch in range(3):  # Adjust the number of epochs
        for batch in DataLoader(tokenized_dataset, batch_size=2, shuffle=True):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
if __name__ == "__main__":
    main()

