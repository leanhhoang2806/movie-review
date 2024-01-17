import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf
from data_loader.load_imdb import load_imdb_dataset
from processors.tokenizer import preprocess_review_data, preprocess_df
from training_strategy.distributed_training import grid_search
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset
from torch import LongTensor
from torch.nn import CrossEntropyLoss
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
        input_text = f"Q: {pair['question']} A: {pair['answer']}"
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, return_tensors='pt', truncation=True, padding='max_length')
        return {'input_ids': input_ids}

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
    print(focus_data.head())

    training_data = [ {"answer": row['review'], "question": row['movie_names']} for _, row in focus_data.iterrows()]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    tokenized_dataset = QADataset(training_data, tokenizer)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()

    train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

    # Training loop
    for _ in range(3):  # Adjust the number of epochs
        for batch in train_dataloader:
            input_ids = batch['input_ids']
            labels = input_ids.clone()

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    prompt = "What is the capital of France?"
    response = generate_response(prompt, model, tokenizer)
    print(response)
    
if __name__ == "__main__":
    main()

