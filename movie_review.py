import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, AdamW
import pandas as pd
model.save_pretrained('custom_qa_model')


def main():

    csv_file_path = './IMDB Dataset.csv'
    imdb_df = pd.read_csv(csv_file_path)
    print(f"Example of the loaded data")
    print(imdb_df.head())
    
    # concat the column into a massive text
    all_text = imdb_df['review'].str.cat(sep=" ")
    print(all_text)

    
if __name__ == "__main__":
    main()

