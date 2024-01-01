# import pandas as pd
# import re
# import string
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from transformers import BertTokenizer
# from transformers import TFBertForSequenceClassification
# import numpy as np


# # Load the IMDb dataset
# csv_file_path = './IMDB Dataset.csv'
# imdb_df = pd.read_csv(csv_file_path)

# # Define a regular expression pattern for extracting text between quotation marks
# movie_name_pattern = re.compile(r'"([^"]+)"')

# # Initialize an empty list to store extracted data
# extracted_data = []

# # Loop through each review and extract text between quotation marks
# for review in imdb_df['review']:
#     # Extract text between quotation marks using the defined pattern
#     matches = re.findall(movie_name_pattern, review)

#     # Filter out names that are not in capitalization format
#     valid_names = [name for name in matches if name.istitle()]

#     # Clean up punctuation at the beginning or end of each title
#     cleaned_names = [name.strip(string.punctuation) for name in valid_names]

#     # Append the review and cleaned text to the list if the list is not empty
#     if cleaned_names and len(cleaned_names) == 1:
#         corrected_name = cleaned_names[0].strip()
#         if len(corrected_name) > 2:
#             extracted_data.append({'review': review, 'movie_names': corrected_name})

# # Create a new DataFrame from the list of extracted data
# extracted_df = pd.DataFrame(extracted_data)

# # Display the new DataFrame
# print(extracted_df.head())

# # Assuming 'train_data' is your training dataset with reviews and extracted movie names
# train_data = extracted_df.sample(frac=0.8, random_state=42)  # Use 80% for training
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Tokenize the entire "review" column
# extracted_df['tokenized_reviews'] = extracted_df['review'].apply(lambda x: tokenizer(x, padding=True, truncation=True, return_tensors='tf', max_length=512)['input_ids'])

# # Extract only the 'input_ids' from the 'tokenized_reviews' column
# extracted_df['input_ids'] = extracted_df['tokenized_reviews'].apply(lambda x: np.array(x['input_ids'])[0])

# # Print the DataFrame with 'input_ids'
# print(extracted_df[['review', 'input_ids']])


import pandas as pd
from transformers import BertTokenizer
import numpy as np

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example DataFrame
data = {'review': ['This is the first review.', 'Another example review.', 'And one more review.']}
df = pd.DataFrame(data)

# Tokenize the entire "review" column
df['tokenized_reviews'] = df['review'].apply(lambda x: tokenizer(x, padding=True, truncation=True, return_tensors='tf', max_length=512))

# Convert the 'input_ids' tensor to a NumPy array
df['input_ids'] = df['tokenized_reviews'].apply(lambda x: np.array(x['input_ids']))

# Print the DataFrame with 'input_ids'
print(df[['review', 'input_ids']])
