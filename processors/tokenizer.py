import re
import string
import pandas as pd
from itertools import chain
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_bert_token(word, tokenizer):
    token_ids = tokenizer.encode(word, add_special_tokens=False, padding=True, truncation=True, max_length=512)
    return token_ids

def extract_movie_data(review, movie_name_pattern):
    matches = re.findall(movie_name_pattern, review)
    valid_names = [name for name in matches if name.istitle()]
    cleaned_names = [name.strip(string.punctuation) for name in valid_names]
    if cleaned_names and len(cleaned_names) == 1:
        return cleaned_names[0].strip()
    return None

def preprocess_review_data(review, tokenizer, movie_name_pattern):
    movie_name = extract_movie_data(review, movie_name_pattern)
    if movie_name and len(movie_name.split()) == 2:
        movie_name_tokens_nested = [get_bert_token(token, tokenizer) for token in movie_name.split()]
        movie_name_tokens_flatten = list(chain(*movie_name_tokens_nested))
        review_tokens = get_bert_token(review, tokenizer)
        if review_tokens:
            return {'review_token': review_tokens, 'movie_names_token': movie_name_tokens_flatten,
                    "review": review, "movie_names": movie_name}
    return None

def preprocess_df(extracted_data, max_review_length, max_movie_length):
    extracted_df = pd.DataFrame(extracted_data)
    extracted_df = extracted_df.dropna(subset=['review_token'])
    extracted_df['review_token'] = extracted_df['review_token'].apply(
        lambda x: pad_sequences([x], maxlen=max_review_length, padding='post', truncating='post')[0])
    extracted_df['movie_names_token'] = extracted_df['movie_names_token'].apply(
        lambda x: pad_sequences([x], maxlen=max_movie_length, padding='post', truncating='post')[0])
    return extracted_df[['review_token', 'movie_names_token']]