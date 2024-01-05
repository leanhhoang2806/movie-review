
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Normalizer:
    def normalize(self, extracted_df):
        # Determine the maximum lengths
        max_review_length = max(extracted_df['review_token'].apply(len))
        max_movie_length = max(extracted_df['movie_names_token'].apply(len))

        # Apply padding to the DataFrame
        extracted_df['review_token'] = extracted_df['review_token'].apply(lambda x: pad_sequences([x], maxlen=max_review_length, padding='post', truncating='post')[0])
        extracted_df['movie_names_token'] = extracted_df['movie_names_token'].apply(lambda x: pad_sequences([x], maxlen=max_movie_length, padding='post', truncating='post')[0])
        return extracted_df[['review_token', 'movie_names_token']]