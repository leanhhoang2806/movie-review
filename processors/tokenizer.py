import re
import string
import pandas as pd
from itertools import chain
from transformers import BertTokenizer

class TokenizedText:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def _get_bert_token(self, word):
        token_ids = self.tokenizer.encode(word, add_special_tokens=False, padding=True, truncation=True, max_length=512)
        return token_ids
    
    def tokenized(self, data_frame):
        movie_name_pattern = re.compile(r'"([^"]+)"')

        # Initialize an empty list to store extracted data
        extracted_data = []

        # Loop through each review and extract text between quotation marks
        for review in data_frame['review']:
            # Extract text between quotation marks using the defined pattern
            matches = re.findall(movie_name_pattern, review)

            # Filter out names that are not in capitalization format
            valid_names = [name for name in matches if name.istitle()]

            # Clean up punctuation at the beginning or end of each title
            cleaned_names = [name.strip(string.punctuation) for name in valid_names]

            # Append the review and cleaned text to the list if the list is not empty
            if cleaned_names and len(cleaned_names) == 1:
                corrected_name = cleaned_names[0].strip()
                if len(corrected_name.split()) == 2:
                    # Get BERT token for movie name
                    movie_name_tokens_nested = [self._get_bert_token(token) for token in corrected_name.split()]
                    movie_name_tokens_flatten = list(chain(*movie_name_tokens_nested))

                    # Get BERT token for review
                    review_tokens = self._get_bert_token(review)
                    extracted_data.append({'review_token': review_tokens, 'movie_names_token': movie_name_tokens_flatten, "review": review, "movie_names": corrected_name})
        return pd.DataFrame(extracted_data)
