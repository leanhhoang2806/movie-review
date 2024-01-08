

def predict_movie_name(extracted_df, model, tokenizer):
    # Tokenize the reviews (assuming they are already padded)
    tokenized_reviews = extracted_df['review_token'].tolist()

    # Predict movie names
    predicted_tokens = model.predict(tokenized_reviews)

    # Map predicted tokens back to words using the tokenizer
    predicted_movie_names = [tokenizer.decode(tokens) for tokens in predicted_tokens]

    # Add the predictions to the DataFrame
    output_df = extracted_df.copy()
    output_df['predicted_movie_name'] = predicted_movie_names

    return output_df[['review', 'movie_names', 'predicted_movie_name']]
