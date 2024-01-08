

def predict_movie_name(extracted_df, model):
    extracted_df['predicted_movie_name'] = extracted_df['review_token'].apply(lambda x: model.predict(x))
    return extracted_df[['review', 'movie_names', 'predicted_movie_name']]