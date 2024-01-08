import pandas as pd
import tensorflow as tf

# Load CSV into Pandas DataFrame
def load_csv(file_path):
    return pd.read_csv(file_path)

# Load TensorFlow model
def load_tf_model(model_path):
    return tf.keras.models.load_model(model_path)

# Predict movie names using the loaded model
def predict_movie_names(model, data):
    # Assuming the model.predict function takes a Pandas DataFrame column as input
    predictions = model.predict(data['review_token'])
    data[['review', 'predicted_movie_name']] = predictions
    return data

# Save DataFrame to CSV
def save_to_csv(data, output_path):
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Replace 'your_data.csv' with the path to your CSV file
    input_csv_path = '/Documents/work/movie-review/extracted_data.csv'

    # Replace 'your_model.h5' with the path to your TensorFlow model file
    model_path = '/Documents/work/movie-review/best_model.h5'

    # Replace 'output_predictions.csv' with the desired output CSV file path
    output_csv_path = './output_predictions.csv'

    # Load data into Pandas DataFrame
    data = load_csv(input_csv_path)

    # Load TensorFlow model
    model = load_tf_model(model_path)

    # Predict movie names
    data = predict_movie_names(model, data)

    # Save results to CSV
    save_to_csv(data, output_csv_path)

    print(f"Predictions saved to {output_csv_path}")
