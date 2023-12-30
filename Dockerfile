# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:2.3.0-gpu


# Set the working directory inside the container
WORKDIR /app
RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip setuptools

# Copy the requirements.txt file to the container's working directory
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application files
COPY . .


# Set the entry point to run main.py when the container starts
ENTRYPOINT ["python", "-m", "movie_review"]
