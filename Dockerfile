# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu as builder

# Install required packages for downloading Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    cuda-toolkit-12-2 \
    openssh-client

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container's working directory
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Use a smaller base image for the final image
FROM tensorflow/tensorflow:latest-gpu

# Copy the installed dependencies from the builder image
COPY --from=builder /usr/local/cuda /usr/local/cuda

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of the application files
COPY . .

# Set the entry point to run main.py when the container starts
ENTRYPOINT ["python", "movie_review"]
