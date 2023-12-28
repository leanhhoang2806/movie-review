# Use the TensorFlow GPU base image
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04 as builder

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    openssh-client

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of the application files
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Use a smaller base image for the final image
FROM tensorflow/tensorflow:latest-gpu

# Copy the installed dependencies from the builder image
COPY --from=builder /usr/local/cuda /usr/local/cuda

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Set the entry point to run main.py when the container starts
ENTRYPOINT ["python", "-m", "movie_review"]

