# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu as builder

# Install required packages for downloading Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    cuda-toolkit-12-2 \
    openssh-client

# Use a smaller base image for the final image
FROM tensorflow/tensorflow:latest-gpu

# Copy the installed dependencies from the builder image
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to run main.py when the container starts
ENTRYPOINT ["python", "-m", "movie_review.py"]
