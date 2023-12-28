# Use the TensorFlow GPU base image
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04 as builder

# Install required packages for downloading Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    cuda-toolkit-12-2 \
    openssh-client

# Copy the installed dependencies from the builder image
COPY --from=builder /usr/local/cuda /usr/local/cuda
# Copy just the requirements file to leverage Docker cache
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set the entry point to run main.py when the container starts
ENTRYPOINT ["python", "-m", "movie_review"]
