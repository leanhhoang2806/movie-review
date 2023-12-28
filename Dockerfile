# Use the TensorFlow GPU base image
# Use the NVIDIA CUDA image with specified versions
FROM nvidia/cuda:12.2-cudnn8-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.9 \
    python3.9-dev \
    python3-pip \
    clang-16 \
    bazel-6.1.0

# Create a symbolic link for Python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
# Copy just the requirements file to leverage Docker cache
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set the entry point to run main.py when the container starts
ENTRYPOINT ["python", "-m", "movie_review"]
