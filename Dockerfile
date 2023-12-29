# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:2.15.0-gpu

# Install required packages for downloading Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    cuda-toolkit-12-2 \
    openssh-client \
    libcudnn8-dev

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container's working directory
COPY requirements.txt /app/requirements.txt

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# # Copy the installed dependencies from the builder image
# COPY --from=builder /usr/local/cuda /usr/local/cuda

# Copy the rest of the application files
COPY . .
ENV TF_FORCE_GPU_ALLOW_GROWTH=true


# Set the entry point to run main.py when the container starts
ENTRYPOINT ["python", "-m", "movie_review"]
