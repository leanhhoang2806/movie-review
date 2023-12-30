# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:2.3.0-gpu

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install required packages for downloading Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    openssh-client 

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
