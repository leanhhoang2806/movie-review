# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu

# Install required packages for downloading cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.9.0.26-1+cuda12.2 \
    libcudnn8-dev=8.9.0.26-1+cuda12.2
# Print CUDA toolkit version
RUN nvcc --version > cuda_version.txt

# Print TensorFlow version
RUN python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" > tensorflow_version.txt

# Print cuDNN version
RUN cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 > cudnn_version.txt

# Display the contents of the version files
RUN cat cuda_version.txt
RUN cat tensorflow_version.txt
RUN cat cudnn_version.txt



# ==================

# # Install required packages for downloading Miniconda
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     python3-dev \
#     python3-pip \
#     cuda-toolkit-12-2 \
#     openssh-client

# # Set the working directory inside the container
# WORKDIR /app

# # Copy the requirements.txt file to the container's working directory
# COPY requirements.txt /app/requirements.txt

# # Install dependencies from requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # # Copy the installed dependencies from the builder image
# # COPY --from=builder /usr/local/cuda /usr/local/cuda

# # Copy the rest of the application files
# COPY . .

# # Set the entry point to run main.py when the container starts
# ENTRYPOINT ["python", "-m", "movie_review"]
