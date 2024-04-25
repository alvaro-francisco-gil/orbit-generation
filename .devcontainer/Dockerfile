# Start from the NVIDIA CUDA base image that supports TensorFlow with GPU
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libhdf5-dev \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid compatibility issues
RUN pip3 install --upgrade pip

# Install Python dependencies including TensorFlow Keras and wandb
RUN pip3 install h5py scipy numpy matplotlib plotly nbdev pandas pytest tsgm tensorflow-keras wandb

# Copy the current directory contents into the container
COPY . /workspace

# Command to run on container start
CMD ["bash"]
