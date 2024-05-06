# Start from the NVIDIA CUDA base image that supports TensorFlow with GPU
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /orbit-generation

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

# Install Python dependencies including TensorFlow, Keras, and wandb
RUN pip3 install h5py scipy numpy matplotlib plotly nbdev pandas pytest tsgm wandb umap-learn

# Clone the repository
RUN git clone https://github.com/alvaro-francisco-gil/orbit-generation.git /orbit-generation

# Configure nbdev
RUN nbdev_install_quarto

# Set the working directory to the cloned repository
WORKDIR /orbit-generation

# Copy the data folder into the Docker image
COPY data /data

# Command to run on container start
CMD ["bash"]