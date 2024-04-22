# Start from the NVIDIA CUDA base image that supports TensorFlow with GPU
FROM tensorflow/tensorflow:2.9.1-gpu

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

# Install Python dependencies directly from requirements
RUN pip3 tensorflow-gpu install h5py scipy numpy matplotlib plotly nbdev pandas pytest tsgm

# Copy the current directory contents into the container
COPY . /workspace

# Command to run on container start
CMD ["bash"]

