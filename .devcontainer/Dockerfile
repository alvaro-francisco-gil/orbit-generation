# Start from the NVIDIA CUDA base image that supports TensorFlow with GPU
FROM tensorflow/tensorflow:2.9.1-gpu

# Set the working directory in the container
WORKDIR /orbit-generation

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libhdf5-dev \
    python3-pip \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid compatibility issues
RUN pip3 install --upgrade pip

# Clone the repository
RUN git clone --branch add_tsgm https://github.com/alvaro-francisco-gil/orbit-generation.git /orbit-generation

# Install Python dependencies from requirements.txt
RUN pip3 install -r /orbit-generation/requirements.txt

# Set the working directory to the cloned repository
WORKDIR /orbit-generation

# Copy the data folder into the Docker image
COPY data /data

# Install a new Jupyter kernel
RUN python3 -m ipykernel install --name tensorflow-gpu --display-name "Python (tensorflow-gpu)"

# Configure nbdev
RUN nbdev_install_quarto

# Command to run on container start
CMD ["bash"]