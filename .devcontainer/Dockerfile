# Start from the NVIDIA CUDA base image with PyTorch support
FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu20.04

# Set the working directory in the container
WORKDIR /orbit-generation

# Install system dependencies and upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libhdf5-dev \
    python3-pip \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip

# Install PyTorch, torchvision, and torchaudio for CUDA 12.1
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Clone your repository
RUN git clone --branch move_to_pytorch https://github.com/alvaro-francisco-gil/orbit-generation.git /orbit-generation \
    && pip3 install $(awk -F= '/^requirements/ {print $2}' /orbit-generation/settings.ini)

# Install Python dependencies
RUN pip3 install -r /orbit-generation/requirements.txt

# Install a new Jupyter kernel
RUN python3 -m ipykernel install --name pytorch --display-name "Python (PyTorch)"

# Install nbdev for notebook development
RUN pip3 install nbdev
RUN nbdev_install_quarto

# Command to run on container start
CMD ["bash"]
