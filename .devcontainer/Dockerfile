# Start from the NVIDIA CUDA base image that supports TensorFlow with GPU
FROM tensorflow/tensorflow:2.9.1-gpu

# Set the working directory in the container
WORKDIR /orbit-generation

# Install system dependencies, upgrade pip, clone repository, and install Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libhdf5-dev \
    python3-pip \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip \
    && pip3 install ipykernel \
    && git clone --branch add_tsgm https://github.com/alvaro-francisco-gil/orbit-generation.git /orbit-generation \
    && pip3 install -r /orbit-generation/requirements.txt

# Install a new Jupyter kernel
RUN python3 -m ipykernel install --name tensorflow-gpu --display-name "Python (tensorflow-gpu)"

# Configure nbdev
RUN pip3 install nbdev
RUN nbdev_install_quarto

# Command to run on container start
CMD ["bash"]
