# Use a base image with CUDA and cuDNN installed, suitable for development
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

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
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip

# Install Miniconda
# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy
ENV PATH=$CONDA_DIR/bin:$PATH

# 1. Clone the repo
RUN git clone https://github.com/azencot-group/ImagenTime /external_libs/ImagenTime

# 2. Copy the conda env file from that repo
COPY /external_libs/ImagenTime/environment.yaml /tmp/imagen_env.yaml
RUN conda env create -f /tmp/imagen_env.yaml -n imagentime

# Install PyTorch for CUDA 12.1
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu121

# Clone your repository
RUN git clone --branch development https://github.com/alvaro-francisco-gil/orbit-generation.git /orbit-generation \
    && pip3 install $(awk -F= '/^requirements/ {print $2}' /orbit-generation/settings.ini)

# Install ipykernel and Jupyter for notebook support
RUN pip3 install ipykernel jupyter

# Install a new Jupyter kernel
RUN python3 -m ipykernel install --name pytorch --display-name "Python (PyTorch)"

# Install nbdev for notebook development
RUN pip3 install nbdev
RUN nbdev_install_quarto

# Install PyJulia
RUN pip3 install julia

# Install Julia 1.10.5
ARG JULIA_VERSION=1.10.5
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    && tar -xvzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    && mv julia-${JULIA_VERSION} /opt/julia \
    && ln -s /opt/julia/bin/julia /usr/local/bin/julia \
    && rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz

# Set up Julia environment
ENV JULIA_DEPOT_PATH=/opt/julia_depot
RUN mkdir -p $JULIA_DEPOT_PATH

# Install Julia packages (add your required packages here)
RUN julia -e 'using Pkg; Pkg.add(["DifferentialEquations", "Polynomials", "LinearAlgebra", "MAT", "NPZ"])'

# Install IJulia for Jupyter integration
RUN julia -e 'using Pkg; Pkg.add("IJulia")'

# Add Julia kernel to Jupyter
RUN julia -e 'using IJulia; IJulia.installkernel("Julia")'

# Initialize PyJulia
RUN python3 -c "import julia; julia.install()"

# Command to run on container start
CMD ["bash"]