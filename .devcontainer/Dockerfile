# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly from requirements
RUN pip install h5py scipy numpy matplotlib plotly nbdev pandas pytest

# Copy the current directory contents into the container
COPY . /workspace

# Command to run on container start
CMD ["tail", "-f", "/dev/null"]
