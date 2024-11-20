FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /workspace/HyperDAS

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo vim curl nano ncdu screen \
        build-essential libsystemd0 libsystemd-dev libudev0 libudev-dev cmake \
        libncurses5-dev libncursesw5-dev git libdrm-dev python3-pip python3.10-dev nvtop && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create a symbolic link to alias python to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Ensure pip is up to date
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set the Python path
ENV PATH="/usr/local/bin:${PATH}" \
    PYTHONPATH="/usr/local/lib/python3.10/dist-packages:${PYTHONPATH}"

# Set up git, vim, and other tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends git vim sudo curl && \
    rm -rf /var/lib/apt/lists/* && \
    echo "set -o vi" >> ~/.bashrc && \
    git config --global core.editor "vim"

# Set git configs (using ENV for environment variables)
ENV GIT_EMAIL=${GIT_EMAIL:-"default@example.com"}
ENV GIT_NAME=${GIT_NAME:-"Default User"}
RUN git config --global user.email ${GIT_EMAIL} && \
    git config --global user.name ${GIT_NAME}

WORKDIR /workspace/HyperDAS
