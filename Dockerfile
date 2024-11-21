FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ARG GIT_EMAIL
ARG GIT_NAME

ENV GIT_EMAIL=${GIT_EMAIL}
ENV GIT_NAME=${GIT_NAME}

WORKDIR /workspace/HyperDAS
COPY ./ .

# Combine all apt-get commands and cleanup in a single layer
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libdrm-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libsystemd-dev \
    libsystemd0 \
    libudev-dev \
    libudev0 \
    nano \
    ncdu \
    nvtop \
    openssh-client \
    python3-pip \
    python3.10-dev \
    screen \
    sudo \
    vim \
    wget \
    zsh && \
    sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p https://github.com/zsh-users/zsh-history-substring-search \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    echo "set -o vi" >> ~/.bashrc && \
    git config --global core.editor "vim"

# Set zsh as default shell
SHELL ["/bin/zsh", "-c"]

# Set environment variables
ENV PATH="/usr/local/bin:${PATH}" \
    PYTHONPATH="/usr/local/lib/python3.10/dist-packages:${PYTHONPATH}"

# Configure git
RUN git config --global user.email ${GIT_EMAIL} && \
    git config --global user.name ${GIT_NAME}

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy and set up entrypoint script
COPY scripts/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]