FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ARG GIT_EMAIL
ARG GIT_NAME

ENV GIT_EMAIL=${GIT_EMAIL}
ENV GIT_NAME=${GIT_NAME}

WORKDIR /workspace/HyperDAS

COPY requirements.txt .
COPY scripts/entrypoint.sh /usr/local/bin/

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

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
    software-properties-common \
    screen \
    sudo \
    vim \
    wget \
    zsh && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3.12-venv && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    echo "set -o vi" >> ~/.bashrc && \
    git config --global core.editor "vim"

# Set environment variables
ENV PATH="/usr/local/bin:${PATH}" \
    PYTHONPATH="/usr/local/lib/python3.10/dist-packages:${PYTHONPATH}"

# Configure git (modified to use zsh syntax)
RUN git config --global user.email "${GIT_EMAIL}" && \
    git config --global user.name "${GIT_NAME}"

# Set zsh as default shell
SHELL ["/bin/zsh", "-c"]

COPY requirements.txt .
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --system --no-cache-dir --upgrade pip setuptools wheel && \
    /root/.local/bin/uv pip install --system --no-cache-dir -r requirements.txt

# Copy and set up entrypoint script
COPY scripts/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]