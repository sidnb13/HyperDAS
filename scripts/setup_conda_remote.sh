#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/setup_conda_remote.sh <remote_ip> [environment_name]"
    exit 1
fi

REMOTE_IP=$1
ENV_NAME=${2:-$(conda info --envs | grep '*' | awk '{print $1}')}  # Use provided name or get current env
PROJECT_ROOT=$(git rev-parse --show-toplevel)
PYTHON_VERSION=$(python -V | cut -d' ' -f2)  # Get current Python version

# ... existing env loading code ...

echo "🔄 Setting up conda environment on remote machine..."

# Install Miniconda if not already installed
echo "📦 Ensuring Miniconda is installed..."
ssh ubuntu@$REMOTE_IP "if [ ! -f ~/miniconda3/bin/conda ]; then \
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    echo 'export PATH=~/miniconda3/bin:\$PATH' >> ~/.bashrc && \
    source ~/.bashrc; \
fi"

# Create conda environment with same Python version
echo "🐍 Creating conda environment: $ENV_NAME..."
ssh ubuntu@$REMOTE_IP "source ~/.bashrc && \
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION && \
    conda activate $ENV_NAME && \
    pip freeze > /tmp/requirements.txt"

# Sync project code (excluding unnecessary files)
echo "📦 Syncing code..."
rsync -avz --progress -e ssh \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'node_modules' \
    --exclude '.venv' \
    --exclude '*.egg-info' \
    --exclude '.git' \
    $PROJECT_ROOT/ \
    ubuntu@$REMOTE_IP:~/projects/$ENV_NAME/

# Install dependencies
echo "📚 Installing dependencies..."
ssh ubuntu@$REMOTE_IP "source ~/.bashrc && \
    conda activate $ENV_NAME && \
    cd ~/projects/$ENV_NAME && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    if [ -f setup.py ]; then pip install -e .; fi"

echo "✅ Setup complete! You can now connect to the remote environment using:"
echo "ssh ubuntu@$REMOTE_IP"
echo "conda activate $ENV_NAME"