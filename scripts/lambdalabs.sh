#!/bin/bash
set -e

echo "🚀 Initializing HyperDAS container setup..."

# Add user to docker group if not already a member
if ! groups "$(id -un)" | grep -q "\bdocker\b"; then
    echo "👥 Adding user to docker group..."
    sudo adduser "$(id -un)" docker
fi

# Source environment variables
ENV_FILE="$HOME/projects/HyperDAS/.env"  # Changed from "~/projects/HyperDAS/.env"
if [ -f "$ENV_FILE" ]; then
    echo "📝 Loading environment variables from $ENV_FILE..."
    set -a
    source "$ENV_FILE"
    set +a
    
    echo "🔍 Current environment:"
    echo "- GIT_NAME: ${GIT_NAME:-not set}"
    echo "- GITHUB_TOKEN: ${GITHUB_TOKEN:+set|not set}"
else
    echo "❌ Error: .env file not found at $ENV_FILE"
    echo "Current location: $(pwd)"  # Added for debugging
    echo "Contents of ~/projects/HyperDAS/: $(ls ~/projects/HyperDAS/)"  # Added for debugging
    exit 1
fi

# Check required variables
if [ -z "$GIT_NAME" ] || [ -z "$GITHUB_TOKEN" ] || [ -z "$GIT_EMAIL" ]; then
    echo "❌ Error: Required environment variables not set"
    echo "Required: GIT_NAME, GIT_EMAIL,GITHUB_TOKEN"
    exit 1
fi

# Remove existing container if it exists
echo "🧹 Cleaning up any existing containers..."
docker rm -f hyperdas 2>/dev/null || true

# Login to GHCR
echo "🔑 Authenticating with GitHub Container Registry..."
echo "$GITHUB_TOKEN" | docker login ghcr.io -u $GIT_NAME --password-stdin

echo "📦 Launching container..."
echo "-----------------------------------"

# Run container
docker run -d \
    --name hyperdas \
    --gpus all \
    --ipc host \
    -v ~/projects/HyperDAS:/workspace/HyperDAS \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.ssh/id_ed25519:/root/.ssh/id_ed25519:ro \
    -v ~/.config/hyperdas/.env:/workspace/HyperDAS/.env:ro \
    -v ~/.gitconfig:/root/.gitconfig:ro \
    -v vscode-extensions:/root/.vscode-server/extensions \
    -v vscode-extensions-insiders:/root/.vscode-server-insiders/extensions \
    -e GITHUB_TOKEN=$GITHUB_TOKEN \
    ghcr.io/$GIT_NAME/hyperdas:latest \
    sleep infinity  # Changed from tail -f /dev/null for better Dev Container support

# Wait a moment for container to start
echo "⏳ Waiting for container to initialize..."
sleep 2

# Check if container is running
if docker ps | grep -q hyperdas; then
    echo "✅ Container started successfully!"
    echo "🖥️  Available GPUs:"
    docker exec hyperdas nvidia-smi --list-gpus
    echo "-----------------------------------"
    echo "🔌 Connecting to container..."
    echo "📝 Type 'exit' to leave the container"
    echo "-----------------------------------"
    docker exec -it hyperdas /bin/bash
else
    echo "❌ Error: Container failed to start"
    echo "📋 Container logs:"
    echo "-----------------------------------"
    docker logs hyperdas
    exit 1
fi