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
    echo "- GIT_NAME: ${GIT_NAME}"
    echo "- GITHUB_TOKEN: ${GITHUB_TOKEN}"
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

# Function to get remote image digest without pulling
get_remote_digest() {
    local image=$1
    # Login to GHCR first
    echo "$GITHUB_TOKEN" | docker login ghcr.io -u $GIT_NAME --password-stdin >/dev/null 2>&1
    # Get manifest and extract digest
    docker manifest inspect "$image" 2>/dev/null | grep -i '"digest"' | head -1 | tr -d ' ",' | cut -d':' -f2-3 || echo "none"
}

# Function to get local image digest
get_local_digest() {
    local image=$1
    # Add sha256: prefix to match remote digest format
    local digest=$(docker image inspect "$image" --format='{{index .Id}}' 2>/dev/null | cut -d':' -f2 || echo "none")
    if [ "$digest" != "none" ]; then
        echo "sha256:$digest"
    else
        echo "none"
    fi
}

# Check if we need to update the image
echo "🔍 Checking for updates..."
LOCAL_DIGEST=$(get_local_digest "ghcr.io/$GIT_NAME/hyperdas:latest")
REMOTE_DIGEST=$(get_remote_digest "ghcr.io/$GIT_NAME/hyperdas:latest")

echo "📝 Local image digest: ${LOCAL_DIGEST#sha256:}"
echo "📝 Remote image digest: ${REMOTE_DIGEST#sha256:}"

if [ "$LOCAL_DIGEST" != "$REMOTE_DIGEST" ]; then
    echo "🔄 New version detected, updating container..."
    docker pull "ghcr.io/$GIT_NAME/hyperdas:latest"
    # Remove existing container if it exists
    docker rm -f hyperdas 2>/dev/null || true
    docker rmi ghcr.io/$GIT_NAME/hyperdas:latest 2>/dev/null || true
else
    echo "✅ Container is up to date"
    # Check if container exists but is not running
    if docker ps -a | grep -q hyperdas && ! docker ps | grep -q hyperdas; then
        echo "🔄 Found stopped container, removing it..."
        docker rm -f hyperdas 2>/dev/null || true
    # Check if container is already running
    elif docker ps | grep -q hyperdas; then
        echo "🐳 Container is already running"
        docker exec -it hyperdas /bin/bash
        exit 0
    fi
fi

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
    -v ~/.ssh:/root/.ssh \
    -v ~/.config/hyperdas/.env:/workspace/HyperDAS/.env \
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