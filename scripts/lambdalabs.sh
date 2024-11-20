#!/bin/bash

# Add user to docker group if not already a member
if ! groups "$(id -un)" | grep -q "\bdocker\b"; then
    sudo adduser "$(id -un)" docker
fi

# Check if GITHUB_TOKEN is provided
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable is not set"
    exit 1
fi

# Remove existing container if it exists
docker rm -f hyperdas 2>/dev/null

# Login to GHCR
echo "$GITHUB_TOKEN" | docker login ghcr.io -u sidnb13 --password-stdin

# Run container
docker run -d \
    --name hyperdas \
    --gpus all \
    --ipc host \
    -v ./assets:/workspace/HyperDAS/assets \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e GITHUB_TOKEN=$GITHUB_TOKEN \
    ghcr.io/sidnb13/hyperdas:latest \
    tail -f /dev/null

# Wait a moment for container to start
sleep 2

# Check if container is running
if docker ps | grep -q hyperdas; then
    echo "Container started successfully. Entering container..."
    docker exec -it hyperdas /bin/bash
else
    echo "Error: Container failed to start"
    docker logs hyperdas
    exit 1
fi