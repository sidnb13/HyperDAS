#!/bin/bash

# Login to GHCR
echo $GITHUB_PAT | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

# Pull the latest image
docker pull ghcr.io/$GITHUB_USERNAME/hyperdas:latest

# Run the container
docker run -d \
  --gpus all \
  -p 8888:8888 \
  ghcr.io/${GITHUB_USERNAME}/hyperdas:latest