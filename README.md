# HyperDAS

## Quick Start

1. Copy the environment template and configure your variables:
```bash
cp .env.example .env
```

2. Set required variables in `.env`:
```bash
GIT_NAME="your-github-username"
GITHUB_TOKEN="your-github-token"  # Needs read:packages scope
```

3. Launch on Lambda Labs:
```bash
# First time setup
./scripts/setup_lambda.sh <instance-ip>

# Subsequent code syncs
./scripts/sync.sh <instance-ip>
```

## Docker Setup

1. Build the container locally (need a GPU-enabled Linux machine with up-to-date NVIDIA drivers):
```bash
docker build -t ghcr.io/$GIT_NAME/hyperdas:latest .
```

2. Push to GitHub Container Registry:
```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u $GIT_NAME --password-stdin

# Push the image
docker push ghcr.io/$GIT_NAME/hyperdas:latest
```

3. Run locally (optional):
```bash
docker run -it --gpus all \
    --name hyperdas \
    --ipc host \
    -v .:/workspace/HyperDAS \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    ghcr.io/$GIT_NAME/hyperdas:latest
```

## Scripts

- `setup_lambda.sh`: Initial setup of Lambda instance
  - Syncs credentials and environment
  - Sets up Docker container
  - Mounts necessary volumes
  - Enables GPU access

- `sync.sh`: Quick sync of code changes
  - Excludes unnecessary files (.git, __pycache__, etc.)
  - Preserves running container state

- `lambdalabs.sh`: Container management
  - Handles Docker setup
  - Manages environment variables
  - Provides GPU access
  - Sets up development environment

## Container Features

- CUDA-enabled PyTorch environment
- Pre-configured for HuggingFace
- VSCode remote development support
- Persistent extension storage
- SSH key forwarding
- Environment variable management

## Development

1. Connect to your instance via VSCode:
   - Install "Remote - SSH" extension
   - Add SSH config for your Lambda instance
   - Connect to `ubuntu@<instance-ip>`

2. Container will automatically mount:
   - Project files: `/workspace/HyperDAS`
   - HuggingFace cache: `~/.cache/huggingface`
   - SSH keys and Git config
   - Environment variables

## Requirements

- Local:
  - SSH key pair
  - GitHub token with `read:packages` scope
  - Lambda Labs account
  - Docker with NVIDIA Container Toolkit (for local builds)

- Remote:
  - Lambda Labs instance with GPU
  - Ubuntu-based system
  - Docker support
