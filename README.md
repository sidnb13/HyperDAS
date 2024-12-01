# HyperDAS

## Quick Start

1. Copy the environment template and configure your variables:
```bash
cp .env.example .env
```

2. Set required variables in `.env`:
```bash
GIT_NAME="your-github-username"
GIT_EMAIL="your-github-email"
GITHUB_TOKEN="your-github-token"  # Needs read:packages and write:packages scopes
...etc.
```

You will need a Github Personal Access Token with the `read:packages` and `write:packages` scopes in ordet to push and pull from the GitHub Container Registry.

3. Launch on Lambda Labs:

The container starts a local Ray cluster, to which you can submit & queue GPU jobs from within the Docker container.
The script will expose an SSH tunnel to the remote machine, allowing you to access the Ray dashboard at `http://localhost:8765` to monitor job status. It will sync code, SSH keys, and secrets to the remote machine, and pull the latest image from GHCR & start the container.

```bash
./scripts/setup_lambda.sh <instance-ip>
```

4. Launching jobs:

We use [Hydra](https://github.com/facebookresearch/hydra) for configuration management. You can specify an experiment and train a model with `python train.py experiment=<experiment_name>/<experiment_config>` which is a collection of model, training, and loss hyperparameters. Specify configuration overrides under the experiment config, e.g. `experiment.model.das_dimension=128`.

Queue a non-blocking job to the local ray cluster with `ray job submit --entrypoint-num-gpus=<num-gpus> --no-wait -- python train.py experiment=<experiment_name>/<experiment_config>`.

## Manual Docker Setup

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

The build workflow located under `.github/workflows/build.yml` will automatically build and push the image to the GitHub Container Registry when the dockerfiles or dependencies change on push to the main branch.

3. Run locally (optional):
```bash
docker run -it --gpus all \
    --name hyperdas \
    --ipc host \
    -v .:/workspace/HyperDAS \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    ghcr.io/$GIT_NAME/hyperdas:latest
```