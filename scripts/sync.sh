#!/bin/bash
set -e

if [ -z "$1" ] || { [ "$2" != "push" ] && [ "$2" != "pull" ] && [ -n "$2" ]; }; then
    echo "Usage: ./sync.sh <lambda_instance_ip> [push|pull] [--no-git]"
    echo "  push: sync local changes to remote (default)"
    echo "  pull: sync remote changes to local"
    echo "  --no-git: exclude .git directory from sync"
    exit 1
fi

LAMBDA_IP=$1
DIRECTION=${2:-push}  # Default to push if no direction specified
PROJECT_ROOT=$(git rev-parse --show-toplevel)

# Common rsync excludes
EXCLUDES=(
    --exclude '__pycache__'
    --exclude '*.pyc'
    --exclude 'assets'
    --exclude '.env'
    --exclude 'node_modules'
    --exclude '.venv'
)

# Add .git to excludes if --no-git is specified
if [[ " $* " == *" --no-git "* ]]; then
    EXCLUDES+=(--exclude '.git')
fi

if [ "$DIRECTION" = "push" ]; then
    echo "🔄 Pushing local changes to remote..."
    rsync -avz --progress -e ssh \
        "${EXCLUDES[@]}" \
        "$PROJECT_ROOT/" \
        "ubuntu@$LAMBDA_IP:~/projects/HyperDAS/"
else
    echo "🔄 Pulling remote changes to local..."
    rsync -avz --progress -e ssh \
        "${EXCLUDES[@]}" \
        "ubuntu@$LAMBDA_IP:~/projects/HyperDAS/" \
        "$PROJECT_ROOT/"
fi

echo "✅ Sync complete!"