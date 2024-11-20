#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./sync.sh <lambda_instance_ip>"
    exit 1
fi

LAMBDA_IP=$1
PROJECT_ROOT=$(git rev-parse --show-toplevel)

echo "🔄 Syncing code changes..."
rsync -avz --progress -e ssh \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'assets' \
    --exclude '.env' \
    --exclude 'node_modules' \
    --exclude '.venv' \
    $PROJECT_ROOT/ \
    ubuntu@$LAMBDA_IP:~/projects/HyperDAS/

echo "✅ Sync complete!"