#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/setup_lambda.sh <lambda_instance_ip>"
    exit 1
fi

LAMBDA_IP=$1
PROJECT_ROOT=$(git rev-parse --show-toplevel)

setup_ssh_tunnel() {
    local ip=$1
    
    echo "🔍 Checking for existing tunnels..."
    # Using new ports
    lsof -ti:8765 | xargs kill -9 2>/dev/null || true
    lsof -ti:6380 | xargs kill -9 2>/dev/null || true
    lsof -ti:10001 | xargs kill -9 2>/dev/null || true
    
    echo "🔗 Creating new SSH tunnel..."
    # Updated port mappings
    ssh -N -L 8765:localhost:8765 -L 6380:localhost:6380 -L 10001:localhost:10001 ubuntu@$ip &
    
    echo $! > /tmp/hyperdas_tunnel.pid
    sleep 2
    
    echo "✅ SSH tunnel established. Ray dashboard available at http://localhost:8765"
}

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "📝 Loading local environment variables..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "🔄 Setting up Lambda Labs instance..."

# Create directories
ssh ubuntu@$LAMBDA_IP "mkdir -p ~/.config/hyperdas ~/projects/HyperDAS"

# Set correct permissions before syncing
echo "🔒 Setting correct SSH permissions..."
ssh ubuntu@$LAMBDA_IP "mkdir -p ~/.ssh && chmod 700 ~/.ssh"

# Sync credentials with correct permissions
echo "🔑 Syncing credentials..."
rsync -avz -e ssh \
    ~/.ssh/github* \
    ubuntu@$LAMBDA_IP:~/.ssh/ && \
ssh ubuntu@$LAMBDA_IP "chmod 600 ~/.ssh/github*"

# Add CLUSTER_IP to local .env if not already present
if ! grep -q "^CLUSTER_IP=" "$PROJECT_ROOT/.env"; then
    echo "📝 Adding CLUSTER_IP to local .env..."
    echo "CLUSTER_IP=$LAMBDA_IP" >> "$PROJECT_ROOT/.env"
    echo "✅ Added CLUSTER_IP=$LAMBDA_IP to local environment"
fi

# Sync environment file and append GITHUB_TOKEN if it exists locally
echo "📄 Syncing .env file..."
rsync -avz -e ssh \
    $PROJECT_ROOT/.env \
    ubuntu@$LAMBDA_IP:~/projects/HyperDAS/

# Sync code (excluding unnecessary files)
echo "📦 Syncing code..."
rsync -avz --progress -e ssh \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'assets' \
    --exclude '.env' \
    --exclude 'node_modules' \
    --exclude '.venv' \
    $PROJECT_ROOT/ \
    ubuntu@$LAMBDA_IP:~/projects/HyperDAS/

echo "🔧 Making scripts executable..."
ssh ubuntu@$LAMBDA_IP "chmod +x ~/projects/HyperDAS/scripts/*.sh"

echo "🚀 Starting container on Lambda instance..."
ssh -t ubuntu@$LAMBDA_IP "cd ~/projects/HyperDAS && ./scripts/lambdalabs.sh"

echo "🔗 Setting up SSH tunnel..."
setup_ssh_tunnel $LAMBDA_IP

# If the SSH session ends, clean up the tunnel
trap 'pkill -F /tmp/hyperdas_tunnel.pid; rm /tmp/hyperdas_tunnel.pid' EXIT

echo "✅ Setup complete!"
