#!/bin/bash
set -e  # Exit on error

# Print system information
echo "🖥️  Container System Information:"
nvidia-smi
echo "-----------------------------------"

# Set up SSH directory permissions
if [ -f "/root/.ssh/github" ]; then
    echo "🔑 Setting up SSH permissions..."
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/github
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA is available"
    # Get number of available GPUs
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "📊 Found ${GPU_COUNT} GPU(s)"
    # Set CUDA_VISIBLE_DEVICES to all available GPUs (0,1,2,etc.)
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((GPU_COUNT-1)))
    echo "🎯 CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES}"
else
    echo "⚠️  WARNING: CUDA is not available"
fi

# Print Python environment information
echo "🐍 Python Environment Information:"
python --version
pip list

# Print working directory information
echo "📂 Current working directory: $(pwd)"
echo "📂 Contents of current directory:"
ls -la

# Check if we're running in VSCode environment
if [ -d "/.vscode-server" ]; then
    echo "👨‍💻 VSCode server directory detected"
fi

echo "🚀 Container is ready!"
echo "-----------------------------------"

# Execute the command passed to docker run
echo "🔄 Executing command: $@"
exec "$@"