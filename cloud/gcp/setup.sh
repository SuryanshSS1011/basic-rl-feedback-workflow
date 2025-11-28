#!/bin/bash
# GCP VM Setup Script for Multi-LLM Benchmark
# Recommended instances:
# - n1-standard-8 + NVIDIA T4 (16GB) - Cost-effective
# - a2-highgpu-1g (A100 40GB) - Best performance
# - n1-standard-16 + NVIDIA V100 (16GB) - Good balance

set -e

echo "=========================================="
echo "GCP VM Benchmark Environment Setup"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Update system
echo "[1/8] Updating system..."
$SUDO apt-get update -y

# Install NVIDIA drivers (GCP specific)
echo "[2/8] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers for GCP..."
    # GCP Deep Learning VM images come with drivers pre-installed
    # For other images, install manually:
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb
    $SUDO dpkg -i cuda-keyring_1.0-1_all.deb
    $SUDO apt-get update
    $SUDO apt-get install -y cuda-toolkit-12-1
fi

# Install Python 3.11
echo "[3/8] Setting up Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    $SUDO apt-get install -y software-properties-common
    $SUDO add-apt-repository ppa:deadsnakes/ppa -y
    $SUDO apt-get update
    $SUDO apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Install system dependencies
echo "[4/8] Installing system dependencies..."
$SUDO apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    cmake \
    curl \
    wget \
    unzip

# Create working directory
WORK_DIR="${HOME}/benchmark-workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Setup repository
echo "[5/8] Setting up repository..."
REPO_DIR="$WORK_DIR/basic-rl-feedback-workflow"
if [ ! -d "$REPO_DIR" ]; then
    echo "Please clone your repository to $REPO_DIR"
    echo "git clone <your-repo-url> $REPO_DIR"
fi

# Create virtual environment
echo "[6/8] Creating virtual environment..."
VENV_DIR="$WORK_DIR/benchmark_venv"
if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install PyTorch with CUDA support
echo "[7/8] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install benchmark dependencies
pip install transformers accelerate datasets tqdm pandas matplotlib seaborn requests

# Install security analysis tools
pip install bandit

# Setup HuggingFace cache (use SSD for better performance)
echo "[8/8] Configuring environment..."
HF_CACHE="/mnt/disks/hf_cache"
if [ ! -d "$HF_CACHE" ]; then
    # Fall back to home directory
    HF_CACHE="$HOME/.cache/huggingface"
fi
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"

# Install CodeQL
CODEQL_DIR="/opt/codeql"
if [ ! -d "$CODEQL_DIR" ]; then
    echo "Installing CodeQL..."
    cd /tmp
    wget -q https://github.com/github/codeql-action/releases/download/codeql-bundle-v2.23.2/codeql-bundle-linux64.tar.gz
    $SUDO tar -xzf codeql-bundle-linux64.tar.gz -C /opt
    rm codeql-bundle-linux64.tar.gz
fi
export PATH="$PATH:/opt/codeql"

# Create activation script
cat > "$WORK_DIR/activate_benchmark.sh" << EOF
#!/bin/bash
export HF_HOME="$HF_CACHE"
export PATH="\$PATH:/opt/codeql"
source $VENV_DIR/bin/activate
cd $REPO_DIR
echo "Benchmark environment activated!"
echo "Run: python -m benchmark.batch_processor --help"
EOF
chmod +x "$WORK_DIR/activate_benchmark.sh"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source $WORK_DIR/activate_benchmark.sh"
echo ""
echo "To start the benchmark:"
echo "  python -m benchmark.batch_processor"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""
