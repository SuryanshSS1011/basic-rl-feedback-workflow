#!/bin/bash
# AWS EC2 Setup Script for Multi-LLM Benchmark
# Recommended instances:
# - p3.2xlarge (V100 16GB) - Best for full 7B-15B models
# - g4dn.xlarge (T4 16GB) - Cost-effective option
# - g5.xlarge (A10G 24GB) - Good price/performance

set -e

echo "=========================================="
echo "AWS EC2 Benchmark Environment Setup"
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
$SUDO apt-get upgrade -y

# Install NVIDIA drivers if not present
echo "[2/8] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    $SUDO apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit
fi

# Install Python 3.11 if not present
echo "[3/8] Setting up Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
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

# Clone repository (if not exists)
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

# Setup HuggingFace cache
echo "[8/8] Configuring environment..."
HF_CACHE="/opt/hf_cache"
$SUDO mkdir -p "$HF_CACHE"
$SUDO chmod 777 "$HF_CACHE"
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
cat > "$WORK_DIR/activate_benchmark.sh" << 'EOF'
#!/bin/bash
export HF_HOME="/opt/hf_cache"
export PATH="$PATH:/opt/codeql"
source ~/benchmark-workspace/benchmark_venv/bin/activate
cd ~/benchmark-workspace/basic-rl-feedback-workflow
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
