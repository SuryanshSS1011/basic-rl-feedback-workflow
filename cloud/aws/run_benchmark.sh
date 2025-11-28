#!/bin/bash
# Run benchmark on AWS EC2 instance
# Usage: ./run_benchmark.sh [--stage all|generate|analyze|metrics]

set -e

# Default stage
STAGE="${1:-all}"

# Activate environment
source ~/benchmark-workspace/activate_benchmark.sh

# Navigate to repo
cd ~/benchmark-workspace/basic-rl-feedback-workflow

# Check GPU availability
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader
else
    echo "Warning: No GPU detected. Running on CPU."
fi

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./benchmark/results_${TIMESTAMP}"

# Run benchmark
echo ""
echo "Starting benchmark (stage: $STAGE)..."
echo "Results will be saved to: $RESULTS_DIR"
echo ""

python -m benchmark.batch_processor \
    --config ./benchmark/config_benchmark.json \
    --results "$RESULTS_DIR" \
    --stage "$STAGE"

# Compress results for easy download
if [ -d "$RESULTS_DIR" ]; then
    echo ""
    echo "Compressing results..."
    tar -czf "results_${TIMESTAMP}.tar.gz" "$RESULTS_DIR"
    echo "Results compressed to: results_${TIMESTAMP}.tar.gz"
fi

echo ""
echo "Benchmark complete!"
echo ""
