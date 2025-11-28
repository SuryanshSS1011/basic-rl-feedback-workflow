#!/bin/bash
# Main orchestration script for multi-LLM benchmarking
# Runs: dataset loading → code generation → analysis → metrics computation

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Multi-LLM Code Generation Benchmark Pipeline         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$SCRIPT_DIR"

# Check if running in test mode
TEST_MODE=${1:-"full"}  # "test" or "full"

if [ "$TEST_MODE" = "test" ]; then
    echo "🧪 Running in TEST MODE (limited samples)"
    echo ""
fi

# Activate virtual environment
# Try local environment first, then scratch space
LOCAL_VENV="$PROJECT_ROOT/benchmark_venv"
SCRATCH_VENV="/scratch/$(whoami)/klee-venv"

if [ -d "$LOCAL_VENV" ]; then
    VENV_PATH="$LOCAL_VENV"
    echo "Using local virtual environment: $VENV_PATH"
elif [ -d "$SCRATCH_VENV" ]; then
    VENV_PATH="$SCRATCH_VENV"
    echo "Using scratch virtual environment: $VENV_PATH"
else
    echo "❌ Virtual environment not found!"
    echo "Tried:"
    echo "  - $LOCAL_VENV"
    echo "  - $SCRATCH_VENV"
    echo ""
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv benchmark_venv"
    echo "  source benchmark_venv/bin/activate"
    echo "  pip install -r benchmark/requirements.txt"
    exit 1
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Verify Python dependencies
echo "Verifying dependencies..."
python -c "import torch; import transformers; import datasets; import pandas; import matplotlib; import seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing missing dependencies..."
    pip install -q datasets pandas matplotlib seaborn
fi

echo "✓ Environment ready"
echo ""

# Step 1: Load datasets
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Loading Datasets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$BENCHMARK_DIR"

if [ "$TEST_MODE" = "test" ]; then
    python - <<EOF
import sys
sys.path.append('$BENCHMARK_DIR')
from dataset_loader import DatasetLoader

loader = DatasetLoader()
prompts = loader.load_all(limit_per_dataset=5)
loader.save_prompts(prompts, './dataset_cache/test_prompts.json')
EOF
    PROMPTS_FILE="./dataset_cache/test_prompts.json"
else
    python - <<EOF
import sys
sys.path.append('$BENCHMARK_DIR')
from dataset_loader import DatasetLoader

loader = DatasetLoader()
prompts = loader.load_all(limit_per_dataset=None)
loader.save_prompts(prompts, './dataset_cache/all_prompts.json')
EOF
    PROMPTS_FILE="./dataset_cache/all_prompts.json"
fi

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ Failed to load datasets"
    exit 1
fi

PROMPT_COUNT=$(python -c "import json; data=json.load(open('$PROMPTS_FILE')); print(len(data))")
echo "✓ Loaded $PROMPT_COUNT prompts"
echo ""

# Step 2: Generate code with multiple models
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Generating Code with Multiple LLMs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$TEST_MODE" = "test" ]; then
    MODELS='["deepseek-small"]'
else
    MODELS='["starcoder2", "deepseek", "codellama", "wizardcoder"]'
fi

python - <<EOF
import sys
import json
import os
sys.path.append('$BENCHMARK_DIR')
from multi_model_runner import MultiModelRunner

# Load prompts
with open('$PROMPTS_FILE', 'r') as f:
    prompts = json.load(f)

# Initialize runner - use local cache if scratch not available
scratch_cache = "/scratch/{}/hf_cache".format(os.getlogin())
local_cache = "$PROJECT_ROOT/.hf_cache"
cache_dir = scratch_cache if os.path.exists("/scratch/{}".format(os.getlogin())) else local_cache
os.makedirs(cache_dir, exist_ok=True)
print(f"Using cache directory: {cache_dir}")

runner = MultiModelRunner(cache_dir=cache_dir, results_dir="./results")

# Models to benchmark
models = $MODELS

# Run benchmark
results = runner.run_benchmark(
    models=models,
    prompts=prompts,
    max_new_tokens=512,
    temperature=0.7
)

print(f"\n✓ Code generation complete for {len(models)} models")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Code generation failed"
    exit 1
fi

echo ""

# Step 3: Analyze generated code
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Analyzing Generated Code"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python - <<EOF
import sys
import json
sys.path.append('$BENCHMARK_DIR')
from analyze_multi import CodeAnalyzer

# Get models from results directory
import os
from pathlib import Path

results_dir = Path("./results")
models = [d.name for d in results_dir.iterdir() if d.is_dir() and d.name != 'summary']

print(f"Analyzing code for models: {models}")

analyzer = CodeAnalyzer(results_dir="./results")
results = analyzer.batch_analyze(models)

print(f"\n✓ Analysis complete: {len(results)} samples analyzed")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Analysis failed"
    exit 1
fi

echo ""

# Step 4: Compute metrics and generate report
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Computing Metrics and Generating Report"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python - <<EOF
import sys
sys.path.append('$BENCHMARK_DIR')
from compute_metrics import MetricsComputer

computer = MetricsComputer(results_dir="./results")
computer.run()
EOF

if [ $? -ne 0 ]; then
    echo "❌ Metrics computation failed"
    exit 1
fi

echo ""

# Summary
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Benchmark Complete!                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results Summary:"
echo "  • Generated code: $BENCHMARK_DIR/results/{model}/{dataset}/{prompt_id}/"
echo "  • Metrics CSV: $BENCHMARK_DIR/results/summary/metrics.csv"
echo "  • Detailed metrics: $BENCHMARK_DIR/results/summary/detailed_metrics.csv"
echo "  • Report: $BENCHMARK_DIR/results/summary/benchmark_report.md"
echo "  • Visualizations: $BENCHMARK_DIR/results/summary/visualizations/"
echo ""
echo "To view the report:"
echo "  cat $BENCHMARK_DIR/results/summary/benchmark_report.md"
echo ""
echo "To view metrics:"
echo "  cat $BENCHMARK_DIR/results/summary/metrics.csv"
echo ""

deactivate
