# Benchmark Quick Start Guide

Get started with the multi-LLM benchmarking suite in 5 minutes.

## Step 1: Setup (One-time)

Ensure the main environment is set up:
```bash
cd /Users/suryanshss/CMPSC461/Honors\ Work/basic-rl-feedback-workflow
./prerequisites-setup.sh  # If not already done
```

Install benchmark dependencies:
```bash
source /scratch/$(whoami)/klee-venv/bin/activate
pip install datasets pandas matplotlib seaborn tqdm
```

## Step 2: Test Run

Run a quick test with 5 prompts:

```bash
cd benchmark
./run_benchmark.sh test
```

This uses a small model (`deepseek-1.3b`) and limited prompts (~15 total).

**Expected runtime**: 15-30 minutes

## Step 3: View Results

After completion:

```bash
# View metrics table
cat results/summary/metrics.csv

# View full report
cat results/summary/benchmark_report.md

# View visualizations
ls results/summary/visualizations/
```

## Step 4: Full Benchmark (Optional)

Run the complete benchmark with all models:

```bash
./run_benchmark.sh full
```

**Warning**: This will:
- Download 4 large models (7B-15B parameters each)
- Take several hours to complete
- Use significant GPU memory and disk space

## Customization

Edit `config_benchmark.json` to:
- Select specific models to benchmark
- Limit number of prompts per dataset
- Adjust generation parameters

Example:
```json
{
  "models": ["deepseek", "codellama"],
  "datasets": {
    "xlcost": {
      "enabled": true,
      "limit": 20
    }
  }
}
```

## Common Commands

```bash
# Test mode (fast, small model)
./run_benchmark.sh test

# Full benchmark (all models)
./run_benchmark.sh full

# Load datasets only
python dataset_loader.py

# Analyze existing results
python analyze_multi.py

# Recompute metrics
python compute_metrics.py
```

## Output Files

Key files to check:
- `results/summary/metrics.csv` - Overall metrics table
- `results/summary/benchmark_report.md` - Full report
- `results/summary/visualizations/model_comparison.png` - Comparison chart

## Troubleshooting

**"No prompts loaded"**
- Check internet connection for dataset downloads
- For QuestionPromptForLLMs: manually download from Google Docs

**"CUDA out of memory"**
- Use test mode with smaller model
- Reduce dataset limits in config

**"CodeQL not found"**
- Run main setup: `cd .. && ./prerequisites-setup.sh`

## Next Steps

See full documentation: `README.md`
