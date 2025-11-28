# Local Setup Guide (macOS/Linux)

This guide is for running the benchmark on your local machine (not on the server with /scratch directory).

## ✅ Setup Complete!

The environment has been set up for you. Here's what was done:

### 1. Virtual Environment Created
```bash
Location: ../benchmark_venv/
Python: 3.13
```

### 2. Dependencies Installed
- ✅ PyTorch 2.9.0
- ✅ Transformers 4.57.1
- ✅ Datasets 4.4.1
- ✅ Pandas, Matplotlib, Seaborn
- ✅ All other requirements

### 3. Configuration Updated
The benchmark script now automatically detects:
- Local virtual environment (`benchmark_venv/`)
- Local cache directory (`.hf_cache/`)

## 🚀 Running the Benchmark

### Quick Test (Recommended First Run)

From the project root:
```bash
cd benchmark
./run_benchmark.sh test
```

This will:
- Load 5 prompts per dataset (~15 total)
- Use `deepseek-coder-1.3b` (small model)
- Take about 15-30 minutes
- Download ~2-3GB for the model

### What to Expect

The test run will:

1. **Load Datasets** (~1-2 min)
   - Downloads xlcost from HuggingFace
   - Downloads APPS_Plus from GitHub
   - QuestionPromptForLLMs (skipped if not manually added)

2. **Generate Code** (~10-20 min)
   - Downloads model (~2GB) on first run
   - Generates 15 code samples
   - Shows progress bar

3. **Analyze Code** (~2-5 min)
   - Checks compilation with gcc
   - ⚠️ **Note**: CodeQL analysis will be skipped if not installed
   - This is fine for testing the workflow

4. **Generate Report** (~1 min)
   - Creates CSV metrics
   - Generates visualizations
   - Writes markdown report

## 📊 Viewing Results

After completion:

```bash
# View metrics
cat results/summary/metrics.csv

# View full report
cat results/summary/benchmark_report.md

# Open visualizations
open results/summary/visualizations/  # macOS
```

## ⚠️ Known Limitations (Local Setup)

### CodeQL Not Available
- **Impact**: Security analysis will be skipped
- **Workaround**:
  - Install CodeQL separately
  - Or run on server with full setup
- **Note**: Compilation analysis still works

### KLEE Not Available
- **Impact**: Symbolic execution is skipped
- **Workaround**: Only available on server setup
- **Note**: This doesn't affect the benchmark metrics

## 🔧 Troubleshooting

### "Model download failed"
- Check internet connection
- Ensure ~10GB free disk space
- Try using a smaller model in config

### "Memory error"
- Close other applications
- Use smaller model: `deepseek-small`
- Reduce batch size in config

### "Command not found: gcc"
- Install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

## 📝 Full Benchmark (Optional)

Once test works, run full benchmark:

```bash
cd benchmark
./run_benchmark.sh full
```

**Warning**: This will:
- Download 4 large models (20-50GB total)
- Take several hours
- Process 100+ prompts

## 🎯 Next Steps

1. ✅ Test run completed? Check results
2. Edit `config_benchmark.json` to customize
3. Add more datasets if desired
4. Run full benchmark when ready

## 📧 Support

If you encounter issues:
1. Check error messages carefully
2. Verify all dependencies installed
3. Try test mode first before full run
4. Check disk space and memory
