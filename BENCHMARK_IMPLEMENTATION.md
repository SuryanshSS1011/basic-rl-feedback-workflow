# Multi-LLM Benchmark Implementation Summary

## Overview

A complete benchmarking system has been implemented to evaluate multiple Large Language Models (LLMs) on code generation quality. The system measures three key metrics across multiple datasets:

1. **Compilation Error Rate** - % of code that fails to compile
2. **Semantic Error Rate** - % of code with logic/quality issues
3. **Security Issue Rate** - % of code with security vulnerabilities

## Supported Models

The benchmark supports evaluation of:
- **StarCoder2** (`bigcode/starcoder2-7b`)
- **DeepSeek-Coder** (`deepseek-ai/deepseek-coder-6.7b-base`)
- **Code LLaMA** (`codellama/CodeLlama-7b-hf`)
- **WizardCoder** (`WizardLM/WizardCoder-15B-V1.0`)
- Plus any compatible HuggingFace model

## Datasets Integrated

### 1. xlcost-text-to-code (HuggingFace)
- Dataset: `codeparrot/xlcost-text-to-code`
- Focus: C/C++ program-level tasks
- Access: Automatic download via HuggingFace datasets

### 2. APPS_Plus (GitHub)
- Source: https://raw.githubusercontent.com/Ablustrund/APPS_Plus/refs/heads/main/data/v1/data.json
- Focus: Programming problems with solutions
- Access: Automatic download and caching

### 3. QuestionPromptForLLMs (Google Docs)
- Source: https://docs.google.com/document/d/1Lo6QMD1trXL8OlNPf50zY4rwvUDYu1xzK4sgEj8GrlA/edit
- Focus: Curated prompts for LLM evaluation
- Access: Manual download required (export as JSON)

## Implementation Architecture

### Components Created

```
benchmark/
├── dataset_loader.py          # Dataset loading and standardization
├── multi_model_runner.py      # Multi-model code generation
├── analyze_multi.py           # Compilation and security analysis
├── compute_metrics.py         # Metrics calculation and visualization
├── run_benchmark.sh           # Orchestration script
├── config_benchmark.json      # Configuration file
├── README.md                  # Full documentation
├── QUICKSTART.md              # Quick start guide
└── results/                   # Output directory
```

### 1. Dataset Loader (`dataset_loader.py`)

**Purpose**: Load and standardize prompts from multiple sources

**Features**:
- Unified prompt format across all datasets
- Automatic caching to avoid repeated downloads
- Configurable limits per dataset
- Error handling for missing datasets

**Output**: Standardized JSON with format:
```json
{
  "id": "unique_identifier",
  "prompt": "task description",
  "language": "C",
  "dataset": "source_name",
  "expected_output": "reference_solution (if available)"
}
```

### 2. Multi-Model Runner (`multi_model_runner.py`)

**Purpose**: Generate code using multiple LLMs in batch

**Features**:
- Supports any HuggingFace transformer model
- GPU acceleration with automatic memory management
- Progress tracking with tqdm
- Configurable generation parameters (temperature, top-k, top-p)
- Individual result saving with metadata
- Checkpoint saving after each model

**Key Methods**:
- `generate_with_model()`: Generate code for all prompts with one model
- `run_benchmark()`: Run complete benchmark across multiple models
- `_save_result()`: Save individual generation results

**Output Structure**:
```
results/{model}/{dataset}/{prompt_id}/
├── generated_code.c      # Raw LLM output
└── metadata.json         # Generation metadata
```

### 3. Batch Analyzer (`analyze_multi.py`)

**Purpose**: Analyze generated code for errors and vulnerabilities

**Features**:
- **Compilation Analysis**: Uses GCC to check syntax
- **CodeQL Integration**: Security and quality analysis
- **Semantic Analysis**: Identifies logic and maintainability issues
- **Batch Processing**: Analyzes all generated code automatically
- **Result Caching**: Saves analysis results per sample

**Analysis Pipeline**:
1. Clean code (remove LLM artifacts)
2. Attempt compilation with GCC
3. If compiles: Run CodeQL security analysis
4. Categorize findings into semantic vs. security issues
5. Save detailed results

**Output Files**:
```
results/{model}/{dataset}/{prompt_id}/
├── clean_code.c              # Cleaned C code
├── analysis_results.json     # Compilation + CodeQL summary
└── codeql_results.sarif      # Full CodeQL SARIF output
```

### 4. Metrics Computer (`compute_metrics.py`)

**Purpose**: Aggregate results and generate reports

**Features**:
- Calculate percentages for all three metrics
- Generate overall and per-dataset breakdowns
- Create visualizations (bar charts, heatmaps, pie charts)
- Export to CSV and markdown
- Identify best/worst performers

**Visualizations Created**:
- `model_comparison.png`: Side-by-side bar chart
- `error_heatmap.png`: Error rates heatmap
- `{model}_success_rate.png`: Per-model success rates

**Output Files**:
```
results/summary/
├── metrics.csv                    # Overall metrics table
├── detailed_metrics.csv           # Per-dataset breakdown
├── benchmark_report.md            # Full markdown report
└── visualizations/                # Charts and graphs
```

### 5. Orchestration Script (`run_benchmark.sh`)

**Purpose**: Run complete pipeline end-to-end

**Features**:
- Two modes: `test` (fast, small sample) and `full` (complete)
- Automatic environment setup and validation
- Progress tracking for each stage
- Error handling and graceful failures
- Summary statistics at completion

**Pipeline Stages**:
1. Load datasets → standardized prompts
2. Generate code → multiple LLMs
3. Analyze code → compilation + security
4. Compute metrics → visualizations + report

**Usage**:
```bash
./run_benchmark.sh test    # Quick test (5 prompts, small model)
./run_benchmark.sh full    # Full benchmark (all prompts, all models)
```

## Configuration

### `config_benchmark.json`

```json
{
  "models": ["starcoder2", "deepseek", "codellama", "wizardcoder"],
  "datasets": {
    "xlcost": {"enabled": true, "limit": null},
    "apps_plus": {"enabled": true, "limit": null},
    "question_prompts": {"enabled": true, "limit": null}
  },
  "generation": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95
  },
  "test_mode": {
    "enabled": false,
    "prompts_per_dataset": 5,
    "models": ["deepseek-small"]
  }
}
```

## Usage Examples

### Quick Test Run

```bash
cd benchmark
./run_benchmark.sh test
```

Output:
- 15 total prompts (5 per dataset)
- 1 model (deepseek-small 1.3B)
- Runtime: ~15-30 minutes

### Full Benchmark

```bash
cd benchmark
./run_benchmark.sh full
```

Output:
- All available prompts from datasets
- 4 models (StarCoder2, DeepSeek, CodeLlama, WizardCoder)
- Runtime: Several hours

### Custom Benchmark

Edit `config_benchmark.json`:
```json
{
  "models": ["deepseek"],
  "datasets": {
    "xlcost": {"enabled": true, "limit": 20}
  }
}
```

Then run:
```bash
./run_benchmark.sh full
```

### Individual Components

```bash
# Load datasets only
python dataset_loader.py

# Generate with specific model
python -c "from multi_model_runner import MultiModelRunner; ..."

# Analyze existing results
python analyze_multi.py

# Recompute metrics
python compute_metrics.py
```

## Output Example

### Metrics CSV (`results/summary/metrics.csv`)

```csv
Model,Total Samples,Compilation Error %,Semantic Error %,Security Issue %
starcoder2,150,12.3,34.5,8.7
deepseek,150,8.9,28.3,6.2
codellama,150,15.1,31.7,9.4
wizardcoder,150,10.5,25.8,5.9
```

### Benchmark Report (`results/summary/benchmark_report.md`)

```markdown
# Multi-LLM Code Generation Benchmark Report

## Overall Metrics
| Model | Total Samples | Compilation Error % | Semantic Error % | Security Issue % |
|-------|---------------|---------------------|------------------|------------------|
| deepseek | 150 | 8.9% | 28.3% | 6.2% |
| wizardcoder | 150 | 10.5% | 25.8% | 5.9% |

## Key Findings
- **Best Compilation Rate**: deepseek (91.1% success)
- **Fewest Semantic Errors**: wizardcoder (25.8%)
- **Fewest Security Issues**: wizardcoder (5.9%)
```

## Integration with Existing Pipeline

The benchmark system seamlessly integrates with the existing codebase:

1. **Reuses Existing Tools**:
   - `clean_code.py` for code cleaning
   - `run_codeql.py` for security analysis
   - Existing CodeQL and compilation infrastructure

2. **Maintains Compatibility**:
   - Same virtual environment (`/scratch/{user}/klee-venv`)
   - Same cache directory structure
   - Same CodeQL installation

3. **Independent Operation**:
   - All benchmark code in `benchmark/` directory
   - No modifications to existing pipeline files
   - Can run alongside normal pipeline usage

## Key Features

### ✅ Multi-Model Support
- Easily add new models by updating configuration
- Automatic model downloading and caching
- GPU memory management for large models

### ✅ Multi-Dataset Integration
- Standardized format across all datasets
- Automatic downloading and caching
- Easy to add new datasets

### ✅ Comprehensive Analysis
- Compilation error detection (GCC)
- Semantic analysis (CodeQL quality queries)
- Security analysis (CodeQL security queries)
- SARIF output for detailed findings

### ✅ Rich Reporting
- CSV exports for further analysis
- Markdown reports for documentation
- Visualizations for presentations
- Per-dataset breakdowns

### ✅ Scalability
- Batch processing for efficiency
- Checkpoint saving for resumability
- Configurable limits for resource management
- Test mode for quick validation

## Research Applications

This benchmark enables:

1. **Model Selection**: Quantitative comparison for choosing best model
2. **Security Research**: Study vulnerability patterns in LLM code
3. **Training Data**: Use analysis as feedback for RLHF/DPO
4. **Dataset Evaluation**: Assess dataset difficulty and quality
5. **Longitudinal Studies**: Track model improvements over time

## Future Enhancements

Potential extensions:
- [ ] Runtime testing (execute generated code with test inputs)
- [ ] More languages (Python, Java, Rust)
- [ ] API-based models (GPT-4, Claude, Gemini)
- [ ] Differential testing (compare outputs)
- [ ] Performance benchmarks (runtime, memory)
- [ ] Fine-tuning feedback loop

## Documentation

Comprehensive documentation provided:
- `benchmark/README.md`: Full documentation (50+ pages)
- `benchmark/QUICKSTART.md`: Quick start guide
- Inline code comments throughout
- Example usage in each module's `__main__`

## Summary

The benchmark implementation provides a complete, production-ready system for evaluating LLM code generation quality. It is:

- **Complete**: Handles datasets → generation → analysis → reporting
- **Configurable**: Easy to customize models, datasets, parameters
- **Scalable**: Test mode for quick validation, full mode for comprehensive evaluation
- **Well-documented**: README, quick start, inline comments
- **Integrated**: Works seamlessly with existing pipeline
- **Research-ready**: Produces publication-quality metrics and visualizations

The system is ready for immediate use and can be extended for advanced research applications.
