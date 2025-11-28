# Multi-LLM Code Generation Benchmark

A comprehensive benchmarking suite for evaluating multiple Large Language Models on code generation quality. Measures compilation errors, semantic issues, and security vulnerabilities across multiple datasets.

## 🎯 Overview

This benchmark suite evaluates the following LLMs:
- **StarCoder2** (`bigcode/starcoder2-7b`)
- **DeepSeek-Coder** (`deepseek-ai/deepseek-coder-6.7b-base`)
- **Code LLaMA** (`codellama/CodeLlama-7b-hf`)
- **WizardCoder** (`WizardLM/WizardCoder-15B-V1.0`)

## 📊 Metrics Evaluated

For each model, the benchmark computes:

1. **Compilation Error Rate**: Percentage of generated code that fails to compile
2. **Semantic Error Rate**: Percentage of code with logic/quality issues (via CodeQL)
3. **Security Issue Rate**: Percentage of code with security vulnerabilities (via CodeQL)

## 📁 Project Structure

```
benchmark/
├── README.md                   # This file
├── config_benchmark.json       # Configuration for models and datasets
├── run_benchmark.sh           # Main orchestration script
├── dataset_loader.py          # Dataset loading and standardization
├── multi_model_runner.py      # Multi-model code generation
├── analyze_multi.py           # Batch code analysis
├── compute_metrics.py         # Metrics calculation and visualization
├── dataset_cache/             # Cached datasets
│   ├── all_prompts.json       # Standardized prompts
│   └── [dataset files]        # Downloaded datasets
└── results/                   # Benchmark results
    ├── {model}/               # Per-model results
    │   ├── {dataset}/
    │   │   └── {prompt_id}/
    │   │       ├── generated_code.c
    │   │       ├── clean_code.c
    │   │       ├── metadata.json
    │   │       ├── analysis_results.json
    │   │       └── codeql_results.sarif
    └── summary/               # Aggregated results
        ├── metrics.csv
        ├── detailed_metrics.csv
        ├── benchmark_report.md
        └── visualizations/
```

## 🔧 Setup

### Prerequisites

Ensure you have completed the main project setup:
```bash
cd ..
./prerequisites-setup.sh
```

### Additional Dependencies

The benchmark requires additional Python packages:
```bash
source /scratch/$(whoami)/klee-venv/bin/activate
pip install datasets pandas matplotlib seaborn
```

## 📚 Datasets

The benchmark uses three datasets:

### 1. **xlcost-text-to-code** (HuggingFace)
- Source: `codeparrot/xlcost-text-to-code`
- Language: C/C++
- Access: Automatically downloaded via HuggingFace datasets library

### 2. **APPS_Plus** (GitHub)
- Source: https://raw.githubusercontent.com/Ablustrund/APPS_Plus/refs/heads/main/data/v1/data.json
- Language: Multiple (filtered for C)
- Access: Automatically downloaded and cached

### 3. **QuestionPromptForLLMs** (Google Docs)
- Source: https://docs.google.com/document/d/1Lo6QMD1trXL8OlNPf50zY4rwvUDYu1xzK4sgEj8GrlA/edit
- Language: Multiple
- Access: **Manual download required**
  - Export as JSON
  - Save to: `benchmark/dataset_cache/question_prompts.json`

## 🚀 Usage

### Quick Start (Test Mode)

Run a small test with 5 prompts per dataset using a lightweight model:

```bash
cd benchmark
./run_benchmark.sh test
```

This will:
- Load 5 prompts from each dataset (15 total)
- Generate code using `deepseek-small` (1.3B parameters)
- Analyze all generated code
- Generate metrics and visualizations

**Expected runtime**: ~15-30 minutes (depending on hardware)

### Full Benchmark

Run the complete benchmark across all models and datasets:

```bash
cd benchmark
./run_benchmark.sh full
```

This will:
- Load all available prompts from datasets
- Generate code using all 4 models (StarCoder2, DeepSeek, CodeLlama, WizardCoder)
- Analyze all generated code (~1000+ samples)
- Generate comprehensive metrics and visualizations

**Expected runtime**: Several hours (depends on dataset size and available GPU)

### Custom Configuration

Edit `config_benchmark.json` to customize:

```json
{
  "models": [
    "starcoder2",      # Models to benchmark
    "deepseek",
    "codellama"
  ],
  "datasets": {
    "xlcost": {
      "enabled": true,
      "limit": 50        # Limit prompts per dataset (null = all)
    },
    "apps_plus": {
      "enabled": true,
      "limit": 50
    }
  },
  "generation": {
    "max_new_tokens": 512,
    "temperature": 0.7
  }
}
```

Then run:
```bash
./run_benchmark.sh full
```

## 📈 Output and Results

### Generated Files

After running the benchmark, you'll find:

1. **Individual Results** (`results/{model}/{dataset}/{prompt_id}/`):
   - `generated_code.c`: Raw model output
   - `clean_code.c`: Cleaned C code
   - `metadata.json`: Generation metadata
   - `analysis_results.json`: Compilation and CodeQL results
   - `codeql_results.sarif`: Full CodeQL SARIF output

2. **Summary Files** (`results/summary/`):
   - `metrics.csv`: Overall metrics per model
   - `detailed_metrics.csv`: Metrics broken down by dataset
   - `benchmark_report.md`: Markdown report with findings
   - `{model}_results.json`: Per-model generation results

3. **Visualizations** (`results/summary/visualizations/`):
   - `model_comparison.png`: Bar chart comparing all models
   - `error_heatmap.png`: Heatmap of error rates
   - `{model}_success_rate.png`: Per-model success rate pie charts

### Example Output

```
BENCHMARK SUMMARY
============================================================
Model          Total Samples  Compilation Error %  Semantic Error %  Security Issue %
starcoder2     150           12.3                 34.5              8.7
deepseek       150           8.9                  28.3              6.2
codellama      150           15.1                 31.7              9.4
wizardcoder    150           10.5                 25.8              5.9
============================================================
```

## 🔍 Understanding the Metrics

### Compilation Error %
- **What it measures**: Syntactic correctness
- **How**: Attempts to compile with `gcc -c -Wall -Wextra`
- **Lower is better**: Indicates model generates valid C syntax

### Semantic Error %
- **What it measures**: Code quality and logic issues
- **How**: CodeQL static analysis (quality queries)
- **Lower is better**: Indicates cleaner, more maintainable code

### Security Issue %
- **What it measures**: Potential security vulnerabilities
- **How**: CodeQL security analysis suite
- **Lower is better**: Indicates safer code generation

## 🛠️ Individual Components

### 1. Dataset Loader

Load and test datasets independently:

```bash
python dataset_loader.py
```

### 2. Multi-Model Runner

Generate code with specific models:

```python
from multi_model_runner import MultiModelRunner
import json

# Load prompts
with open('dataset_cache/all_prompts.json') as f:
    prompts = json.load(f)

# Generate with specific model
runner = MultiModelRunner(cache_dir="/scratch/user/hf_cache")
results = runner.generate_with_model(
    model_name='deepseek',
    prompts=prompts[:10],  # First 10 prompts
    max_new_tokens=512
)
```

### 3. Code Analyzer

Analyze existing generated code:

```bash
python analyze_multi.py
```

### 4. Metrics Computer

Compute metrics from analysis results:

```bash
python compute_metrics.py
```

## 🐛 Troubleshooting

### Issue: "CodeQL not found"

**Solution**: Ensure CodeQL is installed:
```bash
ls /scratch/$(whoami)/codeql/codeql
```

If missing, run the main setup:
```bash
cd .. && ./prerequisites-setup.sh
```

### Issue: "CUDA out of memory"

**Solution**:
1. Use smaller models for testing: `deepseek-small`
2. Reduce batch size in config: `"limit": 10`
3. Generate models sequentially (automatic)

### Issue: "Dataset download fails"

**Solution**:
1. Check internet connection
2. For QuestionPromptForLLMs: manually download and place in `dataset_cache/`
3. Run with only available datasets by editing `config_benchmark.json`

### Issue: "Compilation timeout"

**Solution**: Some generated code may have infinite loops. These are marked as compilation failures. Adjust timeout in `config_benchmark.json`:
```json
"analysis": {
  "compilation_timeout": 60
}
```

## 📊 Sample Benchmark Report

A typical benchmark report includes:

```markdown
# Multi-LLM Code Generation Benchmark Report

## Overall Metrics
| Model | Total Samples | Compilation Error % | Semantic Error % | Security Issue % |
|-------|---------------|---------------------|------------------|------------------|
| starcoder2 | 150 | 12.3% | 34.5% | 8.7% |
| deepseek | 150 | 8.9% | 28.3% | 6.2% |

## Key Findings
- **Best Compilation Rate**: deepseek (91.1% success)
- **Fewest Semantic Errors**: wizardcoder (25.8%)
- **Fewest Security Issues**: wizardcoder (5.9%)

## Visualizations
![Model Comparison](visualizations/model_comparison.png)
```

## 🔬 Research Applications

This benchmark is useful for:

1. **Model Selection**: Choose the best model for code generation tasks
2. **Model Comparison**: Quantitatively compare LLM code quality
3. **Training Feedback**: Use analysis results for RLHF training
4. **Security Research**: Study security vulnerability patterns in LLM-generated code
5. **Dataset Evaluation**: Assess dataset quality and difficulty

## 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{llm-code-benchmark,
  title={Multi-LLM Code Generation Quality Benchmark},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/repo}}
}
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more models (GPT-4, Claude, Gemini via APIs)
- [ ] Support more languages (Python, Java, Rust)
- [ ] Add runtime testing (execute generated code)
- [ ] Implement differential testing
- [ ] Add performance benchmarks

## 📄 License

This project follows the license of the parent repository.

## 🔗 Related Work

- Original pipeline: `../README.md`
- CodeQL documentation: https://codeql.github.com/
- KLEE symbolic execution: https://klee.github.io/
- HuggingFace transformers: https://huggingface.co/docs/transformers/
