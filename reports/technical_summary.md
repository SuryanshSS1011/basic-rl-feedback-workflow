# Technical Summary: SecureCodeRL Benchmark & Training Pipeline

## Executive Summary

This document provides a technical overview of the SecureCodeRL project, which combines a security-aware code generation benchmark with an RL-based training pipeline. The analysis evaluates three LLMs (DeepSeek, CodeLlama, StarCoder2) on 7,413 APPS+ problems, revealing high error rates (91-99% semantic errors) that highlight the challenging nature of competitive programming code generation.

---

## 1. Project Overview

### 1.1 Objectives
1. Construct a security-aware code generation benchmark
2. Evaluate baseline model performance with detailed metrics
3. Develop RL training pipeline for improvement
4. Generate comprehensive analysis and reports

### 1.2 Repository Structure

```
basic-rl-feedback-workflow/
├── benchmark/                    # Benchmark system
│   └── full_results/            # Analysis results (22K samples)
│       ├── deepseek/            # DeepSeek outputs
│       ├── codellama/           # CodeLlama outputs
│       ├── starcoder2/          # StarCoder2 outputs
│       └── prompts.json         # Test cases (17MB)
├── rl_training/                  # RL training modules
│   ├── config.py                # Configuration classes
│   ├── data_converter.py        # Data preparation
│   ├── reward_calculator.py     # Reward computation
│   ├── sft_trainer.py           # SFT implementation
│   └── ppo_trainer.py           # PPO implementation
├── data/                         # Processed data
│   ├── sft/                     # SFT training data
│   │   ├── train.jsonl          # 3,588 samples
│   │   └── val.jsonl            # 398 samples
│   ├── prompts/                 # PPO prompts
│   └── *_subset_ids.json        # Dataset splits
├── reports/                      # Generated reports
├── filter_stdin_dataset.py       # Dataset filtering
├── compute_filtered_metrics.py   # Metrics computation
├── prepare_training_data.py      # Training data prep
├── train_sft_stdin.py           # SFT training script
└── train_ppo.py                 # PPO training script
```

---

## 2. Dataset Analysis

### 2.1 APPS+ Overview

| Metric | Value |
|--------|-------|
| Total problems | 7,413 |
| Total samples (3 models) | 22,224 |
| Stdin-style problems | 4,779 (64.5%) |
| Function-call problems | 2,634 (35.5%) |

### 2.2 Input Format Distribution

**Stdin-style** (64.5%): Test inputs are strings like `"3\n5 0 -5\n"` passed via stdin
**Function-call** (35.5%): Test inputs are Python objects like `[[1, 2, 3], "abc"]`

Primary analysis uses stdin-style subset for methodological clarity.

---

## 3. Model Evaluation Results

### 3.1 Stdin Subset Performance (Primary Results)

| Model | Total | Compile% | Output% | Test Pass% | Security% |
|-------|-------|----------|---------|------------|-----------|
| DeepSeek | 4,779 | 83.4 | 9.0 | 14.3 | 3.72 |
| CodeLlama | 4,779 | 48.9 | 3.7 | 7.6 | 0.46 |
| StarCoder2 | 4,778 | 20.2 | 0.6 | 1.3 | 0.31 |

### 3.2 Full Dataset Performance

| Model | Total | Compile% | Semantic Error% | Security% |
|-------|-------|----------|-----------------|-----------|
| DeepSeek | 7,408 | 80.1 | 93.9 | 3.40 |
| CodeLlama | 7,408 | 50.2 | 97.6 | 0.65 |
| StarCoder2 | 7,408 | 25.4 | 99.5 | 0.49 |

### 3.3 Detailed DeepSeek Analysis (Best Model)

```
Stdin Subset Metrics:
  Compiling samples:     3,986
  Semantic errors:       3,629 (91.0%)
  Produces output:         357 (9.0%)
  Tests passed total:    1,399
  Tests total:           9,790
  Test pass rate:        14.3%

Security:
  Issues found:            178
  Issue rate:            3.72%
  Low severity:            177
  Medium severity:           1
```

---

## 4. Training Data Preparation

### 4.1 Data Pipeline

```
benchmark/full_results/
    └── Load prompts.json (7,413 problems)
        └── Filter stdin-style (4,779)
            └── Load analysis.json per sample
                └── Calculate rewards
                    └── Split train/val (90/10)
                        └── Save to data/sft/
```

### 4.2 Training Data Statistics

| Metric | Value |
|--------|-------|
| Total compiling samples | 3,986 |
| Samples with tests passed | 775 |
| Training samples | 3,588 |
| Validation samples | 398 |
| Unique prompts | 3,986 |

### 4.3 Reward Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 0.0-0.2 | 1 | 0.0% |
| 0.2-0.4 | 173 | 4.3% |
| 0.4-0.6 | 3,038 | 76.2% |
| 0.6-0.8 | 284 | 7.1% |
| 0.8-1.0 | 136 | 3.4% |

### 4.4 Reward Statistics

```
Total reward: min=0.160, max=1.000, mean=0.477
R_func:       min=0.000, max=1.000, mean=0.137
R_sec:        min=0.400, max=1.000, mean=0.987
```

---

## 5. RL Training Configuration

### 5.1 Reward Function

```python
R = α × R_func + β × R_sec

α = 0.6  # Functional correctness weight
β = 0.4  # Security weight

R_func = tests_passed / tests_total
R_sec = 1 - normalized_security_penalty

Security penalty weights:
  Low:    0.3
  Medium: 0.6
  High:   1.0
```

### 5.2 SFT Configuration

```yaml
Model: deepseek-ai/deepseek-coder-1.3b-instruct
LoRA:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
Training:
  epochs: 3
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-5
  max_seq_length: 2048
```

### 5.3 PPO Configuration

```yaml
Learning rate: 1e-5
Batch size: 4
Mini-batch size: 2
PPO epochs: 4
Clip range: 0.2
KL penalty: 0.1
Target KL: 0.01
Episodes: 1000
```

---

## 6. Implementation Details

### 6.1 Key Scripts

| Script | Purpose |
|--------|---------|
| `filter_stdin_dataset.py` | Filter dataset to stdin-style subset |
| `compute_filtered_metrics.py` | Compute metrics on filtered data |
| `prepare_training_data.py` | Prepare SFT training data |
| `train_sft_stdin.py` | Run SFT training |
| `train_ppo.py` | Run PPO training |

### 6.2 Running the Pipeline

```bash
# 1. Filter dataset
python filter_stdin_dataset.py

# 2. Compute metrics
python compute_filtered_metrics.py

# 3. Prepare training data
python prepare_training_data.py

# 4. Run SFT training (requires GPU)
python train_sft_stdin.py --epochs 3 --batch_size 4

# 5. Run PPO training (requires SFT checkpoint)
python train_ppo.py --sft_checkpoint ./checkpoints/sft_stdin/best
```

### 6.3 Hardware Requirements

| Phase | Memory | Compute | Time Est. |
|-------|--------|---------|-----------|
| Data prep | 8GB RAM | CPU | 5 min |
| SFT | 16GB VRAM | GPU | 2-4 hours |
| PPO | 24GB VRAM | GPU | 8-16 hours |

---

## 7. Key Findings

### 7.1 Model Performance Insights

1. **DeepSeek outperforms** CodeLlama and StarCoder2 across all metrics
2. **Compilation ≠ Correctness**: 83% compile rate but only 14% test pass rate
3. **High semantic error rates** (91-99%) indicate fundamental code generation challenges
4. **Security issues inversely correlate** with model capability

### 7.2 Code Generation Patterns

Analysis of generated code reveals:
- **92% have empty function bodies** - LLMs put solution at module level
- **47% crash at runtime** - ValueError, SyntaxError during execution
- **26% produce output** - Module-level code runs correctly
- **Entry point mismatch** - Prompts define functions, but code doesn't fill them

### 7.3 Research Implications

- Current LLMs struggle with competitive programming tasks
- Security-aware evaluation reveals additional failure modes
- RL with tool feedback provides principled improvement approach
- Methodological transparency is crucial for reproducible research

---

## 8. Files Generated

### 8.1 Data Files

```
data/
├── stdin_subset_ids.json        # 4,779 stdin-style IDs
├── function_call_subset_ids.json # 2,634 function-call IDs
├── full_dataset_ids.json        # All 7,413 IDs
├── sft/
│   ├── train.jsonl             # Training data
│   └── val.jsonl               # Validation data
└── prompts/
    ├── ppo_prompts.txt         # PPO prompts (text)
    └── ppo_prompts.json        # PPO prompts (JSON)
```

### 8.2 Analysis Files

```
analysis_results.json           # Full dataset metrics
analysis_results_stdin.json     # Stdin subset metrics
```

### 8.3 Report Files

```
reports/
├── research_paper.md           # Full research paper
└── technical_summary.md        # This document
```

---

## 9. Next Steps

### 9.1 Immediate Actions

1. Run SFT training on GPU-enabled machine
2. Run PPO training with tool feedback
3. Evaluate trained model on held-out test set
4. Generate training curves and visualizations

### 9.2 Future Work

1. Scale to larger models (7B, 13B)
2. Add runtime security analysis
3. Implement multi-turn generation
4. Extend to other programming languages

---

## Appendix: Quick Reference

### Command Reference

```bash
# Dataset operations
python filter_stdin_dataset.py          # Filter to stdin subset
python compute_filtered_metrics.py      # Compute metrics
python prepare_training_data.py         # Prepare training data

# Training
python train_sft_stdin.py --dry_run     # Verify setup
python train_sft_stdin.py               # Run SFT
python train_ppo.py --sft_checkpoint ./checkpoints/sft_stdin/best

# Analysis
python run_analysis.py --help           # Run benchmark analysis
```

### Key Metrics Summary

| Metric | DeepSeek | CodeLlama | StarCoder2 |
|--------|----------|-----------|------------|
| Compile% | 83.4 | 48.9 | 20.2 |
| Output% | 9.0 | 3.7 | 0.6 |
| Test Pass% | 14.3 | 7.6 | 1.3 |
| Security% | 3.72 | 0.46 | 0.31 |
