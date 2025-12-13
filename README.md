# SecureCodeRL: Reinforcement Learning for Secure Code Generation

A reinforcement learning framework that trains LLMs to generate code that is both **functionally correct** AND **secure**.

## The Problem

LLMs generate code that has two critical issues:
1. **Functional errors** - Code doesn't work correctly (wrong output, crashes)
2. **Security vulnerabilities** - Code contains dangerous patterns (eval, exec, command injection)

Existing approaches optimize for one OR the other, but not both simultaneously.

## Our Solution

**Combined Reward Signal:**
```
R = 0.6 × Rfunc + 0.4 × Rsec
```

Where:
- **Rfunc** = Functional correctness (test pass rate)
- **Rsec** = Security score (absence of vulnerabilities via Bandit analysis)

## Key Innovation: Partial Credit System

Traditional RL for code uses binary rewards (pass/fail), creating a **sparse reward problem** - the model gets no learning signal when tests fail.

We introduced **graduated rewards**:

| Stage | Condition | Score |
|-------|-----------|-------|
| 0 | Syntax error | 0.0 |
| 1 | Valid syntax | 0.2 |
| 2 | Runs without error | 0.4 |
| 3 | Produces output | 0.6 |
| 4 | Tests pass | 1.0 |

This provides continuous learning gradient, allowing the model to improve incrementally.

## Results

| Model | Syntax Valid | Test Pass | Security |
|-------|-------------|-----------|----------|
| SFT Baseline | 45% | 0% | 100% |
| PPO-simple (binary rewards) | 15% | 0% | 100% |
| PPO-fresh (partial credit) | 25% | 0% | 100% |
| **PPO-continue (partial credit)** | **60%** | **5%** | **100%** |

**Key achievements:**
- **+33% syntax improvement** over SFT baseline (45% → 60%)
- **Only model with non-zero test pass rate** (5%)
- **100% security compliance maintained**
- **Continuing from PPO-simple outperforms training fresh**

## Project Phases

### Phase 1: Multi-LLM Benchmark
Evaluated multiple code generation models on APPS+ dataset:
- DeepSeek-Coder (1.3B, 6.7B)
- CodeLlama (7B)
- StarCoder2 (7B)

Found: Even best models have 91%+ semantic error rates and 3-5% security issues.

### Phase 2: Error Analysis
Categorized errors into:
- **Compilation errors** - Syntax issues
- **Semantic errors** - Logic bugs, wrong output
- **Security vulnerabilities** - Dangerous patterns

Discovered the **"missing print" bug**: Models generate `(n*2)` instead of `print(n*2)`.

### Phase 3: SFT Training
- **Model:** DeepSeek-Coder-1.3B-Instruct
- **Method:** LoRA (r=16, alpha=32)
- **Data:** APPS+ dataset (3,588 stdin-style problems)
- **Params:** 6.3M trainable / 1.35B total (0.47%)

### Phase 4: PPO Training
- **Algorithm:** Proximal Policy Optimization
- **Reward:** R = 0.6×Rfunc + 0.4×Rsec
- **Security:** Bandit static analysis
- **Episodes:** 100-500

### Phase 5: Partial Credit Fix
Implemented graduated rewards to solve sparse reward problem. Result: Model now learns incrementally.

## Architecture

| Component | Tool | Purpose |
|-----------|------|---------|
| Security Analysis | Bandit | Python static security scanner |
| Functional Testing | Test Execution | stdin/stdout test cases |
| Syntax Checking | AST Parse | Python syntax validation |
| Training | PPO + LoRA | Parameter-efficient RL |

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers peft datasets accelerate bandit
```

### 2. Run SFT Training
```bash
python train_sft_stdin.py
```

### 3. Run PPO Training
```bash
python train_ppo.py \
    --sft_checkpoint ./checkpoints/sft_stdin/best \
    --prompts_file ./data/prompts/ppo_prompts.json \
    --use_bandit \
    --episodes 100
```

### 4. Evaluate Model
```bash
python evaluate_dual_metrics.py \
    --checkpoints ./checkpoints/sft_stdin/best \
    --prompts_file ./data/prompts/ppo_prompts.json \
    --num_samples 50
```

## Project Structure

```
SecureCodeRL/
├── train_sft_stdin.py        # SFT training script
├── train_ppo.py              # PPO training script
├── evaluate_dual_metrics.py  # Model evaluation
├── prepare_ppo_data.py       # Data preparation
├── run_both_experiments.sh   # Full experiment pipeline
│
├── rl_training/              # Core RL modules
│   ├── ppo_trainer.py        # PPO with partial credit
│   ├── reward_calculator.py  # R = α×Rfunc + β×Rsec
│   ├── bandit_runner.py      # Security analysis
│   ├── scoring_agent.py      # Code evaluation
│   └── sft_trainer.py        # SFT training
│
├── data/
│   └── prompts/              # Training prompts
│
├── paper/                    # LaTeX paper
│   ├── main.tex              # IEEE format paper
│   └── figures/              # TikZ figures
│
├── benchmark/                # Multi-LLM benchmark system
│
├── reports/
│   └── paper_draft.md        # Research paper draft
│
└── results_from_vm/          # Experiment results
```

## Security Patterns Detected

The pipeline detects these Python vulnerabilities:
- `eval()`, `exec()` - Code injection
- `os.system()`, `subprocess(shell=True)` - Command injection
- `pickle.load()` - Unsafe deserialization
- `__import__()` - Dynamic imports
- Hardcoded credentials

## Training Configuration

### SFT Config
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

### PPO Config
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-6 |
| Batch size | 2 |
| PPO epochs | 4 |
| Clip range | 0.2 |
| α (functional) | 0.6 |
| β (security) | 0.4 |

## Hardware Requirements

- **GPU:** NVIDIA V100 16GB+ (or equivalent)
- **CUDA:** 11.8+
- **RAM:** 32GB+ recommended

## Documentation

- **Paper Draft:** `reports/paper_draft.md` - Research paper
- **LaTeX Paper:** `paper/main.tex` - Final paper (IEEE format)

