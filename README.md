# SecureCodeRL

**Scheduled Partial-Credit RL for Reliable Code Generation with Small Language Models**

LCTES '26 (WIP) — [DOI: 10.1145/3814943.3816167](https://doi.org/10.1145/3814943.3816167)

This repository contains the source code, training pipeline, and evaluation
scripts that accompany the LCTES '26 WIP paper. The camera-ready PDF is at
[`paper/lctes26-camera-ready.pdf`](paper/lctes26-camera-ready.pdf)
and is the source of truth for every claim in this README. The system name
`SecureCodeRL` matches the prior preprint (arXiv:2501.01184); the LCTES paper
reframes the headline metric from security to reliability given the null
Bandit findings on APPS+.

## Abstract

Small language models (≤1.5B parameters) are attractive for embedded and
resource-limited code generation because they can run under single-GPU or CPU
budgets and be adapted without distributed training, but they are brittle under
strict sandboxed evaluation, and reinforcement learning with binary test
rewards is too sparse to train them reliably. This WIP paper presents a
reliability-first RL framework with a partial-credit functional reward that
distinguishes common near-miss outcomes (syntax error, crash, missing output,
partial test success) and assigns intermediate credit for each. A
static-analysis term in the objective discourages unsafe shortcuts during
training. On DeepSeek-Coder-1.3B evaluated on 100 stdin-style APPS+ prompts,
partial-credit PPO improves syntax validity to 63% and produces solutions that
pass at least one test in 9% of prompts, while binary-reward PPO regresses
below the supervised fine-tuning baseline. A binary-to-partial-credit
curriculum outperforms training with partial credit from scratch.

## Citation

If you use this work, cite the paper:

```bibtex
@inproceedings{sijwali2026partialcredit,
  title     = {Scheduled Partial-Credit RL for Reliable Code Generation with Small Language Models (WIP)},
  author    = {Sijwali, Suryansh Singh and Saha, Suman},
  booktitle = {Proceedings of the 27th ACM SIGPLAN/SIGBED International Conference on Languages, Compilers, and Tools for Embedded Systems (LCTES '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Boulder, CO, USA},
  pages     = {1--5},
  doi       = {10.1145/3814943.3816167},
  url       = {https://dl.acm.org/doi/10.1145/3814943.3816167},
  isbn      = {979-8-4007-2721-4},
  note      = {Work in progress.}
}
```

To cite this software artifact specifically (Zenodo deposit):

```bibtex
@software{sijwali2026securecoderl,
  title     = {SecureCodeRL: Scheduled Partial-Credit RL for Reliable Code Generation with Small Language Models (LCTES '26 artifact)},
  author    = {Sijwali, Suryansh Singh and Saha, Suman},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.1-lctes-final},
  doi       = {10.5281/zenodo.19999641},
  url       = {https://doi.org/10.5281/zenodo.19999641}
}
```

## Method

The policy `π_θ` generates a program `y` for prompt `x` and is trained with
PPO against a scalar reward

```
R(y) = α · R_func(y) + β · R_sec(y),   α = 0.6, β = 0.4
```

### Partial-credit functional reward (paper Table 1)

| Stage | Condition                       | R_func           |
|-------|---------------------------------|------------------|
| 0     | Syntax error / not parseable    | 0.0              |
| 1     | Valid syntax                    | 0.2              |
| 2     | Executes without crash          | 0.4              |
| 3     | Produces any stdout             | 0.6              |
| 4     | Passes k of T tests             | 0.6 + 0.4 · k/T  |

T is the total number of test cases for the prompt; k is the number passed.

### Security term

```
R_sec = 1 − V,   V ∈ [0, 1]
```

V is a normalized severity score over Bandit findings. We run Bandit at the
`-ll` (medium-and-up) level, so LOW findings are excluded; benign `input()`
usage required by the APPS+ stdin format would otherwise dominate. HIGH
findings contribute 1.0 to V and MEDIUM findings 0.5. Clean code yields
`R_sec = 1.0`. The linear form keeps `R_sec` in `[0, 1]` so it shares a range
with `R_func` (the earlier `exp(-V)` form was bounded below at `e^-1 ≈ 0.37`,
which a reviewer flagged as range-mismatched). The empirical impact on the
reported numbers is nil since `V = 0` across all 400 evaluation samples.

### KL regularization

PPO uses a KL penalty `λ = 0.1` against the SFT reference to discourage reward
hacking.

## Results

### Train/eval split

Training draws prompts from the 7,408-problem stdin-style APPS+ corpus. PPO
runs for 500 episodes at batch 2, covering about 1,000 unique prompts. The
evaluation set is 100 prompts sampled from the same corpus under
`seed=42`. Expected overlap between training and evaluation is about 15
prompts, leaving roughly 85 truly held out. The eval set is fixed by the
seed; reruns under the same seed pick the same 100 prompts.

### Headline numbers (paper Table 3)

| Model                | Syn.% | ≥1-P% | All-P% | mean R |
|----------------------|------:|------:|-------:|-------:|
| SFT Baseline         |  44.0 |   3.0 |    1.0 |   0.40 |
| PPO-simple (binary)  |  18.0 |   0.0 |    0.0 |   0.38 |
| PPO-fresh (partial)  |  27.0 |   2.0 |    0.0 |   0.40 |
| PPO-cont. (partial)  |  63.0 |   9.0 |    2.0 |   0.42 |

Bootstrap CIs: ±10% on Syn.%, ±6% on ≥1-P%. PPO-simple and PPO-fresh are
statistically indistinguishable on syntax validity. The near-uniform mean R
reflects `R_sec = 1.0` for all variants — Bandit found no MEDIUM/HIGH issues
across 400 samples (100 prompts × 4 variants), as APPS+ algorithmic prompts
rarely produce vulnerability-relevant code.

## Minimal working example

To verify the pipeline runs without budgeting the full ~24h V100 reproduction,
run

```bash
bash scripts/minimal_example.sh
```

This exercises the data prep, SFT, PPO, evaluation, and Bandit paths on a tiny
4-prompt subset and prints `MINIMAL EXAMPLE: PASS` on success. Runtime is
roughly 10 to 30 minutes on a V100 and roughly 30 to 90 minutes on CPU. It
does **not** reproduce paper Table 3 numbers — see the next section for that.

## Reproduction quick-start

```bash
# 1. install dependencies (see requirements.txt)
pip install -r requirements.txt

# 2. prepare the APPS+ stdin training data with up to 5 test cases per problem
python prepare_ppo_data.py --output data/prompts/ppo_prompts_with_tests.json

# 3. SFT warm-start on the stdin subset (3 epochs, LoRA r=16 α=32, cross-entropy)
python train_sft_stdin.py

# 4. PPO with partial-credit reward (500 episodes, batch 2, lr 1e-6, KL λ=0.1, T=0.7)
python train_ppo.py \
    --sft_checkpoint ./checkpoints/sft_stdin/best \
    --prompts_file data/prompts/ppo_prompts_with_tests.json \
    --use_bandit \
    --episodes 500 \
    --batch_size 2 \
    --learning_rate 1e-6
```

For the three-variant table (PPO-simple, PPO-fresh, PPO-continue) the same
PPO command is invoked three times: with `--binary_reward` and SFT init for
PPO-simple; with no `--binary_reward` and SFT init for PPO-fresh; with no
`--binary_reward` and `--ppo_checkpoint <PPO-simple-best> --resume` for
PPO-continue. `run_both_experiments.sh` orchestrates the full sweep.

## Hardware

NVIDIA V100 16 GB, single GPU.

## Project structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── benchmark/                  # multi-LLM baseline at single sample, no in-context examples (paper Table 2)
├── data/
│   ├── full_dataset_ids.json
│   ├── stdin_subset_ids.json
│   ├── function_call_subset_ids.json
│   ├── prompts/                # PPO prompts + test cases (regenerated by prepare_ppo_data.py)
│   └── sft/                    # SFT train/val JSONL (regenerated)
├── evaluate_dual_metrics.py    # 100-prompt eval (Syn., ≥1-P, All-P, R)
├── experiment_results/         # smaller pilot runs (20-sample)
├── paper/                      # camera-ready PDF (LCTES '26 final); source of truth for all claims
├── prepare_ppo_data.py         # APPS+ → stdin prompts with test cases
├── rl_training/
│   ├── ppo_trainer.py          # PPO with partial-credit reward
│   ├── reward_calculator.py    # R = α·R_func + β·R_sec
│   ├── bandit_runner.py        # Bandit MEDIUM/HIGH analysis (paper Eq (2))
│   ├── scoring_agent.py        # rules-based scoring
│   ├── security_weights.py     # CWE/CVSS severity mapping (used by C/KLEE follow-up)
│   ├── sft_trainer.py
│   └── config.py               # RewardConfig, PPOConfig, ModelConfig
├── run_both_experiments.sh     # end-to-end pipeline
├── train_ppo.py                # PPO entry point
└── train_sft_stdin.py          # SFT entry point
```

## Reproducing the paper

The four-step quick-start above reproduces the SFT baseline and the
partial-credit PPO row of paper Table 3. For the full three-variant sweep
(PPO-simple, PPO-fresh, PPO-continue), invoke `run_both_experiments.sh`,
which runs the data prep, SFT, all three PPO variants in sequence, and the
100-prompt evaluation. Expect ~24–28 hours total wall-clock on a single
NVIDIA V100 16 GB. Seeds are fixed at 42 across the eval and data-prep code
paths (`evaluate_dual_metrics.py`, `rl_training/sft_trainer.py`,
`rl_training/data_converter.py`); the LCTES experiments used the same seed.

## Acknowledgments

Forked from [`Achintya-Lakshmanan/basic-rl-feedback-workflow`](https://github.com/Achintya-Lakshmanan/basic-rl-feedback-workflow), which itself forks the root upstream [`SJh29/basic-rl-feedback-workflow`](https://github.com/SJh29/basic-rl-feedback-workflow). The Python-only, partial-credit pipeline, SLM-focused experiments, and APPS+ stdin evaluation are contributions of this LCTES '26 work.
