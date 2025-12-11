#!/usr/bin/env python3
"""
Prepare RL training data from APPS+ analysis results.

Creates:
- data/sft/train.jsonl - Training data for SFT
- data/sft/val.jsonl - Validation data for SFT
- data/prompts/ppo_prompts.txt - Prompts for online PPO training
"""

import json
import os
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "benchmark", "full_results")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROMPTS_PATH = os.path.join(RESULTS_DIR, "prompts.json")

# Configuration
RANDOM_SEED = 42
VAL_SPLIT = 0.1  # 10% validation
MIN_REWARD_THRESHOLD = 0.0  # Include samples with R >= 0

# Reward weights
ALPHA = 0.6  # Functional correctness weight
BETA = 0.4   # Security weight

# Security penalty weights (for Rsec calculation)
SEVERITY_WEIGHTS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 1.0
}


@dataclass
class Sample:
    """Training sample with prompt, code, and reward."""
    prompt_id: str
    prompt: str
    code: str
    model: str
    reward: float
    r_func: float
    r_sec: float
    tests_passed: int
    tests_total: int
    security_issues: int
    compiles: bool


def load_prompts() -> Dict[str, Dict]:
    """Load prompts indexed by ID."""
    with open(PROMPTS_PATH, 'r') as f:
        prompts = json.load(f)
    return {p['id']: p for p in prompts}


def load_stdin_ids() -> set:
    """Load stdin-style subset IDs."""
    path = os.path.join(DATA_DIR, "stdin_subset_ids.json")
    with open(path, 'r') as f:
        data = json.load(f)
    return set(data['ids'])


def load_generated_code(model: str, prompt_id: str) -> Optional[str]:
    """Load generated code for a sample."""
    code_path = os.path.join(RESULTS_DIR, model, "apps_plus", prompt_id, "generated_code.py")
    if os.path.exists(code_path):
        with open(code_path, 'r') as f:
            return f.read()
    return None


def load_analysis(model: str, prompt_id: str) -> Optional[Dict]:
    """Load analysis results for a sample."""
    path = os.path.join(RESULTS_DIR, model, "apps_plus", prompt_id, "analysis.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def calculate_rewards(analysis: Dict) -> Tuple[float, float, float]:
    """
    Calculate reward components from analysis.

    Returns: (total_reward, r_func, r_sec)
    """
    # Functional correctness: tests_passed / tests_total
    test_exec = analysis.get('test_execution', {})
    passed = test_exec.get('passed', 0)
    total = test_exec.get('total_tests', 1)
    r_func = passed / max(total, 1)

    # Security score: 1 - weighted penalty
    security = analysis.get('security', {})
    issues = security.get('issues', [])

    # Count by severity
    severity_counts = {'low': 0, 'medium': 0, 'high': 0}
    for issue in issues:
        sev = issue.get('severity', 'low').lower()
        if sev in severity_counts:
            severity_counts[sev] += 1

    # Calculate penalty (normalized by number of issues to cap at 1.0)
    penalty = 0.0
    total_issues = sum(severity_counts.values())
    if total_issues > 0:
        for sev, count in severity_counts.items():
            penalty += SEVERITY_WEIGHTS.get(sev, 0.3) * count
        penalty = min(penalty / total_issues, 1.0)  # Normalize

    r_sec = 1.0 - penalty

    # Combined reward
    r_total = ALPHA * r_func + BETA * r_sec

    return r_total, r_func, r_sec


def collect_samples(model: str, stdin_ids: set, prompts: Dict[str, Dict]) -> List[Sample]:
    """Collect all samples for a model that meet criteria."""
    samples = []

    for prompt_id in stdin_ids:
        if prompt_id not in prompts:
            continue

        analysis = load_analysis(model, prompt_id)
        if analysis is None:
            continue

        # Check if compiles
        compilation = analysis.get('compilation', {})
        compiles = compilation.get('compiles', False)

        if not compiles:
            continue  # Skip samples that don't compile

        # Load code
        code = load_generated_code(model, prompt_id)
        if code is None:
            continue

        # Get prompt text
        prompt_data = prompts[prompt_id]
        prompt_text = prompt_data.get('prompt', '')

        # Calculate rewards
        r_total, r_func, r_sec = calculate_rewards(analysis)

        # Get stats
        test_exec = analysis.get('test_execution', {})
        security = analysis.get('security', {})

        sample = Sample(
            prompt_id=prompt_id,
            prompt=prompt_text,
            code=code,
            model=model,
            reward=r_total,
            r_func=r_func,
            r_sec=r_sec,
            tests_passed=test_exec.get('passed', 0),
            tests_total=test_exec.get('total_tests', 0),
            security_issues=len(security.get('issues', [])),
            compiles=True
        )

        if sample.reward >= MIN_REWARD_THRESHOLD:
            samples.append(sample)

    return samples


def sample_to_dict(sample: Sample) -> Dict:
    """Convert sample to dictionary for JSON serialization.

    Uses field names compatible with rl_training/data_converter.py:
    - completion (not code)
    - rfunc (not r_func)
    - rsec (not r_sec)
    """
    return {
        'prompt_id': sample.prompt_id,
        'prompt': sample.prompt,
        'completion': sample.code,  # Renamed for compatibility
        'model': sample.model,
        'dataset': 'apps_plus',
        'reward': sample.reward,
        'rfunc': sample.r_func,  # Renamed for compatibility
        'rsec': sample.r_sec,  # Renamed for compatibility
        'compiles': sample.compiles,
        'tests_passed': sample.tests_passed,
        'tests_total': sample.tests_total,
        'has_security_issues': sample.security_issues > 0
    }


def main():
    print("=" * 70)
    print("Preparing RL Training Data")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    # Load data
    print("\nLoading prompts...")
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    print("Loading stdin subset IDs...")
    stdin_ids = load_stdin_ids()
    print(f"Loaded {len(stdin_ids)} stdin-style IDs")

    # Collect samples from DeepSeek (best performing model)
    print("\nCollecting samples from DeepSeek (best performing model)...")
    samples = collect_samples('deepseek', stdin_ids, prompts)
    print(f"Collected {len(samples)} compiling samples with reward >= {MIN_REWARD_THRESHOLD}")

    if not samples:
        print("ERROR: No samples found!")
        return

    # Statistics
    rewards = [s.reward for s in samples]
    r_funcs = [s.r_func for s in samples]
    r_secs = [s.r_sec for s in samples]

    print(f"\nReward statistics:")
    print(f"  Total reward: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={sum(rewards)/len(rewards):.3f}")
    print(f"  R_func:       min={min(r_funcs):.3f}, max={max(r_funcs):.3f}, mean={sum(r_funcs)/len(r_funcs):.3f}")
    print(f"  R_sec:        min={min(r_secs):.3f}, max={max(r_secs):.3f}, mean={sum(r_secs)/len(r_secs):.3f}")

    # Filter to samples with some tests passed
    samples_with_tests = [s for s in samples if s.tests_passed > 0]
    print(f"\nSamples with at least 1 test passed: {len(samples_with_tests)}")

    # For SFT, we want samples with higher rewards
    # Use all compiling samples for broader training signal
    sft_samples = samples

    # Shuffle and split
    random.shuffle(sft_samples)
    val_size = int(len(sft_samples) * VAL_SPLIT)
    val_samples = sft_samples[:val_size]
    train_samples = sft_samples[val_size:]

    print(f"\nSFT data split:")
    print(f"  Training:   {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")

    # Save SFT data
    os.makedirs(os.path.join(DATA_DIR, "sft"), exist_ok=True)

    train_path = os.path.join(DATA_DIR, "sft", "train.jsonl")
    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample_to_dict(sample)) + '\n')
    print(f"\nSaved training data to: {train_path}")

    val_path = os.path.join(DATA_DIR, "sft", "val.jsonl")
    with open(val_path, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample_to_dict(sample)) + '\n')
    print(f"Saved validation data to: {val_path}")

    # Save PPO prompts (unique prompts from all samples)
    os.makedirs(os.path.join(DATA_DIR, "prompts"), exist_ok=True)

    unique_prompts = list(set(s.prompt for s in samples))
    ppo_path = os.path.join(DATA_DIR, "prompts", "ppo_prompts.txt")
    with open(ppo_path, 'w') as f:
        for prompt in unique_prompts:
            # Write prompt with delimiter
            f.write(prompt.strip() + '\n' + '=' * 50 + '\n')
    print(f"Saved PPO prompts to: {ppo_path}")
    print(f"  Unique prompts: {len(unique_prompts)}")

    # Also save as JSON for easier parsing
    ppo_json_path = os.path.join(DATA_DIR, "prompts", "ppo_prompts.json")
    with open(ppo_json_path, 'w') as f:
        json.dump({
            'count': len(unique_prompts),
            'prompts': unique_prompts
        }, f, indent=2)
    print(f"Saved PPO prompts JSON to: {ppo_json_path}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total compiling samples:    {len(samples)}")
    print(f"Samples with tests passed:  {len(samples_with_tests)}")
    print(f"Training samples:           {len(train_samples)}")
    print(f"Validation samples:         {len(val_samples)}")
    print(f"Unique prompts for PPO:     {len(unique_prompts)}")

    # Reward distribution
    print(f"\nReward distribution:")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in bins:
        count = sum(1 for s in samples if low <= s.reward < high)
        pct = count / len(samples) * 100
        print(f"  {low:.1f}-{high:.1f}: {count:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 70)
    print("Done!")

    return {
        'total_samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'unique_prompts': len(unique_prompts)
    }


if __name__ == "__main__":
    main()
