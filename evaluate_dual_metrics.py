#!/usr/bin/env python3
"""
Dual Metrics Evaluation Script for SecureCodeRL

Evaluates models using BOTH simple (training) and enhanced (proposal) metrics
to establish baselines and measure the value of enhanced pipeline.

Metrics computed:
- Simple: Rfunc_simple (syntax), Rsec_simple (regex patterns)
- Enhanced: Rfunc_true (test execution), Rsec_bandit (Bandit analysis)

Usage:
    # Evaluate a single model
    python evaluate_dual_metrics.py \
        --checkpoint checkpoints/ppo/ppo/best \
        --prompts_file data/prompts/ppo_prompts_with_tests.json \
        --num_samples 50

    # Compare multiple models
    python evaluate_dual_metrics.py \
        --checkpoints checkpoints/sft_stdin/best checkpoints/ppo/ppo/best \
        --prompts_file data/prompts/ppo_prompts_with_tests.json \
        --output_dir results/dual_metrics
"""

import argparse
import ast
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

# Check for torch availability
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch transformers peft")


@dataclass
class DualMetrics:
    """Metrics computed using both simple and enhanced methods"""
    # Simple metrics (used during training)
    rfunc_simple: float      # Syntax validity (0 or 1)
    rsec_simple: float       # 1 - (0.2 * regex_pattern_count)
    reward_simple: float     # α * rfunc_simple + β * rsec_simple

    # Enhanced metrics (proposal-aligned)
    rfunc_true: float        # tests_passed / total_tests
    rsec_bandit: float       # Bandit-based security score
    reward_enhanced: float   # α * rfunc_true + β * rsec_bandit

    # Details
    syntax_valid: bool
    tests_passed: int
    tests_total: int
    regex_findings: int
    bandit_findings: int


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model"""
    model_name: str
    checkpoint_path: str
    num_samples: int

    # Aggregate metrics
    mean_rfunc_simple: float
    mean_rfunc_true: float
    mean_rsec_simple: float
    mean_rsec_bandit: float
    mean_reward_simple: float
    mean_reward_enhanced: float

    # Pass rates
    syntax_pass_rate: float
    test_pass_rate: float      # % with at least one test passing
    security_clean_rate: float # % with no security findings

    # Detailed results
    per_sample_metrics: List[Dict]


class DualMetricsEvaluator:
    """Evaluates code generation models with both simple and enhanced metrics"""

    # Regex patterns for simple security check
    SECURITY_PATTERNS = [
        (r'\beval\s*\(', 'eval'),
        (r'\bexec\s*\(', 'exec'),
        (r'\b__import__\s*\(', '__import__'),
        (r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True', 'shell=True'),
        (r'\bos\.system\s*\(', 'os.system'),
        (r'\bos\.popen\s*\(', 'os.popen'),
        (r'\bpickle\.loads?\s*\(', 'pickle'),
        (r'\byaml\.load\s*\([^)]*\)', 'yaml.load'),
        (r'input\s*\(\s*\)', 'input'),
    ]

    def __init__(
        self,
        model_path: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
        alpha: float = 0.6,
        beta: float = 0.4,
        use_bandit: bool = True,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.alpha = alpha
        self.beta = beta
        self.use_bandit = use_bandit
        self.device = device

        self.model = None
        self.tokenizer = None
        self.bandit_runner = None

        # Initialize Bandit runner if available
        if use_bandit:
            try:
                from rl_training.bandit_runner import BanditRunner
                self.bandit_runner = BanditRunner(use_bandit=True)
            except ImportError:
                print("Warning: Bandit runner not available, using regex fallback")
                self.bandit_runner = None

    def load_model(self, checkpoint_path: str):
        """Load model with LoRA checkpoint"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        print(f"Loading model from {checkpoint_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Load LoRA weights if checkpoint exists
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
                is_trainable=False,
            )
            print(f"Loaded LoRA weights from {checkpoint_path}")

        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate code completion"""
        formatted = f"### Instruction:\nComplete the following code:\n\n{prompt}\n\n### Response:\n"

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = outputs[0, prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def compute_simple_rfunc(self, code: str) -> Tuple[float, bool]:
        """Compute simple Rfunc (syntax validity)"""
        if not code or len(code.strip()) < 5:
            return 0.0, False
        try:
            ast.parse(code)
            return 1.0, True
        except SyntaxError:
            return 0.0, False

    def compute_simple_rsec(self, code: str) -> Tuple[float, int]:
        """Compute simple Rsec (regex pattern count)"""
        findings = 0
        for pattern, _ in self.SECURITY_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                findings += 1

        rsec = max(0.0, 1.0 - (0.2 * findings))
        return rsec, findings

    def compute_true_rfunc(
        self,
        code: str,
        test_cases: Dict,
    ) -> Tuple[float, int, int]:
        """Compute true Rfunc (test execution)"""
        import subprocess

        inputs = test_cases.get('inputs', [])
        outputs = test_cases.get('outputs', [])

        if not inputs or not outputs:
            # No tests - fall back to syntax check
            syntax_valid, _ = self.compute_simple_rfunc(code)
            return syntax_valid, 0, 0

        num_tests = min(len(inputs), len(outputs))
        passed = 0

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name

        try:
            for i in range(num_tests):
                test_input = inputs[i]
                expected_output = outputs[i]

                try:
                    result = subprocess.run(
                        ['python', code_file],
                        input=test_input,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    actual = result.stdout.strip()
                    expected = expected_output.strip()

                    if actual == expected:
                        passed += 1

                except (subprocess.TimeoutExpired, Exception):
                    pass

        finally:
            try:
                os.unlink(code_file)
            except OSError:
                pass

        rfunc = passed / num_tests if num_tests > 0 else 0.0
        return rfunc, passed, num_tests

    def compute_bandit_rsec(self, code: str) -> Tuple[float, int]:
        """Compute Bandit-based Rsec"""
        if self.bandit_runner is None:
            # Fall back to regex
            return self.compute_simple_rsec(code)

        findings = self.bandit_runner.analyze(code)
        rsec = self.bandit_runner.compute_rsec(findings)
        return rsec, len(findings)

    def compute_dual_metrics(
        self,
        code: str,
        test_cases: Optional[Dict] = None,
    ) -> DualMetrics:
        """Compute both simple and enhanced metrics for code"""

        # Simple metrics
        rfunc_simple, syntax_valid = self.compute_simple_rfunc(code)
        rsec_simple, regex_findings = self.compute_simple_rsec(code)
        reward_simple = self.alpha * rfunc_simple + self.beta * rsec_simple

        # Enhanced metrics
        test_cases = test_cases or {'inputs': [], 'outputs': []}
        rfunc_true, tests_passed, tests_total = self.compute_true_rfunc(code, test_cases)
        rsec_bandit, bandit_findings = self.compute_bandit_rsec(code)
        reward_enhanced = self.alpha * rfunc_true + self.beta * rsec_bandit

        return DualMetrics(
            rfunc_simple=rfunc_simple,
            rsec_simple=rsec_simple,
            reward_simple=reward_simple,
            rfunc_true=rfunc_true,
            rsec_bandit=rsec_bandit,
            reward_enhanced=reward_enhanced,
            syntax_valid=syntax_valid,
            tests_passed=tests_passed,
            tests_total=tests_total,
            regex_findings=regex_findings,
            bandit_findings=bandit_findings,
        )

    def evaluate(
        self,
        checkpoint_path: str,
        prompts: List[Dict],
        num_samples: int = 50,
    ) -> EvaluationResult:
        """Evaluate a model checkpoint on prompts"""

        # Load model
        self.load_model(checkpoint_path)

        # Sample prompts
        if len(prompts) > num_samples:
            prompts = random.sample(prompts, num_samples)

        # Generate and evaluate
        all_metrics = []
        for i, prompt_item in enumerate(prompts):
            if isinstance(prompt_item, dict):
                prompt = prompt_item.get('prompt', '')
                test_cases = prompt_item.get('test_cases', {'inputs': [], 'outputs': []})
            else:
                prompt = prompt_item
                test_cases = {'inputs': [], 'outputs': []}

            # Generate code
            try:
                completion = self.generate(prompt)
            except Exception as e:
                print(f"Generation failed for sample {i}: {e}")
                completion = ""

            # Compute metrics
            metrics = self.compute_dual_metrics(completion, test_cases)
            all_metrics.append(asdict(metrics))

            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(prompts)} samples")

        # Aggregate
        n = len(all_metrics)
        if n == 0:
            raise ValueError("No samples evaluated")

        result = EvaluationResult(
            model_name=Path(checkpoint_path).name,
            checkpoint_path=str(checkpoint_path),
            num_samples=n,
            mean_rfunc_simple=sum(m['rfunc_simple'] for m in all_metrics) / n,
            mean_rfunc_true=sum(m['rfunc_true'] for m in all_metrics) / n,
            mean_rsec_simple=sum(m['rsec_simple'] for m in all_metrics) / n,
            mean_rsec_bandit=sum(m['rsec_bandit'] for m in all_metrics) / n,
            mean_reward_simple=sum(m['reward_simple'] for m in all_metrics) / n,
            mean_reward_enhanced=sum(m['reward_enhanced'] for m in all_metrics) / n,
            syntax_pass_rate=sum(1 for m in all_metrics if m['syntax_valid']) / n,
            test_pass_rate=sum(1 for m in all_metrics if m['tests_passed'] > 0) / n,
            security_clean_rate=sum(1 for m in all_metrics if m['bandit_findings'] == 0) / n,
            per_sample_metrics=all_metrics,
        )

        # Clear GPU memory
        del self.model
        self.model = None
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()

        return result


def load_prompts(prompts_file: str) -> List[Dict]:
    """Load prompts from JSON file"""
    with open(prompts_file) as f:
        data = json.load(f)

    if data.get('format_version') == '2.0' and 'prompts_with_tests' in data:
        return data['prompts_with_tests']
    elif 'prompts' in data:
        return [{'prompt': p, 'test_cases': {'inputs': [], 'outputs': []}} for p in data['prompts']]
    else:
        raise ValueError(f"Unknown prompts format in {prompts_file}")


def print_results_table(results: List[EvaluationResult]):
    """Print comparison table"""
    print("\n" + "=" * 100)
    print("DUAL METRICS EVALUATION RESULTS")
    print("=" * 100)

    # Header
    print(f"\n{'Model':<20} | {'Rfunc_simple':>12} | {'Rfunc_true':>10} | {'Rsec_simple':>11} | {'Rsec_bandit':>11} | {'R_simple':>8} | {'R_enhanced':>10}")
    print("-" * 100)

    # Results
    for r in results:
        print(f"{r.model_name:<20} | {r.mean_rfunc_simple:>12.4f} | {r.mean_rfunc_true:>10.4f} | {r.mean_rsec_simple:>11.4f} | {r.mean_rsec_bandit:>11.4f} | {r.mean_reward_simple:>8.4f} | {r.mean_reward_enhanced:>10.4f}")

    print("-" * 100)

    # Additional stats
    print(f"\n{'Model':<20} | {'Syntax Pass%':>12} | {'Test Pass%':>10} | {'Security Clean%':>15}")
    print("-" * 65)
    for r in results:
        print(f"{r.model_name:<20} | {100*r.syntax_pass_rate:>11.1f}% | {100*r.test_pass_rate:>9.1f}% | {100*r.security_clean_rate:>14.1f}%")

    # Insights
    if len(results) >= 2:
        base = results[0]
        for r in results[1:]:
            print(f"\n--- Comparison: {r.model_name} vs {base.model_name} ---")
            print(f"  Rfunc_simple: {'+' if r.mean_rfunc_simple >= base.mean_rfunc_simple else ''}{(r.mean_rfunc_simple - base.mean_rfunc_simple)*100:.1f}%")
            print(f"  Rfunc_true:   {'+' if r.mean_rfunc_true >= base.mean_rfunc_true else ''}{(r.mean_rfunc_true - base.mean_rfunc_true)*100:.1f}%")
            print(f"  Rsec_bandit:  {'+' if r.mean_rsec_bandit >= base.mean_rsec_bandit else ''}{(r.mean_rsec_bandit - base.mean_rsec_bandit)*100:.1f}%")
            print(f"  R_enhanced:   {'+' if r.mean_reward_enhanced >= base.mean_reward_enhanced else ''}{(r.mean_reward_enhanced - base.mean_reward_enhanced)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Dual Metrics Evaluation")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint to evaluate")
    parser.add_argument("--checkpoints", nargs="+", help="Multiple checkpoints to compare")
    parser.add_argument("--prompts_file", type=str, required=True, help="Prompts JSON file")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/dual_metrics", help="Output directory")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for Rfunc")
    parser.add_argument("--beta", type=float, default=0.4, help="Weight for Rsec")
    parser.add_argument("--no_bandit", action="store_true", help="Disable Bandit (use regex)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    # Determine checkpoints to evaluate
    checkpoints = []
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    elif args.checkpoints:
        checkpoints = args.checkpoints
    else:
        print("Error: Provide --checkpoint or --checkpoints")
        return 1

    # Load prompts
    print(f"Loading prompts from {args.prompts_file}...")
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts")

    # Create evaluator
    evaluator = DualMetricsEvaluator(
        model_path=args.model_path,
        alpha=args.alpha,
        beta=args.beta,
        use_bandit=not args.no_bandit,
    )

    # Evaluate each checkpoint
    results = []
    for checkpoint in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint}")
        print('='*60)

        result = evaluator.evaluate(checkpoint, prompts, args.num_samples)
        results.append(result)

    # Print results
    print_results_table(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Save summary
    summary_file = output_dir / "summary.md"
    with open(summary_file, 'w') as f:
        f.write("# Dual Metrics Evaluation Summary\n\n")
        f.write(f"**Samples per model:** {args.num_samples}\n")
        f.write(f"**Alpha (Rfunc weight):** {args.alpha}\n")
        f.write(f"**Beta (Rsec weight):** {args.beta}\n\n")

        f.write("## Results Table\n\n")
        f.write("| Model | Rfunc_simple | Rfunc_true | Rsec_simple | Rsec_bandit | R_simple | R_enhanced |\n")
        f.write("|-------|--------------|------------|-------------|-------------|----------|------------|\n")
        for r in results:
            f.write(f"| {r.model_name} | {r.mean_rfunc_simple:.4f} | {r.mean_rfunc_true:.4f} | {r.mean_rsec_simple:.4f} | {r.mean_rsec_bandit:.4f} | {r.mean_reward_simple:.4f} | {r.mean_reward_enhanced:.4f} |\n")

        f.write("\n## Pass Rates\n\n")
        f.write("| Model | Syntax Pass | Test Pass | Security Clean |\n")
        f.write("|-------|-------------|-----------|----------------|\n")
        for r in results:
            f.write(f"| {r.model_name} | {100*r.syntax_pass_rate:.1f}% | {100*r.test_pass_rate:.1f}% | {100*r.security_clean_rate:.1f}% |\n")

    print(f"Summary saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
