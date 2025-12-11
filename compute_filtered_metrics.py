#!/usr/bin/env python3
"""
Compute metrics on filtered stdin-style subset of APPS+ dataset.

Aggregates analysis results for each model, filtered to stdin-style samples only.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "benchmark", "full_results")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODELS = ['deepseek', 'codellama', 'starcoder2']


def load_stdin_ids() -> set:
    """Load the stdin-style subset IDs."""
    path = os.path.join(DATA_DIR, "stdin_subset_ids.json")
    with open(path, 'r') as f:
        data = json.load(f)
    return set(data['ids'])


def load_analysis(model: str, prompt_id: str) -> Optional[Dict]:
    """Load analysis.json for a specific sample."""
    path = os.path.join(RESULTS_DIR, model, "apps_plus", prompt_id, "analysis.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def compute_model_metrics(model: str, stdin_ids: set) -> Dict[str, Any]:
    """Compute aggregated metrics for a model on stdin subset."""
    # Counters
    total = 0
    compiles = 0
    compilation_errors = defaultdict(int)

    semantic_total = 0  # samples that compile
    semantic_errors = 0
    produces_output = 0
    tests_passed_total = 0
    tests_total = 0

    security_issues = 0
    severity_counts = defaultdict(int)

    # Process each stdin sample
    model_dir = os.path.join(RESULTS_DIR, model, "apps_plus")
    if not os.path.exists(model_dir):
        print(f"Warning: Model directory not found: {model_dir}")
        return {}

    for prompt_id in stdin_ids:
        analysis = load_analysis(model, prompt_id)
        if analysis is None:
            continue

        total += 1

        # Compilation
        compilation = analysis.get('compilation', {})
        if compilation.get('compiles', False):
            compiles += 1
            semantic_total += 1

            # Test execution (semantic)
            test_exec = analysis.get('test_execution', {})
            passed = test_exec.get('passed', 0)
            failed = test_exec.get('failed', 0)
            total_tests = test_exec.get('total_tests', passed + failed)

            tests_passed_total += passed
            tests_total += total_tests

            if test_exec.get('has_semantic_error', True):
                semantic_errors += 1
            else:
                produces_output += 1
        else:
            error = compilation.get('error', 'Unknown')
            error_type = error.split(':')[0] if error else 'Unknown'
            compilation_errors[error_type] += 1

        # Security
        security = analysis.get('security', {})
        if security.get('has_security_issues', False):
            security_issues += 1
            severity = security.get('severity', 'low')
            severity_counts[severity] += 1

    # Compute rates
    compilation_rate = (compiles / total * 100) if total > 0 else 0
    semantic_error_rate = (semantic_errors / semantic_total * 100) if semantic_total > 0 else 0
    output_rate = (produces_output / semantic_total * 100) if semantic_total > 0 else 0
    security_rate = (security_issues / total * 100) if total > 0 else 0
    test_pass_rate = (tests_passed_total / tests_total * 100) if tests_total > 0 else 0

    return {
        'total_samples': total,
        'compilation': {
            'compiles': compiles,
            'errors': total - compiles,
            'success_rate': compilation_rate,
            'error_rate': 100 - compilation_rate,
            'error_types': dict(compilation_errors)
        },
        'semantic': {
            'compiling_samples': semantic_total,
            'errors': semantic_errors,
            'produces_output': produces_output,
            'error_rate': semantic_error_rate,
            'output_rate': output_rate,
            'tests_passed_total': tests_passed_total,
            'tests_total': tests_total,
            'test_pass_rate': test_pass_rate
        },
        'security': {
            'issues_count': security_issues,
            'issue_rate': security_rate,
            'by_severity': dict(severity_counts)
        }
    }


def main():
    print("=" * 70)
    print("APPS+ Stdin Subset: Metrics Computation")
    print("=" * 70)

    # Load stdin subset IDs
    stdin_ids = load_stdin_ids()
    print(f"\nLoaded {len(stdin_ids)} stdin-style prompt IDs")

    results = {}

    for model in MODELS:
        print(f"\n{'-' * 50}")
        print(f"Processing: {model}")
        print(f"{'-' * 50}")

        metrics = compute_model_metrics(model, stdin_ids)
        results[model] = metrics

        if not metrics:
            print("  No data found")
            continue

        # Print summary
        print(f"  Total samples:      {metrics['total_samples']}")
        print(f"  Compilation rate:   {metrics['compilation']['success_rate']:.1f}%")
        print(f"  Semantic error rate: {metrics['semantic']['error_rate']:.1f}%")
        print(f"  Output rate:        {metrics['semantic']['output_rate']:.1f}%")
        print(f"  Test pass rate:     {metrics['semantic']['test_pass_rate']:.1f}%")
        print(f"  Security issues:    {metrics['security']['issue_rate']:.2f}%")

    # Save results
    output_path = os.path.join(BASE_DIR, "analysis_results_stdin.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE: Stdin Subset Metrics")
    print(f"{'=' * 70}")
    print(f"{'Model':<12} {'Compile%':>10} {'Output%':>10} {'Pass%':>10} {'Security%':>10}")
    print("-" * 52)
    for model in MODELS:
        m = results.get(model, {})
        comp = m.get('compilation', {}).get('success_rate', 0)
        out = m.get('semantic', {}).get('output_rate', 0)
        passrate = m.get('semantic', {}).get('test_pass_rate', 0)
        sec = m.get('security', {}).get('issue_rate', 0)
        print(f"{model:<12} {comp:>10.1f} {out:>10.1f} {passrate:>10.1f} {sec:>10.2f}")

    print(f"\n{'=' * 70}")
    print("Done!")

    return results


if __name__ == "__main__":
    main()
