#!/usr/bin/env python3
"""
Filter APPS+ dataset to stdin-style subset.

Stdin-style: Test inputs are strings (passed via stdin)
Function-call style: Test inputs are lists/dicts (passed as arguments)

This script identifies stdin-style samples for methodologically cleaner evaluation.
"""

import json
import os
from typing import Dict, List, Any
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(BASE_DIR, "benchmark", "full_results", "prompts.json")
DATA_DIR = os.path.join(BASE_DIR, "data")


def is_stdin_style(test_cases: Dict[str, Any]) -> bool:
    """
    Check if test cases use stdin-style string inputs.

    Stdin-style: inputs are strings (e.g., "3\n5 0 -5\n")
    Function-call style: inputs are lists/dicts (e.g., [[1, 2, 3], "abc"])
    """
    inputs = test_cases.get('inputs', [])
    if not inputs:
        return False

    # All inputs must be strings for stdin-style
    return all(isinstance(inp, str) for inp in inputs)


def analyze_prompts(prompts: List[Dict]) -> Dict[str, Any]:
    """Analyze the prompts and categorize by input style."""
    stdin_ids = []
    function_call_ids = []

    input_type_counts = defaultdict(int)

    for prompt in prompts:
        prompt_id = prompt.get('id', '')
        test_cases = prompt.get('test_cases', {})
        inputs = test_cases.get('inputs', [])

        # Track input types
        for inp in inputs:
            input_type_counts[type(inp).__name__] += 1

        if is_stdin_style(test_cases):
            stdin_ids.append(prompt_id)
        else:
            function_call_ids.append(prompt_id)

    return {
        'stdin_ids': stdin_ids,
        'function_call_ids': function_call_ids,
        'input_type_counts': dict(input_type_counts)
    }


def main():
    print("=" * 60)
    print("APPS+ Dataset Filtering: Stdin-Style Subset")
    print("=" * 60)

    # Load prompts
    print(f"\nLoading prompts from: {PROMPTS_PATH}")
    with open(PROMPTS_PATH, 'r') as f:
        prompts = json.load(f)

    total_prompts = len(prompts)
    print(f"Total prompts loaded: {total_prompts}")

    # Analyze and categorize
    print("\nAnalyzing test input formats...")
    analysis = analyze_prompts(prompts)

    stdin_ids = analysis['stdin_ids']
    function_call_ids = analysis['function_call_ids']
    input_type_counts = analysis['input_type_counts']

    # Statistics
    stdin_count = len(stdin_ids)
    func_count = len(function_call_ids)
    stdin_pct = (stdin_count / total_prompts) * 100
    func_pct = (func_count / total_prompts) * 100

    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)
    print(f"Total prompts:        {total_prompts}")
    print(f"Stdin-style:          {stdin_count} ({stdin_pct:.1f}%)")
    print(f"Function-call style:  {func_count} ({func_pct:.1f}%)")
    print("\nInput type distribution:")
    for input_type, count in sorted(input_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {input_type}: {count}")

    # Save results
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save stdin subset IDs
    stdin_path = os.path.join(DATA_DIR, "stdin_subset_ids.json")
    with open(stdin_path, 'w') as f:
        json.dump({
            'description': 'APPS+ stdin-style subset (string inputs)',
            'count': len(stdin_ids),
            'percentage': stdin_pct,
            'ids': stdin_ids
        }, f, indent=2)
    print(f"\nSaved stdin subset IDs to: {stdin_path}")

    # Save function-call subset IDs
    func_path = os.path.join(DATA_DIR, "function_call_subset_ids.json")
    with open(func_path, 'w') as f:
        json.dump({
            'description': 'APPS+ function-call style subset (list/dict inputs)',
            'count': len(function_call_ids),
            'percentage': func_pct,
            'ids': function_call_ids
        }, f, indent=2)
    print(f"Saved function-call subset IDs to: {func_path}")

    # Save full dataset IDs
    full_path = os.path.join(DATA_DIR, "full_dataset_ids.json")
    all_ids = [p['id'] for p in prompts]
    with open(full_path, 'w') as f:
        json.dump({
            'description': 'Full APPS+ dataset (all input styles)',
            'count': len(all_ids),
            'ids': all_ids
        }, f, indent=2)
    print(f"Saved full dataset IDs to: {full_path}")

    # Calculate expected samples for 3 models
    models = ['deepseek', 'codellama', 'starcoder2']
    print("\n" + "-" * 40)
    print("EXPECTED SAMPLE COUNTS (3 models)")
    print("-" * 40)
    print(f"Full dataset:    {total_prompts * 3:,} samples ({total_prompts} × 3 models)")
    print(f"Stdin subset:    {stdin_count * 3:,} samples ({stdin_count} × 3 models)")
    print(f"Func-call subset: {func_count * 3:,} samples ({func_count} × 3 models)")

    print("\n" + "=" * 60)
    print("Done! Dataset filtering complete.")
    print("=" * 60)

    return {
        'stdin_count': stdin_count,
        'function_call_count': func_count,
        'total': total_prompts
    }


if __name__ == "__main__":
    main()
