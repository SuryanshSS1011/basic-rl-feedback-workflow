#!/usr/bin/env python3
"""
Prepare PPO Training Data with Test Cases

Extracts prompts and test cases from APPS+ dataset to create
an augmented training data file that enables true Rfunc computation.

Usage:
    python prepare_ppo_data.py
    python prepare_ppo_data.py --output data/prompts/ppo_prompts_with_tests.json

Output format:
{
    "count": N,
    "prompts_with_tests": [
        {
            "prompt": "def func():\\n    ...",
            "test_cases": {
                "inputs": ["5\\n", "10\\n"],
                "outputs": ["120\\n", "3628800\\n"]
            },
            "problem_id": "apps_plus_123",
            "difficulty": "introductory"
        },
        ...
    ]
}
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


def load_apps_plus(cache_path: str = "./benchmark/dataset_cache/apps_plus.json") -> List[Dict]:
    """Load APPS+ dataset from cache or download"""
    if os.path.exists(cache_path):
        print(f"Loading APPS+ from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    # Download if not cached
    print("Downloading APPS+ dataset...")
    import requests
    url = "https://raw.githubusercontent.com/Ablustrund/APPS_Plus/refs/heads/main/data/v1/data.json"

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    data = response.json()

    # Cache for future use
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(data, f)

    return data


def extract_test_cases(item: Dict) -> Dict:
    """Extract test cases from APPS+ item"""
    # APPS+ has 'inputs' and 'outputs' fields
    inputs = item.get('inputs', item.get('input', []))
    outputs = item.get('outputs', item.get('output', []))

    # Ensure they're lists
    if isinstance(inputs, str):
        inputs = [inputs] if inputs else []
    if isinstance(outputs, str):
        outputs = [outputs] if outputs else []

    # Clean up test cases (ensure newline at end)
    cleaned_inputs = []
    cleaned_outputs = []

    for inp in inputs:
        if inp is not None:
            inp_str = str(inp)
            if not inp_str.endswith('\n'):
                inp_str += '\n'
            cleaned_inputs.append(inp_str)

    for out in outputs:
        if out is not None:
            out_str = str(out)
            if not out_str.endswith('\n'):
                out_str += '\n'
            cleaned_outputs.append(out_str)

    return {
        'inputs': cleaned_inputs,
        'outputs': cleaned_outputs,
    }


def extract_prompt_from_question(question: str) -> str:
    """
    Convert APPS+ question to a function stub format.

    Extracts the problem description and creates a starter template.
    """
    # Create a simple function stub with the question as docstring
    # Clean up the question for use as docstring
    question_cleaned = question.strip()

    # Generate a random-ish function name
    import hashlib
    hash_val = hashlib.md5(question[:50].encode()).hexdigest()[:5]
    func_name = f"solve_{hash_val}"

    prompt = f'''def {func_name}():
    """{question_cleaned}
    """
'''
    return prompt


def filter_stdin_problems(data: List[Dict]) -> List[Dict]:
    """Filter to keep only stdin-style problems (not function-call style)"""
    stdin_problems = []

    for item in data:
        # Check if this is a stdin-style problem
        # These typically have input/output test cases
        inputs = item.get('inputs', item.get('input', []))
        outputs = item.get('outputs', item.get('output', []))

        # Must have at least one test case
        if not inputs or not outputs:
            continue

        # Normalize to lists
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]

        # Must have at least one valid test case pair
        if len(inputs) > 0 and len(outputs) > 0:
            stdin_problems.append(item)

    return stdin_problems


def prepare_ppo_data(
    max_problems: Optional[int] = None,
    max_tests_per_problem: int = 5,
    include_all_prompts: bool = False,
) -> Dict:
    """
    Prepare PPO training data with test cases.

    Args:
        max_problems: Maximum number of problems to include
        max_tests_per_problem: Maximum test cases per problem (for speed)
        include_all_prompts: If True, include problems without test cases too

    Returns:
        Dict with prompts and test cases
    """
    # Load APPS+ data
    data = load_apps_plus()
    print(f"Loaded {len(data)} total problems")

    # Filter to stdin-style problems
    stdin_data = filter_stdin_problems(data)
    print(f"Found {len(stdin_data)} stdin-style problems with test cases")

    # Limit if requested
    if max_problems:
        stdin_data = stdin_data[:max_problems]

    # Process each problem
    prompts_with_tests = []

    for i, item in enumerate(stdin_data):
        question = item.get('question', item.get('prompt', item.get('description', '')))
        if not question:
            continue

        # Extract test cases
        test_cases = extract_test_cases(item)

        # Limit test cases per problem
        if max_tests_per_problem:
            test_cases['inputs'] = test_cases['inputs'][:max_tests_per_problem]
            test_cases['outputs'] = test_cases['outputs'][:max_tests_per_problem]

        # Skip if no test cases after filtering
        if not test_cases['inputs'] or not test_cases['outputs']:
            if not include_all_prompts:
                continue

        # Create prompt
        prompt = extract_prompt_from_question(question)

        prompts_with_tests.append({
            'prompt': prompt,
            'test_cases': test_cases,
            'problem_id': item.get('id', item.get('problem_id', f'apps_plus_{i}')),
            'difficulty': item.get('difficulty', 'unknown'),
            'num_tests': len(test_cases['inputs']),
        })

    print(f"Prepared {len(prompts_with_tests)} problems with test cases")

    return {
        'count': len(prompts_with_tests),
        'max_tests_per_problem': max_tests_per_problem,
        'format_version': '2.0',  # Indicates new format with test cases
        'prompts_with_tests': prompts_with_tests,
    }


def convert_existing_prompts(
    existing_path: str,
    output_path: str,
) -> Dict:
    """
    Convert existing prompts file to new format with empty test cases.

    This allows using the existing prompts while the enhanced pipeline
    is being developed.
    """
    print(f"Loading existing prompts from: {existing_path}")
    with open(existing_path, 'r') as f:
        data = json.load(f)

    prompts = data.get('prompts', [])

    # Convert to new format
    prompts_with_tests = []
    for i, prompt in enumerate(prompts):
        prompts_with_tests.append({
            'prompt': prompt,
            'test_cases': {'inputs': [], 'outputs': []},
            'problem_id': f'legacy_{i}',
            'difficulty': 'unknown',
            'num_tests': 0,
        })

    result = {
        'count': len(prompts_with_tests),
        'max_tests_per_problem': 0,
        'format_version': '2.0',
        'prompts_with_tests': prompts_with_tests,
    }

    print(f"Converted {len(prompts_with_tests)} prompts to new format")
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare PPO training data with test cases")
    parser.add_argument(
        "--output",
        type=str,
        default="data/prompts/ppo_prompts_with_tests.json",
        help="Output file path",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Maximum number of problems to include",
    )
    parser.add_argument(
        "--max_tests",
        type=int,
        default=5,
        help="Maximum test cases per problem",
    )
    parser.add_argument(
        "--convert_existing",
        action="store_true",
        help="Convert existing prompts file instead of extracting from APPS+",
    )
    parser.add_argument(
        "--existing_path",
        type=str,
        default="data/prompts/ppo_prompts.json",
        help="Path to existing prompts file (for --convert_existing)",
    )

    args = parser.parse_args()

    if args.convert_existing:
        result = convert_existing_prompts(args.existing_path, args.output)
    else:
        result = prepare_ppo_data(
            max_problems=args.max_problems,
            max_tests_per_problem=args.max_tests,
        )

    # Save to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {result['count']} prompts with test cases to: {output_path}")

    # Print summary
    if result['prompts_with_tests']:
        tests_per_problem = [p['num_tests'] for p in result['prompts_with_tests']]
        total_tests = sum(tests_per_problem)
        avg_tests = total_tests / len(tests_per_problem) if tests_per_problem else 0
        problems_with_tests = sum(1 for t in tests_per_problem if t > 0)

        print(f"\nSummary:")
        print(f"  Total problems: {result['count']}")
        print(f"  Problems with test cases: {problems_with_tests}")
        print(f"  Total test cases: {total_tests}")
        print(f"  Average tests per problem: {avg_tests:.1f}")


if __name__ == "__main__":
    main()
