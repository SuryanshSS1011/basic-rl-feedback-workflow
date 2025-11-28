#!/usr/bin/env python3
"""
Quick test script - generates code for just 1 prompt to verify everything works.
Useful for testing on CPU/slow machines.
"""

import os
import sys
import json
from pathlib import Path

# Add benchmark directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_model_runner import MultiModelRunner
from analyze_multi import CodeAnalyzer
from compute_metrics import MetricsComputer

print("="*60)
print("QUICK TEST - Single Prompt Generation")
print("="*60)
print()

# Create a simple test prompt
test_prompts = [
    {
        'id': 'quick_test_001',
        'prompt': 'Write a C program that calculates the factorial of a number using recursion',
        'language': 'C',
        'dataset': 'quick_test'
    }
]

print(f"Test prompt: {test_prompts[0]['prompt']}")
print()

# Setup paths
cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf_cache")
results_dir = "./results"

print(f"Cache directory: {cache_dir}")
print(f"Results directory: {results_dir}")
print()

# Initialize runner
print("Initializing model runner...")
runner = MultiModelRunner(cache_dir=cache_dir, results_dir=results_dir)

# Generate with small model
print("\nGenerating code (this may take 2-5 minutes on CPU)...")
print("Model: deepseek-coder-1.3b-instruct")
print()

results = runner.generate_with_model(
    model_name='deepseek-small',
    prompts=test_prompts,
    max_new_tokens=256,  # Reduced for speed
    temperature=0.7
)

print("\n" + "="*60)
print("GENERATION COMPLETE!")
print("="*60)

if results and results[0].get('generated_code'):
    print("\nGenerated Code:")
    print("-"*60)
    print(results[0]['generated_code'])
    print("-"*60)

    # Save to file for inspection
    output_path = Path(results_dir) / "quick_test_output.c"
    with open(output_path, 'w') as f:
        f.write(results[0]['generated_code'])
    print(f"\nCode saved to: {output_path}")

    # Try compilation check
    print("\nChecking compilation...")
    analyzer = CodeAnalyzer(results_dir=results_dir)

    # Find the generated file
    code_dir = Path(results_dir) / "deepseek-small" / "quick_test" / "quick_test_001"
    if code_dir.exists():
        analysis = analyzer.analyze_generated_code("deepseek-small", "quick_test", "quick_test_001")

        print(f"\nCompilation result:")
        if analysis['compilation']:
            if analysis['compilation']['compiles']:
                print("✅ Code compiles successfully!")
            else:
                print("❌ Compilation errors:")
                for error in analysis['compilation']['errors']:
                    print(f"  - {error}")

    print("\n" + "="*60)
    print("Quick test complete! The full benchmark will work the same way,")
    print("just with more prompts and models.")
    print("="*60)

else:
    print("❌ Generation failed")
    if results:
        print(f"Error: {results[0].get('error', 'Unknown error')}")
