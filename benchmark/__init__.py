"""
Benchmark package for multi-LLM code generation evaluation.

Modules:
- dataset_loader: Load prompts from multiple datasets
- multi_model_runner: Generate code with multiple LLMs
- analyze_multi: Analyze generated code for errors
- compute_metrics: Compute and visualize metrics
- language_detector: Detect programming language from code
- code_cleaner: Clean LLM-generated code
- test_executor: Execute code against test cases
"""

from .language_detector import LanguageDetector, detect_language
from .code_cleaner import CodeCleaner, clean_code
from .test_executor import TestExecutor, FailureType, TestResult, TestSuiteResult, run_tests

__all__ = [
    'LanguageDetector',
    'detect_language',
    'CodeCleaner',
    'clean_code',
    'TestExecutor',
    'FailureType',
    'TestResult',
    'TestSuiteResult',
    'run_tests',
]
