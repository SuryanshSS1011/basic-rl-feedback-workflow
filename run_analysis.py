#!/usr/bin/env python3
"""
Analyze Generated Code for Benchmark Results

Computes three key metrics:
1. Compilation errors - Python syntax errors (py_compile/ast)
2. Semantic errors - Code fails test cases
3. Security issues - Vulnerabilities detected by Bandit

Usage:
    python run_analysis.py --results_dir ./benchmark/full_results
    python run_analysis.py --results_dir ./benchmark/full_results --model starcoder2
"""

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


# ============================================================================
# Entry Point Injection - Fix for functions that are defined but never called
# ============================================================================

def extract_function_name(prompt: str) -> Optional[str]:
    """
    Extract function name from prompt like 'def OXzHP():'.

    The APPS+ dataset prompts are function stubs with docstrings.
    This extracts the function name so we can call it.
    """
    if not prompt:
        return None
    match = re.search(r'def\s+(\w+)\s*\(', prompt)
    return match.group(1) if match else None


def has_entry_point(code: str, func_name: str) -> bool:
    """
    Check if code already has an entry point that would execute the function.

    Returns True if:
    - Code has 'if __name__' block
    - Code has a standalone call to func_name() at module level
    """
    if not func_name:
        return True  # Can't check without function name

    # Check for __main__ block
    if '__main__' in code:
        return True

    # Check for standalone function call at module level (not inside a function)
    # Look at the last part of the file for a call pattern
    lines = code.strip().split('\n')

    # Check last 10 lines for a module-level call
    for line in lines[-10:]:
        stripped = line.strip()
        # Must not be inside a function (no leading whitespace for module-level)
        # and must be just the function call
        if stripped == f'{func_name}()':
            return True
        # Also check for call with result assignment
        if stripped.endswith(f'{func_name}()') and not stripped.startswith('def '):
            # Check if this line is at module level (not indented)
            if not line.startswith(' ') and not line.startswith('\t'):
                return True

    return False


def inject_entry_point(code: str, func_name: str) -> str:
    """
    Inject entry point if missing.

    Many LLM-generated codes define the solution function but never call it.
    This adds an if __name__ == "__main__": func_name() block at the end.
    """
    if not func_name:
        return code

    if has_entry_point(code, func_name):
        return code

    # Add entry point at end
    entry_code = f'''

if __name__ == "__main__":
    {func_name}()
'''
    return code + entry_code


@dataclass
class AnalysisResult:
    """Result of analyzing a single code sample"""
    prompt_id: str
    model: str

    # Compilation
    compiles: bool = False
    syntax_error: Optional[str] = None

    # Semantic (test execution)
    has_semantic_error: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    test_error: Optional[str] = None

    # Security
    has_security_issues: bool = False
    security_issues: List[Dict] = field(default_factory=list)
    security_severity: str = "none"  # none, low, medium, high

    def to_dict(self) -> Dict:
        return {
            "prompt_id": self.prompt_id,
            "model": self.model,
            "compilation": {
                "compiles": self.compiles,
                "error": self.syntax_error,
            },
            "test_execution": {
                "has_semantic_error": self.has_semantic_error,
                "passed": self.tests_passed,
                "failed": self.tests_failed,
                "total_tests": self.tests_total,
                "error": self.test_error,
            },
            "security": {
                "has_security_issues": self.has_security_issues,
                "issues": self.security_issues,
                "severity": self.security_severity,
            }
        }


class PythonAnalyzer:
    """Analyzes Python code for compilation, semantic, and security issues"""

    def __init__(self, prompts_data: Optional[Dict] = None):
        """
        Initialize analyzer.

        Args:
            prompts_data: Dictionary mapping prompt_id to test cases
        """
        self.prompts_data = prompts_data or {}
        self.bandit_available = self._check_bandit()

    def _check_bandit(self) -> bool:
        """Check if Bandit is installed"""
        try:
            result = subprocess.run(
                ["bandit", "--version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def check_compilation(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if Python code compiles (no syntax errors).

        Args:
            code: Python source code

        Returns:
            Tuple of (compiles, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def _get_function_name_for_prompt(self, prompt_id: str) -> Optional[str]:
        """Get the main function name from the original prompt."""
        if prompt_id in self.prompts_data:
            prompt_info = self.prompts_data[prompt_id]
            original_prompt = prompt_info.get("prompt", "")
            return extract_function_name(original_prompt)
        return None

    def check_semantic(
        self,
        code: str,
        prompt_id: str,
        timeout: int = 5
    ) -> Tuple[bool, int, int, int, Optional[str]]:
        """
        Check for semantic errors by running test cases.

        Args:
            code: Python source code
            prompt_id: ID to look up test cases
            timeout: Execution timeout per test

        Returns:
            Tuple of (has_error, passed, failed, total, error_msg)
        """
        # Get test cases for this prompt
        test_cases = self._get_test_cases(prompt_id)

        if not test_cases:
            # No test cases available - can't determine semantic correctness
            return False, 0, 0, 0, "No test cases available"

        # Get function name to inject entry point if needed
        func_name = self._get_function_name_for_prompt(prompt_id)

        passed = 0
        failed = 0
        error_msg = None

        for i, (test_input, expected_output) in enumerate(test_cases):
            try:
                result = self._run_code(code, test_input, timeout, func_name)

                # Compare output (normalize whitespace)
                result_normalized = result.strip()
                expected_normalized = expected_output.strip()

                if result_normalized == expected_normalized:
                    passed += 1
                else:
                    failed += 1
                    if error_msg is None:
                        error_msg = f"Test {i+1}: expected '{expected_normalized[:50]}...', got '{result_normalized[:50]}...'"

            except subprocess.TimeoutExpired:
                failed += 1
                if error_msg is None:
                    error_msg = f"Test {i+1}: timeout"
            except Exception as e:
                failed += 1
                if error_msg is None:
                    error_msg = f"Test {i+1}: {str(e)[:100]}"

        total = passed + failed
        has_error = failed > 0

        return has_error, passed, failed, total, error_msg

    def _get_test_cases(self, prompt_id: str) -> List[Tuple[str, str]]:
        """Get test cases for a prompt from cached data"""
        if prompt_id in self.prompts_data:
            prompt_info = self.prompts_data[prompt_id]

            # Try different structures
            # Structure 1: test_cases.inputs / test_cases.outputs
            test_cases = prompt_info.get("test_cases", {})
            if isinstance(test_cases, dict):
                inputs = test_cases.get("inputs", [])
                outputs = test_cases.get("outputs", [])
                if inputs and outputs:
                    return list(zip(inputs[:5], outputs[:5]))

            # Structure 2: direct inputs/outputs
            inputs = prompt_info.get("inputs", [])
            outputs = prompt_info.get("outputs", [])
            if inputs and outputs:
                return list(zip(inputs[:5], outputs[:5]))

        return []

    def _run_code(
        self,
        code: str,
        test_input: str,
        timeout: int,
        func_name: Optional[str] = None
    ) -> str:
        """
        Run code with input and return output.

        If func_name is provided and the code doesn't have an entry point,
        automatically inject one to call the function.
        """
        # Inject entry point if function name is known and code lacks one
        if func_name:
            code = inject_entry_point(code, func_name)

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr[:200])

            return result.stdout

        finally:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    def check_security(self, code: str) -> Tuple[bool, List[Dict], str]:
        """
        Check for security issues using Bandit.

        Args:
            code: Python source code

        Returns:
            Tuple of (has_issues, issues_list, max_severity)
        """
        if not self.bandit_available:
            return False, [], "none"

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["bandit", "-f", "json", "-q", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout:
                bandit_output = json.loads(result.stdout)
                results = bandit_output.get("results", [])

                if results:
                    issues = []
                    max_severity = "low"

                    for r in results:
                        severity = r.get("issue_severity", "LOW").lower()
                        issues.append({
                            "rule_id": r.get("test_id", ""),
                            "rule_name": r.get("test_name", ""),
                            "severity": severity,
                            "confidence": r.get("issue_confidence", ""),
                            "message": r.get("issue_text", ""),
                            "line": r.get("line_number", 0),
                        })

                        if severity == "high":
                            max_severity = "high"
                        elif severity == "medium" and max_severity != "high":
                            max_severity = "medium"

                    return True, issues, max_severity

            return False, [], "none"

        except subprocess.TimeoutExpired:
            return False, [], "none"
        except json.JSONDecodeError:
            return False, [], "none"
        except Exception:
            return False, [], "none"
        finally:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    def analyze(self, code: str, prompt_id: str, model: str) -> AnalysisResult:
        """
        Perform full analysis on code.

        Args:
            code: Python source code
            prompt_id: Prompt identifier
            model: Model name

        Returns:
            AnalysisResult with all metrics
        """
        result = AnalysisResult(prompt_id=prompt_id, model=model)

        # 1. Check compilation
        compiles, syntax_error = self.check_compilation(code)
        result.compiles = compiles
        result.syntax_error = syntax_error

        # 2. Check semantic (only if compiles)
        if compiles:
            has_error, passed, failed, total, test_error = self.check_semantic(
                code, prompt_id
            )
            result.has_semantic_error = has_error
            result.tests_passed = passed
            result.tests_failed = failed
            result.tests_total = total
            result.test_error = test_error
        else:
            result.has_semantic_error = True  # Can't run = semantic error

        # 3. Check security (even if doesn't compile)
        has_issues, issues, severity = self.check_security(code)
        result.has_security_issues = has_issues
        result.security_issues = issues
        result.security_severity = severity

        return result


def analyze_sample(args: Tuple[Path, str, Dict]) -> Optional[AnalysisResult]:
    """Analyze a single sample (for parallel processing)"""
    sample_dir, model, prompts_data = args

    try:
        # Read generated code
        code_file = sample_dir / "generated_code.py"
        if not code_file.exists():
            return None

        with open(code_file) as f:
            code = f.read()

        # Get prompt_id from directory name
        prompt_id = sample_dir.name

        # Analyze
        analyzer = PythonAnalyzer(prompts_data)
        result = analyzer.analyze(code, prompt_id, model)

        # Save analysis result
        analysis_file = sample_dir / "analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    except Exception as e:
        print(f"Error analyzing {sample_dir}: {e}")
        return None


def load_prompts_data(results_dir: Path) -> Dict:
    """Load prompts data with test cases"""
    prompts_file = results_dir / "prompts.json"

    if prompts_file.exists():
        try:
            with open(prompts_file) as f:
                data = json.load(f)

            # Index by prompt_id
            indexed = {}
            if isinstance(data, list):
                for item in data:
                    pid = item.get("id") or item.get("prompt_id")
                    if pid:
                        indexed[f"apps_plus_{pid}"] = item
                        indexed[pid] = item
            elif isinstance(data, dict):
                indexed = data

            return indexed
        except Exception as e:
            print(f"Warning: Could not load prompts.json: {e}")

    return {}


def analyze_model(
    results_dir: Path,
    model: str,
    prompts_data: Dict,
    max_workers: int = 4,
    limit: Optional[int] = None
) -> List[AnalysisResult]:
    """Analyze all samples for a model"""

    model_dir = results_dir / model / "apps_plus"

    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return []

    # Get all sample directories
    sample_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

    if limit:
        sample_dirs = sample_dirs[:limit]

    print(f"\nAnalyzing {model}: {len(sample_dirs)} samples")

    results = []

    # Prepare arguments for parallel processing
    args_list = [(d, model, prompts_data) for d in sample_dirs]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_sample, args): args[0]
            for args in args_list
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 500 == 0:
                print(f"  Progress: {completed}/{len(sample_dirs)}")

            result = future.result()
            if result:
                results.append(result)

    return results


def compute_metrics(results: List[AnalysisResult]) -> Dict:
    """Compute aggregate metrics from analysis results"""

    if not results:
        return {}

    total = len(results)

    # Compilation errors
    compilation_errors = sum(1 for r in results if not r.compiles)

    # Semantic errors (among those that compile)
    compiling = [r for r in results if r.compiles]
    semantic_errors = sum(1 for r in compiling if r.has_semantic_error)

    # Security issues
    security_issues = sum(1 for r in results if r.has_security_issues)
    high_severity = sum(1 for r in results if r.security_severity == "high")
    medium_severity = sum(1 for r in results if r.security_severity == "medium")

    return {
        "total_samples": total,
        "compilation": {
            "errors": compilation_errors,
            "error_rate": compilation_errors / total * 100,
            "success_rate": (total - compilation_errors) / total * 100,
        },
        "semantic": {
            "errors": semantic_errors,
            "error_rate": semantic_errors / len(compiling) * 100 if compiling else 0,
            "compiling_samples": len(compiling),
        },
        "security": {
            "issues_count": security_issues,
            "issue_rate": security_issues / total * 100,
            "high_severity": high_severity,
            "medium_severity": medium_severity,
        }
    }


def print_metrics(model: str, metrics: Dict):
    """Print metrics in a nice format"""
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")
    print(f"Total samples: {metrics['total_samples']}")
    print()
    print(f"1. Compilation Errors: {metrics['compilation']['errors']} ({metrics['compilation']['error_rate']:.2f}%)")
    print(f"2. Semantic Errors: {metrics['semantic']['errors']} ({metrics['semantic']['error_rate']:.2f}% of compiling)")
    print(f"3. Security Issues: {metrics['security']['issues_count']} ({metrics['security']['issue_rate']:.2f}%)")
    print(f"   - High severity: {metrics['security']['high_severity']}")
    print(f"   - Medium severity: {metrics['security']['medium_severity']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./benchmark/full_results",
        help="Path to benchmark results directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to analyze (default: all)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit samples per model (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./analysis_results.json",
        help="Output file for metrics"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    # Check for Bandit
    try:
        subprocess.run(["bandit", "--version"], capture_output=True, check=True)
        print("✓ Bandit available for security analysis")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("⚠ Bandit not installed. Install with: pip install bandit")
        print("  Security analysis will be skipped.")

    # Load prompts data
    print("\nLoading prompts data...")
    prompts_data = load_prompts_data(results_dir)
    print(f"Loaded {len(prompts_data)} prompts with test cases")

    # Find models to analyze
    if args.model:
        models = [args.model]
    else:
        models = [
            d.name for d in results_dir.iterdir()
            if d.is_dir() and d.name not in ['dataset_cache', 'checkpoints']
        ]

    print(f"\nModels to analyze: {models}")

    # Analyze each model
    all_metrics = {}

    for model in models:
        results = analyze_model(
            results_dir,
            model,
            prompts_data,
            max_workers=args.workers,
            limit=args.limit
        )

        if results:
            metrics = compute_metrics(results)
            all_metrics[model] = metrics
            print_metrics(model, metrics)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n\nMetrics saved to: {output_path}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Compile Err%':<15} {'Semantic Err%':<15} {'Security%':<15}")
    print("-"*70)

    for model, m in all_metrics.items():
        print(f"{model:<15} {m['compilation']['error_rate']:<15.2f} {m['semantic']['error_rate']:<15.2f} {m['security']['issue_rate']:<15.2f}")

    return 0


if __name__ == "__main__":
    exit(main())
