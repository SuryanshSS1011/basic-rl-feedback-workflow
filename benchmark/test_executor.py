#!/usr/bin/env python3
"""
Test execution module for semantic error detection.
Executes generated code against test cases and categorizes failures.

Semantic errors are defined as any case where code compiles/parses but fails to solve the task:
- Wrong output: Test case mismatch (actual != expected)
- Timeout/hang: Infinite loops, exceeds time limit
- Runtime crash: Segfault, exceptions, assertion failures
- Memory error: Out of memory, stack overflow
- No output: Program exits without producing required output
"""

import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .language_detector import LanguageDetector


class FailureType(Enum):
    """Types of semantic errors (all represent 'failed to solve task')."""
    NONE = "none"                    # Test passed
    WRONG_OUTPUT = "wrong_output"    # Output doesn't match expected
    TIMEOUT = "timeout"              # Exceeded time limit
    CRASH = "crash"                  # Runtime error/segfault/exception
    MEMORY_ERROR = "memory_error"    # Out of memory/stack overflow
    NO_OUTPUT = "no_output"          # Program produced no output
    COMPILATION_FAILED = "compilation_failed"  # Could not compile/parse


@dataclass
class TestResult:
    """Result of a single test case execution."""
    passed: bool
    failure_type: FailureType
    actual_output: str
    expected_output: str
    error_message: Optional[str]
    execution_time_ms: float
    return_code: Optional[int]


@dataclass
class TestSuiteResult:
    """Aggregated results of all test cases."""
    total_tests: int
    passed: int
    failed: int
    skipped: bool
    skip_reason: Optional[str]
    results: List[TestResult]
    failure_breakdown: Dict[str, int]
    has_semantic_error: bool  # True if ANY test failed


class TestExecutor:
    """Execute code against test cases with multi-language support."""

    def __init__(
        self,
        timeout_seconds: int = 5,
        memory_limit_mb: int = 256,
        max_output_size: int = 10000
    ):
        """
        Initialize test executor.

        Args:
            timeout_seconds: Maximum execution time per test case
            memory_limit_mb: Maximum memory usage in MB
            max_output_size: Maximum output size to capture in characters
        """
        self.timeout = timeout_seconds
        self.memory_limit = memory_limit_mb
        self.max_output_size = max_output_size
        self.language_detector = LanguageDetector()

    def run_test_case(
        self,
        code_path: Path,
        language: str,
        test_input: str,
        expected_output: str
    ) -> TestResult:
        """
        Run a single test case.

        Args:
            code_path: Path to source code or binary
            language: Programming language ('python', 'c')
            test_input: Input to provide via stdin
            expected_output: Expected output

        Returns:
            TestResult with pass/fail status and details
        """
        start_time = time.time()

        try:
            if language == 'python':
                result = self._execute_python(code_path, test_input)
            elif language in ('c', 'cpp'):
                result = self._execute_c(code_path, test_input)
            else:
                return TestResult(
                    passed=False,
                    failure_type=FailureType.CRASH,
                    actual_output="",
                    expected_output=expected_output,
                    error_message=f"Unsupported language: {language}",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    return_code=None
                )

            execution_time = (time.time() - start_time) * 1000

            # Check for various failure types
            failure_type, passed = self._categorize_result(
                result, expected_output
            )

            return TestResult(
                passed=passed,
                failure_type=failure_type,
                actual_output=result.get('stdout', '')[:self.max_output_size],
                expected_output=expected_output,
                error_message=result.get('error'),
                execution_time_ms=execution_time,
                return_code=result.get('return_code')
            )

        except Exception as e:
            return TestResult(
                passed=False,
                failure_type=FailureType.CRASH,
                actual_output="",
                expected_output=expected_output,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                return_code=None
            )

    def _execute_python(
        self,
        code_path: Path,
        test_input: str
    ) -> Dict:
        """Execute Python code."""
        try:
            result = subprocess.run(
                ['python3', str(code_path)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'error': result.stderr if result.returncode != 0 else None
            }

        except subprocess.TimeoutExpired:
            return {
                'stdout': '',
                'stderr': '',
                'return_code': -1,
                'error': 'timeout',
                'timeout': True
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'return_code': -1,
                'error': str(e)
            }

    def _execute_c(
        self,
        code_path: Path,
        test_input: str
    ) -> Dict:
        """Compile and execute C/C++ code."""
        # First, compile the code
        binary_path = code_path.with_suffix('.out')

        try:
            # Compile
            compile_result = subprocess.run(
                ['gcc', '-o', str(binary_path), str(code_path), '-lm'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if compile_result.returncode != 0:
                return {
                    'stdout': '',
                    'stderr': compile_result.stderr,
                    'return_code': compile_result.returncode,
                    'error': 'compilation_failed',
                    'compilation_failed': True
                }

            # Execute
            result = subprocess.run(
                [str(binary_path)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Check for common crash signals
            error = None
            if result.returncode < 0:
                sig = -result.returncode
                if sig == signal.SIGSEGV:
                    error = 'segmentation_fault'
                elif sig == signal.SIGFPE:
                    error = 'floating_point_exception'
                elif sig == signal.SIGABRT:
                    error = 'abort'
                else:
                    error = f'signal_{sig}'

            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'error': error
            }

        except subprocess.TimeoutExpired:
            return {
                'stdout': '',
                'stderr': '',
                'return_code': -1,
                'error': 'timeout',
                'timeout': True
            }
        except MemoryError:
            return {
                'stdout': '',
                'stderr': 'Out of memory',
                'return_code': -1,
                'error': 'memory_error'
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'return_code': -1,
                'error': str(e)
            }
        finally:
            # Clean up binary
            if binary_path.exists():
                try:
                    binary_path.unlink()
                except:
                    pass

    def _categorize_result(
        self,
        result: Dict,
        expected_output: str
    ) -> Tuple[FailureType, bool]:
        """
        Categorize execution result into failure type.

        Returns:
            Tuple of (FailureType, passed)
        """
        # Check for compilation failure
        if result.get('compilation_failed'):
            return FailureType.COMPILATION_FAILED, False

        # Check for timeout
        if result.get('timeout'):
            return FailureType.TIMEOUT, False

        # Check for memory error
        error = result.get('error', '')
        if error and any(mem in error.lower() for mem in ['memory', 'oom', 'stack overflow']):
            return FailureType.MEMORY_ERROR, False

        # Check for crash (non-zero return code with error signals)
        if result.get('return_code', 0) < 0:
            return FailureType.CRASH, False

        if result.get('return_code', 0) != 0 and result.get('error'):
            # Could be crash or runtime error
            if any(crash in str(error).lower() for crash in [
                'segfault', 'segmentation', 'abort', 'exception', 'error', 'traceback'
            ]):
                return FailureType.CRASH, False

        # Check for no output
        stdout = result.get('stdout', '').strip()
        if not stdout:
            return FailureType.NO_OUTPUT, False

        # Check output correctness
        if self._compare_output(stdout, expected_output):
            return FailureType.NONE, True
        else:
            return FailureType.WRONG_OUTPUT, False

    def _compare_output(self, actual: str, expected: str) -> bool:
        """
        Compare actual and expected output with normalization.

        Handles:
        - Whitespace differences
        - Trailing newlines
        - Windows vs Unix line endings
        """
        # Normalize both outputs
        actual_normalized = self._normalize_output(actual)
        expected_normalized = self._normalize_output(expected)

        return actual_normalized == expected_normalized

    def _normalize_output(self, output: str) -> str:
        """Normalize output for comparison."""
        if not output:
            return ""

        # Convert Windows line endings
        output = output.replace('\r\n', '\n')

        # Split into lines, strip each, remove empty trailing lines
        lines = output.split('\n')
        lines = [line.strip() for line in lines]

        # Remove trailing empty lines
        while lines and not lines[-1]:
            lines.pop()

        return '\n'.join(lines)

    def run_all_tests(
        self,
        code_path: Path,
        test_cases: Dict,
        language: Optional[str] = None,
        max_tests: Optional[int] = None
    ) -> TestSuiteResult:
        """
        Run all test cases for a code sample.

        Args:
            code_path: Path to source code
            test_cases: Dict with 'inputs' and 'outputs' lists
            language: Language (auto-detected if not provided)
            max_tests: Maximum number of tests to run

        Returns:
            TestSuiteResult with aggregated results
        """
        # Extract inputs and outputs
        inputs = test_cases.get('inputs', [])
        outputs = test_cases.get('outputs', [])

        # Handle empty test cases
        if not inputs or not outputs:
            return TestSuiteResult(
                total_tests=0,
                passed=0,
                failed=0,
                skipped=True,
                skip_reason="No test cases available",
                results=[],
                failure_breakdown={},
                has_semantic_error=False
            )

        # Ensure equal length
        num_tests = min(len(inputs), len(outputs))
        if max_tests:
            num_tests = min(num_tests, max_tests)

        # Detect language if not provided
        if not language:
            with open(code_path, 'r') as f:
                code = f.read()
            language = self.language_detector.detect(code)

        # Run each test
        results = []
        failure_counts = {ft.value: 0 for ft in FailureType}

        for i in range(num_tests):
            test_input = inputs[i] if i < len(inputs) else ""
            expected_output = outputs[i] if i < len(outputs) else ""

            result = self.run_test_case(
                code_path, language, test_input, expected_output
            )
            results.append(result)

            failure_counts[result.failure_type.value] += 1

        # Aggregate results
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        # Remove zero counts from breakdown
        failure_breakdown = {k: v for k, v in failure_counts.items() if v > 0}

        return TestSuiteResult(
            total_tests=len(results),
            passed=passed,
            failed=failed,
            skipped=False,
            skip_reason=None,
            results=results,
            failure_breakdown=failure_breakdown,
            has_semantic_error=(failed > 0)
        )


def run_tests(
    code_path: Path,
    test_cases: Dict,
    language: Optional[str] = None,
    timeout: int = 5
) -> TestSuiteResult:
    """
    Convenience function to run tests.

    Args:
        code_path: Path to source code
        test_cases: Dict with 'inputs' and 'outputs' lists
        language: Language (auto-detected if not provided)
        timeout: Timeout per test in seconds

    Returns:
        TestSuiteResult
    """
    executor = TestExecutor(timeout_seconds=timeout)
    return executor.run_all_tests(code_path, test_cases, language)


if __name__ == "__main__":
    # Test the executor
    import tempfile

    # Create a test Python file
    python_code = """
n = int(input())
result = 1
for i in range(1, n + 1):
    result *= i
print(result)
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_code)
        test_file = Path(f.name)

    test_cases = {
        'inputs': ['5\n', '0\n', '10\n'],
        'outputs': ['120\n', '1\n', '3628800\n']
    }

    executor = TestExecutor()
    result = executor.run_all_tests(test_file, test_cases, language='python')

    print("Test Execution Results")
    print("=" * 50)
    print(f"Total tests: {result.total_tests}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    print(f"Has semantic error: {result.has_semantic_error}")
    print(f"Failure breakdown: {result.failure_breakdown}")

    for i, test_result in enumerate(result.results):
        print(f"\nTest {i + 1}:")
        print(f"  Passed: {test_result.passed}")
        print(f"  Failure type: {test_result.failure_type.value}")
        print(f"  Execution time: {test_result.execution_time_ms:.2f}ms")

    # Clean up
    test_file.unlink()
