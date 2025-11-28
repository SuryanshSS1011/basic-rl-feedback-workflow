#!/usr/bin/env python3
"""
Batch analysis pipeline for benchmarking.
Analyzes generated code for:
1. Compilation errors (syntax/compilation issues)
2. Semantic errors (test case failures - wrong output, timeout, crash, etc.)
3. Security issues (CodeQL/Bandit findings)

Supports multi-language analysis (Python and C/C++).
"""

import json
import os
import py_compile
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import sys

# Import benchmark modules
from .language_detector import LanguageDetector
from .code_cleaner import CodeCleaner
from .test_executor import TestExecutor, FailureType


class CodeAnalyzer:
    """Analyze generated code for compilation, semantic, and security issues."""

    def __init__(
        self,
        results_dir: str = "./benchmark/results",
        codeql_path: Optional[str] = None,
        test_timeout: int = 5,
        max_tests_per_sample: int = 10
    ):
        self.results_dir = Path(results_dir)
        self.username = os.getlogin()

        # Initialize components
        self.language_detector = LanguageDetector()
        self.code_cleaner = CodeCleaner()
        self.test_executor = TestExecutor(timeout_seconds=test_timeout)
        self.max_tests = max_tests_per_sample

        # Auto-detect CodeQL path
        if codeql_path is None:
            self.codeql_path = f"/scratch/{self.username}/codeql/codeql"
        else:
            self.codeql_path = codeql_path

        # Check if CodeQL is available
        self.codeql_available = os.path.exists(self.codeql_path)
        if not self.codeql_available:
            print(f"Warning: CodeQL not found at {self.codeql_path}")
            print("C/C++ security analysis will be skipped")

        # Check if Bandit is available for Python security analysis
        self.bandit_available = shutil.which('bandit') is not None
        if not self.bandit_available:
            print("Warning: Bandit not found in PATH")
            print("Python security analysis will be skipped")

    def check_compilation(self, code_path: Path) -> Dict:
        """
        Check if code compiles successfully.

        Returns:
            Dictionary with compilation status and errors
        """
        result = {
            'compiles': False,
            'errors': [],
            'warnings': [],
            'return_code': None
        }

        try:
            # Try to compile with gcc
            compile_result = subprocess.run(
                ['gcc', '-c', '-Wall', '-Wextra', str(code_path), '-o', '/dev/null'],
                capture_output=True,
                text=True,
                timeout=30
            )

            result['return_code'] = compile_result.returncode
            result['compiles'] = (compile_result.returncode == 0)

            # Parse errors and warnings from stderr
            if compile_result.stderr:
                stderr_lines = compile_result.stderr.strip().split('\n')
                for line in stderr_lines:
                    if 'error:' in line.lower():
                        result['errors'].append(line)
                    elif 'warning:' in line.lower():
                        result['warnings'].append(line)

        except subprocess.TimeoutExpired:
            result['errors'].append("Compilation timeout (>30s)")
        except Exception as e:
            result['errors'].append(f"Compilation exception: {str(e)}")

        return result

    def check_python_syntax(self, code_path: Path) -> Dict:
        """
        Check Python syntax using py_compile.

        Returns:
            Dictionary with syntax check status
        """
        result = {
            'compiles': False,
            'errors': [],
            'warnings': [],
            'return_code': None
        }

        try:
            py_compile.compile(str(code_path), doraise=True)
            result['compiles'] = True
            result['return_code'] = 0
        except py_compile.PyCompileError as e:
            result['errors'].append(str(e))
            result['return_code'] = 1
        except SyntaxError as e:
            result['errors'].append(f"SyntaxError: {e}")
            result['return_code'] = 1
        except Exception as e:
            result['errors'].append(f"Error: {e}")
            result['return_code'] = 1

        return result

    def run_bandit_analysis(self, code_path: Path) -> Dict:
        """
        Run Bandit security scanner for Python code.

        Returns:
            Dictionary with security findings
        """
        result = {
            'success': False,
            'security_issues': [],
            'total_findings': 0,
            'error': None
        }

        if not self.bandit_available:
            result['error'] = "Bandit not available"
            return result

        try:
            # Run bandit with JSON output
            bandit_result = subprocess.run(
                ['bandit', '-f', 'json', '-r', str(code_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Bandit returns 0 for no issues, 1 for issues found
            output = bandit_result.stdout
            if output:
                data = json.loads(output)
                for finding in data.get('results', []):
                    result['security_issues'].append({
                        'rule_id': finding.get('test_id', 'unknown'),
                        'level': finding.get('issue_severity', 'MEDIUM'),
                        'message': finding.get('issue_text', ''),
                        'confidence': finding.get('issue_confidence', 'MEDIUM'),
                        'line': finding.get('line_number', 0)
                    })

            result['total_findings'] = len(result['security_issues'])
            result['success'] = True

        except subprocess.TimeoutExpired:
            result['error'] = "Bandit timeout"
        except json.JSONDecodeError as e:
            result['error'] = f"Failed to parse Bandit output: {e}"
        except Exception as e:
            result['error'] = f"Bandit error: {e}"

        return result

    def run_codeql_analysis(self, code_path: Path, output_dir: Path) -> Dict:
        """
        Run CodeQL security and quality analysis.

        Returns:
            Dictionary with CodeQL analysis results
        """
        result = {
            'success': False,
            'security_issues': [],
            'semantic_issues': [],
            'total_findings': 0,
            'error': None
        }

        if not self.codeql_available:
            result['error'] = "CodeQL not available"
            return result

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create Makefile
            makefile_path = tmpdir / "Makefile"
            code_filename = code_path.name
            binary_name = code_filename.replace('.c', '.out')

            with open(makefile_path, 'w') as f:
                f.write(f"""all: {binary_name}

{binary_name}: {code_filename}
\tgcc -g {code_filename} -o {binary_name}

clean:
\trm -f {binary_name} *.bc

.PHONY: all clean
""")

            # Copy code file
            shutil.copy(code_path, tmpdir / code_filename)

            # Create CodeQL database
            db_path = tmpdir / "codeql_db"
            create_result = subprocess.run(
                [
                    self.codeql_path, "database", "create", str(db_path),
                    f"--source-root={tmpdir}",
                    "--language=cpp",
                    "--command=make",
                    "--overwrite"
                ],
                capture_output=True,
                text=True,
                timeout=120
            )

            if create_result.returncode != 0:
                result['error'] = f"Database creation failed: {create_result.stderr}"
                return result

            # Run CodeQL analysis
            sarif_path = output_dir / "codeql_results.sarif"
            analyze_result = subprocess.run(
                [
                    self.codeql_path, "database", "analyze", str(db_path),
                    "codeql/cpp-queries:codeql-suites/cpp-security-and-quality.qls",
                    "--format=sarif-latest",
                    f"--output={sarif_path}"
                ],
                capture_output=True,
                text=True,
                timeout=180
            )

            if analyze_result.returncode != 0:
                result['error'] = f"Analysis failed: {analyze_result.stderr}"
                return result

            # Parse SARIF results
            try:
                with open(sarif_path, 'r') as f:
                    sarif_data = json.load(f)

                for run in sarif_data.get("runs", []):
                    for finding in run.get("results", []):
                        rule_id = finding.get("ruleId", "unknown")
                        level = finding.get("level", "warning")
                        message = finding.get("message", {}).get("text", "")

                        finding_info = {
                            'rule_id': rule_id,
                            'level': level,
                            'message': message
                        }

                        # Categorize as security or semantic issue
                        if 'security' in rule_id.lower() or level == 'error':
                            result['security_issues'].append(finding_info)
                        else:
                            result['semantic_issues'].append(finding_info)

                result['total_findings'] = len(result['security_issues']) + len(result['semantic_issues'])
                result['success'] = True

            except Exception as e:
                result['error'] = f"Failed to parse SARIF: {str(e)}"

        return result

    def analyze_generated_code(
        self,
        model: str,
        dataset: str,
        prompt_id: str,
        prompt_data: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze a single generated code sample with multi-language support.

        Args:
            model: Model name
            dataset: Dataset name
            prompt_id: Prompt identifier
            prompt_data: Original prompt data with test cases (optional)

        Returns:
            Complete analysis results including semantic error tracking
        """
        code_dir = self.results_dir / model / dataset / prompt_id

        # Try to find the code file (could be .c or .py depending on language)
        code_path = None
        for ext in ['.c', '.py', '.cpp']:
            potential_path = code_dir / f"generated_code{ext}"
            if potential_path.exists():
                code_path = potential_path
                break

        # Fallback to .c if nothing found
        if code_path is None:
            code_path = code_dir / "generated_code.c"

        result = {
            'model': model,
            'dataset': dataset,
            'prompt_id': prompt_id,
            'code_exists': False,
            'detected_language': None,
            'compilation': None,
            'test_execution': None,
            'security': None,
            'has_semantic_error': None,
            'failure_breakdown': {}
        }

        # Check if code file exists
        if not code_path.exists():
            result['error'] = "Generated code file not found"
            return result

        result['code_exists'] = True

        # Read and detect language
        with open(code_path, 'r') as f:
            raw_code = f.read()

        # Get expected language from prompt data if available
        expected_lang = prompt_data.get('language') if prompt_data else None
        detected_lang = self.language_detector.detect(raw_code)

        # Use expected language if provided, otherwise use detected
        language = expected_lang if expected_lang else detected_lang
        result['detected_language'] = language

        # Clean the code
        file_ext = '.py' if language == 'python' else '.c'
        cleaned_path = code_dir / f"clean_code{file_ext}"
        try:
            cleaned_code, _ = self.code_cleaner.clean(raw_code, language)
            with open(cleaned_path, 'w') as f:
                f.write(cleaned_code)
        except Exception as e:
            # Use original if cleaning fails
            cleaned_path = code_path
            result['cleaning_error'] = str(e)

        # Step 1: Check compilation/syntax (language-aware)
        if language == 'python':
            result['compilation'] = self.check_python_syntax(cleaned_path)
        else:
            result['compilation'] = self.check_compilation(cleaned_path)

        # Step 2: Run test cases (semantic error detection)
        test_cases = prompt_data.get('test_cases') if prompt_data else None
        if result['compilation']['compiles'] and test_cases:
            inputs = test_cases.get('inputs', [])
            outputs = test_cases.get('outputs', [])

            if inputs and outputs:
                test_result = self.test_executor.run_all_tests(
                    cleaned_path,
                    test_cases,
                    language=language,
                    max_tests=self.max_tests
                )

                result['test_execution'] = {
                    'total_tests': test_result.total_tests,
                    'passed': test_result.passed,
                    'failed': test_result.failed,
                    'skipped': test_result.skipped,
                    'skip_reason': test_result.skip_reason,
                    'failure_breakdown': test_result.failure_breakdown,
                    'has_semantic_error': test_result.has_semantic_error
                }
                result['has_semantic_error'] = test_result.has_semantic_error
                result['failure_breakdown'] = test_result.failure_breakdown
            else:
                result['test_execution'] = {
                    'skipped': True,
                    'skip_reason': 'No test cases available',
                    'total_tests': 0,
                    'passed': 0,
                    'failed': 0
                }
        else:
            skip_reason = 'compilation_failure' if not result['compilation']['compiles'] else 'no_test_cases'
            result['test_execution'] = {
                'skipped': True,
                'skip_reason': skip_reason,
                'total_tests': 0,
                'passed': 0,
                'failed': 0
            }

        # Step 3: Security analysis (language-aware)
        if result['compilation']['compiles']:
            if language == 'python':
                result['security'] = self.run_bandit_analysis(cleaned_path)
            else:
                result['security'] = self.run_codeql_analysis(cleaned_path, code_dir)
        else:
            result['security'] = {
                'success': False,
                'error': 'Skipped due to compilation/syntax failure',
                'security_issues': [],
                'total_findings': 0
            }

        # Save analysis results
        analysis_path = code_dir / "analysis_results.json"
        with open(analysis_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def batch_analyze(
        self,
        models: List[str],
        datasets: Optional[List[str]] = None,
        prompts_map: Optional[Dict[str, Dict]] = None
    ) -> List[Dict]:
        """
        Analyze all generated code for specified models.

        Args:
            models: List of model names to analyze
            datasets: Optional list of datasets to filter (None = all)
            prompts_map: Optional mapping of prompt_id -> prompt_data with test cases

        Returns:
            List of analysis results
        """
        all_results = []

        # Find all generated code samples
        for model in models:
            model_dir = self.results_dir / model
            if not model_dir.exists():
                print(f"Warning: No results found for model {model}")
                continue

            # Find all dataset/prompt combinations
            samples = []
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue

                dataset = dataset_dir.name
                if datasets and dataset not in datasets:
                    continue

                for prompt_dir in dataset_dir.iterdir():
                    if not prompt_dir.is_dir():
                        continue

                    prompt_id = prompt_dir.name
                    samples.append((model, dataset, prompt_id))

            # Analyze each sample
            print(f"\nAnalyzing {len(samples)} samples for model: {model}")
            for model, dataset, prompt_id in tqdm(samples, desc=f"Analyzing {model}"):
                # Look up prompt data if available
                prompt_data = None
                if prompts_map:
                    prompt_data = prompts_map.get(prompt_id)

                result = self.analyze_generated_code(model, dataset, prompt_id, prompt_data)
                all_results.append(result)

        return all_results

    def batch_analyze_with_prompts(
        self,
        models: List[str],
        prompts: List[Dict],
        datasets: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Analyze all generated code with test case support.

        Args:
            models: List of model names to analyze
            prompts: List of prompt dictionaries with test cases
            datasets: Optional list of datasets to filter (None = all)

        Returns:
            List of analysis results
        """
        # Build prompts map for quick lookup
        prompts_map = {p['id']: p for p in prompts}

        return self.batch_analyze(models, datasets, prompts_map)


if __name__ == "__main__":
    # Test analyzer
    analyzer = CodeAnalyzer()

    # Check if we have any results to analyze
    results_dir = Path("./benchmark/results")
    if not results_dir.exists() or not any(results_dir.iterdir()):
        print("No results found to analyze")
        print("Run multi_model_runner.py first to generate code")
    else:
        # Find available models
        models = [d.name for d in results_dir.iterdir() if d.is_dir() and d.name != 'summary']
        print(f"Found models: {models}")

        if models:
            # Analyze all results
            results = analyzer.batch_analyze(models)
            print(f"\nAnalysis complete: {len(results)} samples analyzed")

            # Save summary
            summary_path = results_dir / "summary" / "analysis_results.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Summary saved to: {summary_path}")
