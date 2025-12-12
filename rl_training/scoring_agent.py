"""
Hybrid Scoring Agent (πϕ)

Pure hybrid scorer that uses:
1. Deterministic rules for objective metrics (compilation, tests, KLEE)
2. Zero-shot prompted LLM for subjective security interpretation

NO separate learned reward model - uses rules + prompted LLM only.
"""

import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .security_weights import SecurityWeights
from .config import RewardConfig


@dataclass
class CompilationResult:
    """Result of compilation check"""
    compiles: bool
    errors: List[str]
    warnings: List[str]
    error_count: int
    warning_count: int


@dataclass
class TestResult:
    """Result of test execution"""
    passed: int
    failed: int
    total: int
    has_semantic_error: bool
    failure_types: Dict[str, int]  # wrong_output, timeout, crash, etc.


@dataclass
class SecurityResult:
    """Result of security analysis"""
    findings: List[Dict]
    severity_score: float
    llm_interpretation: Optional[str] = None


@dataclass
class ScoringResult:
    """Complete scoring result"""
    compilation: CompilationResult
    tests: TestResult
    security: SecurityResult
    klee_bugs: List[Dict]
    rfunc: float
    rsec: float
    severity_v: float


class RulesEngine:
    """
    Deterministic rules for objective metrics.
    Parses syntax errors, test results, and security findings.

    Supports both Python and C/C++ code analysis.
    """

    # GCC error patterns (for C/C++ - deprecated for Python)
    GCC_ERROR_PATTERN = re.compile(r'^(.+):(\d+):(\d+): error: (.+)$', re.MULTILINE)
    GCC_WARNING_PATTERN = re.compile(r'^(.+):(\d+):(\d+): warning: (.+)$', re.MULTILINE)

    # KLEE bug patterns (C-specific - not applicable to Python)
    KLEE_ERROR_PATTERNS = {
        'ptr': re.compile(r'(memory error|ptr|pointer)', re.IGNORECASE),
        'free': re.compile(r'(double free|invalid free|free)', re.IGNORECASE),
        'overflow': re.compile(r'(overflow|out of bound)', re.IGNORECASE),
        'div': re.compile(r'(division by zero|div)', re.IGNORECASE),
        'assert': re.compile(r'(assertion|assert)', re.IGNORECASE),
        'abort': re.compile(r'(abort|aborted)', re.IGNORECASE),
    }

    def parse_python_syntax(self, code: str) -> CompilationResult:
        """
        Check Python code syntax validity using AST parsing.

        Args:
            code: Python source code string

        Returns:
            CompilationResult with syntax validity status
        """
        import ast

        if not code or len(code.strip()) < 5:
            return CompilationResult(
                compiles=False,
                errors=["Empty or too short code"],
                warnings=[],
                error_count=1,
                warning_count=0,
            )

        try:
            ast.parse(code)
            return CompilationResult(
                compiles=True,
                errors=[],
                warnings=[],
                error_count=0,
                warning_count=0,
            )
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}" if e.lineno else str(e.msg)
            return CompilationResult(
                compiles=False,
                errors=[error_msg],
                warnings=[],
                error_count=1,
                warning_count=0,
            )
        except Exception as e:
            return CompilationResult(
                compiles=False,
                errors=[str(e)],
                warnings=[],
                error_count=1,
                warning_count=0,
            )

    def parse_gcc_output(self, output: str) -> CompilationResult:
        """
        Parse GCC compiler output to extract errors and warnings.

        Args:
            output: Raw GCC output string

        Returns:
            CompilationResult with parsed errors and warnings
        """
        errors = []
        warnings = []

        for match in self.GCC_ERROR_PATTERN.finditer(output):
            errors.append({
                'file': match.group(1),
                'line': int(match.group(2)),
                'column': int(match.group(3)),
                'message': match.group(4),
            })

        for match in self.GCC_WARNING_PATTERN.finditer(output):
            warnings.append({
                'file': match.group(1),
                'line': int(match.group(2)),
                'column': int(match.group(3)),
                'message': match.group(4),
            })

        # Check for compilation failure indicators
        compiles = len(errors) == 0 and 'error:' not in output.lower()

        return CompilationResult(
            compiles=compiles,
            errors=[e['message'] for e in errors],
            warnings=[w['message'] for w in warnings],
            error_count=len(errors),
            warning_count=len(warnings),
        )

    def parse_test_results(self, test_output: Dict) -> TestResult:
        """
        Parse test execution results.

        Args:
            test_output: Dictionary with test execution info

        Returns:
            TestResult with pass/fail counts
        """
        passed = test_output.get('passed', 0)
        failed = test_output.get('failed', 0)
        total = test_output.get('total_tests', test_output.get('total', passed + failed))

        has_semantic_error = test_output.get('has_semantic_error', failed > 0)

        # Parse failure types
        failure_types = {}
        for key in ['wrong_output', 'timeout', 'crash', 'memory_error', 'no_output']:
            count = test_output.get(key, 0)
            if count > 0:
                failure_types[key] = count

        return TestResult(
            passed=passed,
            failed=failed,
            total=total if total > 0 else (passed + failed),
            has_semantic_error=has_semantic_error,
            failure_types=failure_types,
        )

    def parse_klee_output(self, klee_output: str) -> List[Dict]:
        """
        Parse KLEE symbolic execution output for bugs.

        Args:
            klee_output: Raw KLEE output string

        Returns:
            List of detected bugs with type and details
        """
        bugs = []

        for line in klee_output.split('\n'):
            for bug_type, pattern in self.KLEE_ERROR_PATTERNS.items():
                if pattern.search(line):
                    bugs.append({
                        'type': bug_type,
                        'message': line.strip(),
                    })
                    break

        return bugs

    def compute_rfunc(
        self,
        compilation: CompilationResult,
        tests: TestResult
    ) -> float:
        """
        Compute functional correctness reward using rules.

        Rfunc = passed / total (or 0 if syntax invalid)

        Args:
            compilation: Compilation/syntax result
            tests: Test result

        Returns:
            Rfunc in [0, 1]
        """
        if not compilation.compiles:
            return 0.0

        if tests.total == 0:
            # No tests - assume correct if syntax is valid
            return 1.0

        return tests.passed / tests.total

    def check_python_security(self, code: str) -> List[Dict]:
        """
        Check Python code for security issues using regex patterns.

        Args:
            code: Python source code

        Returns:
            List of security findings
        """
        findings = []

        # Dangerous Python patterns
        patterns = [
            (r'\beval\s*\(', 'python/code-injection', 'Use of eval() - code injection risk', 'high'),
            (r'\bexec\s*\(', 'python/code-injection', 'Use of exec() - code injection risk', 'high'),
            (r'\b__import__\s*\(', 'python/dynamic-import', 'Dynamic import - potential security risk', 'medium'),
            (r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True',
             'python/shell-injection', 'Shell injection risk', 'high'),
            (r'\bos\.system\s*\(', 'python/command-injection', 'Command injection via os.system', 'high'),
            (r'\bos\.popen\s*\(', 'python/command-injection', 'Command injection via os.popen', 'high'),
            (r'\bpickle\.loads?\s*\(', 'python/unsafe-deserialization', 'Unsafe pickle deserialization', 'high'),
            (r'\byaml\.load\s*\([^)]*\)', 'python/unsafe-yaml', 'Unsafe YAML load (use safe_load)', 'medium'),
            (r'\binput\s*\(\s*\)', 'python/user-input', 'Unvalidated user input', 'low'),
            (r'open\s*\([^)]*["\'][wa]["\']', 'python/file-write', 'File write operation', 'low'),
        ]

        for pattern, rule_id, message, severity in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                findings.append({
                    'rule_id': rule_id,
                    'message': message,
                    'severity': severity,
                })

        return findings


class PromptedLLMScorer:
    """
    Uses a small LLM with zero-shot prompting to interpret security findings.
    NO training - just prompted inference.
    """

    SECURITY_INTERPRETATION_PROMPT = """You are a security expert analyzing code vulnerabilities.
Given the following security findings from static analysis, provide a severity assessment.

Code context:
```
{code_snippet}
```

Security findings:
{findings}

Rate the overall security severity on a scale of 0.0 to 1.0, where:
- 0.0 = No security concerns
- 0.3 = Low severity (code quality issues)
- 0.5 = Medium severity (potential vulnerabilities)
- 0.7 = High severity (likely exploitable)
- 1.0 = Critical (immediate security risk)

Consider:
1. Are these findings exploitable in context?
2. What's the potential impact?
3. Are there mitigating factors in the code?

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
        device: str = "cuda"
    ):
        """
        Initialize LLM scorer.

        Args:
            model_name: HuggingFace model path
            device: Device to use
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Lazy load the model"""
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        except ImportError:
            raise ImportError(
                "transformers required for LLM scoring. "
                "Install with: pip install transformers"
            )

    def interpret_security_findings(
        self,
        code_snippet: str,
        findings: List[Dict],
        max_code_length: int = 500
    ) -> Tuple[float, str]:
        """
        Use LLM to interpret security findings in context.

        Args:
            code_snippet: The generated code
            findings: List of security findings

        Returns:
            Tuple of (severity_adjustment, interpretation_text)
        """
        if not findings:
            return 0.0, "No security findings"

        # Truncate code if too long
        if len(code_snippet) > max_code_length:
            code_snippet = code_snippet[:max_code_length] + "\n... (truncated)"

        # Format findings
        findings_text = "\n".join([
            f"- {f.get('rule_id', 'Unknown')}: {f.get('message', 'No details')}"
            for f in findings[:5]  # Limit to first 5 findings
        ])

        prompt = self.SECURITY_INTERPRETATION_PROMPT.format(
            code_snippet=code_snippet,
            findings=findings_text
        )

        try:
            self._load_model()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
            )
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Parse numeric response
            try:
                severity = float(response.split()[0])
                severity = max(0.0, min(1.0, severity))
            except (ValueError, IndexError):
                severity = 0.5  # Default to medium if parsing fails

            return severity, response

        except Exception as e:
            # Fallback to rule-based if LLM fails
            return 0.5, f"LLM interpretation failed: {e}"


class ScoringAgent:
    """
    Hybrid Scoring Agent combining rules and prompted LLM.

    Supports both Python and C/C++ code analysis.

    Usage:
        agent = ScoringAgent(config)
        result = agent.score(code="...", language="python")
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        security_weights: Optional[SecurityWeights] = None,
        use_llm_for_security: bool = True,
        llm_model: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
        default_language: str = "python"
    ):
        """
        Initialize hybrid scoring agent.

        Args:
            config: Reward configuration
            security_weights: CVSS/CWE weights
            use_llm_for_security: Whether to use LLM for security interpretation
            llm_model: Model to use for LLM scoring
            default_language: Default language for code analysis ("python" or "c")
        """
        self.config = config or RewardConfig()
        self.security_weights = security_weights or SecurityWeights()
        self.rules_engine = RulesEngine()
        self.use_llm = use_llm_for_security
        self.default_language = default_language

        if use_llm_for_security:
            self.llm_scorer = PromptedLLMScorer(model_name=llm_model)
        else:
            self.llm_scorer = None

    def _has_ambiguous_findings(
        self,
        findings: List[Dict],
        code: str,
    ) -> bool:
        """
        Detect findings that may need LLM interpretation due to ambiguity.

        Ambiguous findings include:
        - eval/exec that might be in comments or test code
        - Password/key strings that could be test data
        - Dynamic patterns that need context to assess

        Args:
            findings: List of security findings
            code: The code being analyzed

        Returns:
            True if findings are ambiguous and need LLM interpretation
        """
        # Patterns that often need context to assess properly
        ambiguous_patterns = [
            'eval',           # Could be in comment, test, or legitimate use
            'exec',           # Could be sandboxed or controlled
            'password',       # Could be test data or example
            'key',            # Could be dict key vs crypto key
            'secret',         # Could be test/example data
            'credential',     # Could be placeholder
            'pickle',         # Could be for trusted data
            'shell',          # Could be controlled input
            '__import__',     # Could be legitimate dynamic import
        ]

        # Check if any finding message contains ambiguous patterns
        for finding in findings:
            message = finding.get('message', '').lower()
            rule_id = finding.get('rule_id', '').lower()

            for pattern in ambiguous_patterns:
                if pattern in message or pattern in rule_id:
                    # Additional context checks
                    code_lower = code.lower()

                    # Don't flag as ambiguous if it's clearly malicious
                    clear_malicious = [
                        'os.system(input',       # Direct user input to system
                        'eval(input',            # Direct user input to eval
                        'subprocess.call(input', # Direct user input to subprocess
                    ]
                    if any(m in code_lower.replace(' ', '') for m in clear_malicious):
                        continue  # Clear threat, not ambiguous

                    # Check for test-like context that makes it ambiguous
                    test_indicators = [
                        'test_',
                        'unittest',
                        'pytest',
                        '# example',
                        '# test',
                        'mock',
                        'fake_',
                        'dummy_',
                    ]
                    has_test_context = any(t in code_lower for t in test_indicators)

                    # Check for comment context
                    has_comment_context = (
                        f'# {pattern}' in code_lower or
                        f'"""{pattern}' in code_lower or
                        f"'''{pattern}" in code_lower
                    )

                    if has_test_context or has_comment_context:
                        return True  # Ambiguous - need LLM to interpret

        return False

    def score(
        self,
        code: str,
        gcc_output: Optional[str] = None,
        test_output: Optional[Dict] = None,
        codeql_findings: Optional[List[Dict]] = None,
        klee_output: Optional[str] = None,
        compilation_result: Optional[CompilationResult] = None,
        language: Optional[str] = None,
    ) -> ScoringResult:
        """
        Score generated code using hybrid approach.

        Args:
            code: Generated code
            gcc_output: Raw GCC compiler output (C/C++ only)
            test_output: Test execution results dictionary
            codeql_findings: List of CodeQL/security findings
            klee_output: Raw KLEE output (C/C++ only)
            language: Code language ("python" or "c"). Defaults to self.default_language

        Returns:
            ScoringResult with all metrics
        """
        language = language or self.default_language

        # 1. Syntax/Compilation check
        if compilation_result is not None:
            compilation = compilation_result
        elif language == "python":
            # Use Python AST parsing
            compilation = self.rules_engine.parse_python_syntax(code)
        elif gcc_output is not None:
            compilation = self.rules_engine.parse_gcc_output(gcc_output)
        else:
            # Assume compiles if no info
            compilation = CompilationResult(
                compiles=True, errors=[], warnings=[],
                error_count=0, warning_count=0
            )

        # 2. Tests (rules)
        if test_output is not None:
            tests = self.rules_engine.parse_test_results(test_output)
        else:
            # Assume passes if no info
            tests = TestResult(
                passed=1, failed=0, total=1,
                has_semantic_error=False, failure_types={}
            )

        # 3. KLEE bugs (C/C++ only - not applicable to Python)
        klee_bugs = []
        if klee_output and language != "python":
            klee_bugs = self.rules_engine.parse_klee_output(klee_output)

        # 4. Security analysis
        if language == "python":
            # Use Python-specific security checks
            security_findings = codeql_findings or self.rules_engine.check_python_security(code)
        else:
            security_findings = codeql_findings or []

        # Base severity from rules
        base_severity = self.security_weights.compute_normalized_severity(
            codeql_findings=security_findings,
            klee_bugs=klee_bugs,
        )

        # Optional LLM interpretation for context-aware adjustment
        # Only use LLM when findings are ambiguous (to save compute)
        llm_interpretation = None
        if self.use_llm and self.llm_scorer and security_findings:
            # Check if findings are ambiguous and need LLM interpretation
            if self._has_ambiguous_findings(security_findings, code):
                try:
                    llm_severity, llm_interpretation = self.llm_scorer.interpret_security_findings(
                        code_snippet=code,
                        findings=security_findings,
                    )
                    # Blend rule-based and LLM severity (70% rules, 30% LLM)
                    severity_v = 0.7 * base_severity + 0.3 * llm_severity
                except Exception:
                    severity_v = base_severity
            else:
                # Not ambiguous - use rule-based severity directly
                severity_v = base_severity
        else:
            severity_v = base_severity

        security = SecurityResult(
            findings=security_findings,
            severity_score=severity_v,
            llm_interpretation=llm_interpretation,
        )

        # 5. Compute Rfunc and Rsec
        rfunc = self.rules_engine.compute_rfunc(compilation, tests)

        # Rsec = exp(-V) or 1 - min(V, 1)
        import math
        if self.config.rsec_formula == 'exp':
            rsec = math.exp(-severity_v)
        else:
            rsec = 1.0 - min(severity_v, 1.0)

        return ScoringResult(
            compilation=compilation,
            tests=tests,
            security=security,
            klee_bugs=klee_bugs,
            rfunc=rfunc,
            rsec=rsec,
            severity_v=severity_v,
        )

    def score_from_analysis(
        self,
        code: str,
        analysis_result: Dict
    ) -> ScoringResult:
        """
        Score from benchmark analysis result format.

        Args:
            code: Generated code
            analysis_result: Analysis result dictionary

        Returns:
            ScoringResult
        """
        compilation = analysis_result.get('compilation', {})
        test_output = analysis_result.get('test_execution', {})
        security = analysis_result.get('security', analysis_result.get('codeql', {}))

        # Extract CodeQL findings
        codeql_findings = []
        if isinstance(security, dict):
            issues = security.get('security_issues', [])
            if isinstance(issues, list):
                codeql_findings = [
                    {'rule_id': i} if isinstance(i, str) else i
                    for i in issues
                ]

        # Create compilation result
        comp_result = CompilationResult(
            compiles=compilation.get('compiles', True),
            errors=compilation.get('errors', []),
            warnings=compilation.get('warnings', []),
            error_count=compilation.get('error_count', 0),
            warning_count=compilation.get('warning_count', 0),
        )

        return self.score(
            code=code,
            test_output=test_output,
            codeql_findings=codeql_findings,
            compilation_result=comp_result,
        )
