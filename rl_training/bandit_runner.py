"""
Bandit Integration for Python Security Analysis

Provides comprehensive static security analysis using Bandit,
with mapping to CVSS/CWE severity weights.

Usage:
    from rl_training.bandit_runner import BanditRunner

    runner = BanditRunner()
    findings = runner.analyze(code)
    severity = runner.compute_severity(findings)
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .security_weights import SecurityWeights


@dataclass
class BanditFinding:
    """A single Bandit security finding"""
    test_id: str          # e.g., 'B102', 'B608'
    test_name: str        # e.g., 'exec_used', 'hardcoded_sql_expressions'
    severity: str         # 'LOW', 'MEDIUM', 'HIGH'
    confidence: str       # 'LOW', 'MEDIUM', 'HIGH'
    message: str          # Issue description
    line_number: int      # Line in code
    cwe_id: Optional[str] = None  # Mapped CWE ID
    weight: float = 0.5   # Severity weight [0, 1]

    def to_dict(self) -> Dict:
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'severity': self.severity,
            'confidence': self.confidence,
            'message': self.message,
            'line_number': self.line_number,
            'cwe_id': self.cwe_id,
            'weight': self.weight,
        }


class BanditRunner:
    """
    Runs Bandit static security analysis on Python code.

    Provides both comprehensive analysis (for evaluation) and
    fast regex-based fallback (for online training).
    """

    # Severity multipliers for confidence levels
    CONFIDENCE_MULTIPLIERS = {
        'HIGH': 1.0,
        'MEDIUM': 0.75,
        'LOW': 0.5,
    }

    def __init__(
        self,
        security_weights: Optional[SecurityWeights] = None,
        timeout_seconds: int = 10,
        use_bandit: bool = True,
    ):
        """
        Initialize Bandit runner.

        Args:
            security_weights: SecurityWeights instance for severity mapping
            timeout_seconds: Timeout for Bandit execution
            use_bandit: If False, use fast regex fallback instead
        """
        self.security_weights = security_weights or SecurityWeights()
        self.timeout = timeout_seconds
        self.use_bandit = use_bandit and self._check_bandit_available()

    def _check_bandit_available(self) -> bool:
        """Check if Bandit is installed and available"""
        try:
            result = subprocess.run(
                ['bandit', '--version'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def analyze(self, code: str) -> List[BanditFinding]:
        """
        Analyze Python code for security issues.

        Args:
            code: Python source code string

        Returns:
            List of BanditFinding objects
        """
        if not code or len(code.strip()) < 5:
            return []

        if self.use_bandit:
            return self._run_bandit(code)
        else:
            return self._regex_fallback(code)

    def _run_bandit(self, code: str) -> List[BanditFinding]:
        """Run actual Bandit analysis"""
        findings = []
        code_file = None

        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(code)
                code_file = f.name

            # Run Bandit with JSON output
            result = subprocess.run(
                [
                    'bandit',
                    '-f', 'json',      # JSON output format
                    '-q',               # Quiet mode (no progress)
                    '-ll',              # Only medium+ severity
                    '--exit-zero',      # Don't fail on findings
                    code_file,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # Parse JSON output
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for item in data.get('results', []):
                        finding = self._parse_finding(item)
                        if finding:
                            findings.append(finding)
                except json.JSONDecodeError:
                    pass

        except subprocess.TimeoutExpired:
            # Timeout - fall back to regex
            findings = self._regex_fallback(code)

        except Exception:
            # Any error - fall back to regex
            findings = self._regex_fallback(code)

        finally:
            # Cleanup temp file
            if code_file:
                try:
                    os.unlink(code_file)
                except OSError:
                    pass

        return findings

    def _parse_finding(self, item: Dict) -> Optional[BanditFinding]:
        """Parse a single Bandit finding from JSON"""
        test_id = item.get('test_id', '')
        if not test_id:
            return None

        # Get CWE mapping and weight
        cwe_id = self.security_weights.codeql_to_cwe.get(test_id)
        weight = self.security_weights.get_weight_for_bandit(
            test_id,
            item.get('issue_severity', 'MEDIUM')
        )

        # Adjust weight by confidence
        confidence = item.get('issue_confidence', 'MEDIUM').upper()
        confidence_mult = self.CONFIDENCE_MULTIPLIERS.get(confidence, 0.75)
        adjusted_weight = weight * confidence_mult

        return BanditFinding(
            test_id=test_id,
            test_name=item.get('test_name', ''),
            severity=item.get('issue_severity', 'MEDIUM').upper(),
            confidence=confidence,
            message=item.get('issue_text', ''),
            line_number=item.get('line_number', 0),
            cwe_id=cwe_id,
            weight=adjusted_weight,
        )

    def _regex_fallback(self, code: str) -> List[BanditFinding]:
        """Fast regex-based security check (fallback when Bandit unavailable)"""
        import re

        findings = []

        # Map patterns to Bandit test IDs
        patterns = [
            (r'\beval\s*\(', 'B307', 'eval', 'Use of eval() detected', 'HIGH'),
            (r'\bexec\s*\(', 'B102', 'exec_used', 'Use of exec() detected', 'HIGH'),
            (r'\b__import__\s*\(', 'B403', 'import_pickle', 'Dynamic import detected', 'MEDIUM'),
            (r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True',
             'B602', 'subprocess_popen_with_shell_equals_true',
             'Shell=True in subprocess', 'HIGH'),
            (r'\bos\.system\s*\(', 'B605', 'start_process_with_a_shell',
             'os.system() call detected', 'HIGH'),
            (r'\bos\.popen\s*\(', 'B605', 'start_process_with_a_shell',
             'os.popen() call detected', 'HIGH'),
            (r'\bpickle\.loads?\s*\(', 'B301', 'pickle',
             'Pickle usage detected', 'HIGH'),
            (r'\byaml\.load\s*\([^)]*\)', 'B506', 'yaml_load',
             'Unsafe YAML load', 'MEDIUM'),
            (r'input\s*\(\s*\)', 'B322', 'input',
             'input() usage (check Python version)', 'LOW'),
        ]

        for pattern, test_id, test_name, message, severity in patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE))
            for match in matches:
                # Find line number
                line_num = code[:match.start()].count('\n') + 1

                # Get weight from security weights
                cwe_id = self.security_weights.codeql_to_cwe.get(test_id)
                weight = self.security_weights.get_weight_for_bandit(test_id, severity)

                findings.append(BanditFinding(
                    test_id=test_id,
                    test_name=test_name,
                    severity=severity,
                    confidence='HIGH',  # Regex matches are definite
                    message=message,
                    line_number=line_num,
                    cwe_id=cwe_id,
                    weight=weight,
                ))

        return findings

    def compute_severity(
        self,
        findings: List[BanditFinding],
        max_severity: float = 5.0
    ) -> float:
        """
        Compute normalized severity score V from findings.

        V = sum(wi) / max_severity, capped at 1.0

        Args:
            findings: List of BanditFinding objects
            max_severity: Maximum severity for normalization

        Returns:
            Normalized severity V in [0, 1]
        """
        if not findings:
            return 0.0

        total_weight = sum(f.weight for f in findings)
        normalized = total_weight / max_severity
        return min(normalized, 1.0)

    def compute_rsec(
        self,
        findings: List[BanditFinding],
        formula: str = 'exp'
    ) -> float:
        """
        Compute Rsec (security reward) from findings.

        Args:
            findings: List of BanditFinding objects
            formula: 'exp' for exp(-V), 'linear' for 1-min(V,1)

        Returns:
            Rsec in [0, 1], where 1 = no security issues
        """
        import math

        v = self.compute_severity(findings)

        if formula == 'exp':
            return math.exp(-v)
        else:  # linear
            return 1.0 - min(v, 1.0)

    def format_report(self, findings: List[BanditFinding]) -> str:
        """Format findings as human-readable report"""
        if not findings:
            return "No security issues found."

        lines = [f"Security Analysis: {len(findings)} issue(s) found\n"]
        lines.append("=" * 50)

        # Group by severity
        by_severity = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        for f in findings:
            by_severity.get(f.severity, by_severity['MEDIUM']).append(f)

        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            issues = by_severity[severity]
            if issues:
                lines.append(f"\n{severity} Severity ({len(issues)}):")
                for f in issues:
                    cwe_str = f" ({f.cwe_id})" if f.cwe_id else ""
                    lines.append(f"  Line {f.line_number}: [{f.test_id}]{cwe_str} {f.message}")

        # Summary
        v = self.compute_severity(findings)
        rsec = self.compute_rsec(findings)
        lines.append(f"\nSeverity Score: V = {v:.3f}")
        lines.append(f"Security Reward: Rsec = {rsec:.3f}")

        return "\n".join(lines)


def run_bandit_analysis(code: str, use_bandit: bool = True) -> Dict:
    """
    Convenience function for running Bandit analysis.

    Args:
        code: Python source code
        use_bandit: Whether to use actual Bandit (vs regex fallback)

    Returns:
        Dict with 'findings', 'severity', 'rsec', 'report'
    """
    runner = BanditRunner(use_bandit=use_bandit)
    findings = runner.analyze(code)

    return {
        'findings': [f.to_dict() for f in findings],
        'severity': runner.compute_severity(findings),
        'rsec': runner.compute_rsec(findings),
        'report': runner.format_report(findings),
        'num_issues': len(findings),
    }
