"""
Reward Calculator for RL Training

Computes the combined reward: R(y) = α · Rfunc(y) + β · Rsec(y)

Where:
- Rfunc = passed_tests / total_tests (continuous [0, 1])
- Rsec = exp(-V) or 1 - min(V, 1)
- V = normalized severity score from CodeQL + KLEE findings
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

from .config import RewardConfig
from .security_weights import SecurityWeights


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components"""

    # Functional correctness
    rfunc: float
    tests_passed: int
    tests_total: int
    compiles: bool

    # Security reward
    rsec: float
    severity_v: float
    codeql_count: int
    klee_bug_count: int

    # Combined
    total: float
    alpha: float
    beta: float

    def to_dict(self) -> Dict:
        return {
            'rfunc': self.rfunc,
            'tests_passed': self.tests_passed,
            'tests_total': self.tests_total,
            'compiles': self.compiles,
            'rsec': self.rsec,
            'severity_v': self.severity_v,
            'codeql_count': self.codeql_count,
            'klee_bug_count': self.klee_bug_count,
            'total': self.total,
            'alpha': self.alpha,
            'beta': self.beta,
        }


@dataclass
class RewardResult:
    """Result of reward computation"""

    reward: float  # Combined reward R(y)
    rfunc: float   # Functional correctness reward
    rsec: float    # Security reward
    breakdown: RewardBreakdown = None

    def to_dict(self) -> Dict:
        result = {
            'reward': self.reward,
            'rfunc': self.rfunc,
            'rsec': self.rsec,
        }
        if self.breakdown:
            result['breakdown'] = self.breakdown.to_dict()
        return result


class RewardCalculator:
    """
    Computes rewards for generated code based on functional correctness
    and security analysis.

    R(y) = α · Rfunc(y) + β · Rsec(y)

    Usage:
        calculator = RewardCalculator(config)
        result = calculator.compute_reward(
            compiles=True,
            tests_passed=8,
            tests_total=10,
            codeql_findings=[{'rule_id': 'cpp/null-dereference'}],
            klee_bugs=[{'type': 'ptr'}]
        )
        print(f"Reward: {result.reward}")
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        security_weights: Optional[SecurityWeights] = None
    ):
        """
        Initialize reward calculator.

        Args:
            config: Reward configuration (alpha, beta, formula)
            security_weights: Security weights database
        """
        self.config = config or RewardConfig()
        self.security_weights = security_weights or SecurityWeights(
            klee_bug_weight=self.config.klee_bug_weight
        )

    def compute_rfunc(
        self,
        compiles: bool,
        tests_passed: int,
        tests_total: int
    ) -> float:
        """
        Compute functional correctness reward.

        Rfunc = passed_tests / total_tests

        Args:
            compiles: Whether code compiles successfully
            tests_passed: Number of tests passed
            tests_total: Total number of tests

        Returns:
            Rfunc in [0, 1]
        """
        if not compiles:
            return self.config.compilation_failure_rfunc

        if tests_total == 0:
            # No tests available - assume correct if it compiles
            return 1.0

        return tests_passed / tests_total

    def compute_rsec(
        self,
        codeql_findings: List[Dict],
        klee_bugs: Optional[List[Dict]] = None,
        max_severity: float = 5.0
    ) -> tuple[float, float]:
        """
        Compute security reward.

        Rsec = exp(-V) or 1 - min(V, 1)
        where V = normalized severity

        Args:
            codeql_findings: List of CodeQL findings
            klee_bugs: List of KLEE bugs
            max_severity: Maximum severity for normalization

        Returns:
            Tuple of (Rsec, V)
        """
        # Compute normalized severity V
        v = self.security_weights.compute_normalized_severity(
            codeql_findings=codeql_findings,
            klee_bugs=klee_bugs,
            max_severity=max_severity
        )

        # Compute Rsec based on formula
        if self.config.rsec_formula == 'exp':
            rsec = math.exp(-v)
        else:  # linear
            rsec = 1.0 - min(v, 1.0)

        return rsec, v

    def compute_reward(
        self,
        compiles: bool,
        tests_passed: int,
        tests_total: int,
        codeql_findings: Optional[List[Dict]] = None,
        klee_bugs: Optional[List[Dict]] = None,
        max_severity: float = 5.0,
        include_breakdown: bool = True
    ) -> RewardResult:
        """
        Compute combined reward R(y) = α · Rfunc + β · Rsec

        Args:
            compiles: Whether code compiles
            tests_passed: Number of tests passed
            tests_total: Total number of tests
            codeql_findings: List of CodeQL findings
            klee_bugs: List of KLEE bugs
            max_severity: Maximum severity for normalization
            include_breakdown: Include detailed breakdown

        Returns:
            RewardResult with combined reward and components
        """
        codeql_findings = codeql_findings or []
        klee_bugs = klee_bugs or []

        # Compute components
        rfunc = self.compute_rfunc(compiles, tests_passed, tests_total)
        rsec, severity_v = self.compute_rsec(
            codeql_findings, klee_bugs, max_severity
        )

        # Combined reward
        alpha = self.config.alpha
        beta = self.config.beta
        total_reward = alpha * rfunc + beta * rsec

        # Build breakdown if requested
        breakdown = None
        if include_breakdown:
            breakdown = RewardBreakdown(
                rfunc=rfunc,
                tests_passed=tests_passed,
                tests_total=tests_total,
                compiles=compiles,
                rsec=rsec,
                severity_v=severity_v,
                codeql_count=len(codeql_findings),
                klee_bug_count=len(klee_bugs),
                total=total_reward,
                alpha=alpha,
                beta=beta,
            )

        return RewardResult(
            reward=total_reward,
            rfunc=rfunc,
            rsec=rsec,
            breakdown=breakdown,
        )

    def compute_from_analysis_result(
        self,
        analysis_result: Dict,
        include_breakdown: bool = True
    ) -> RewardResult:
        """
        Compute reward from benchmark analysis result format.

        Args:
            analysis_result: Dictionary with 'compilation', 'test_execution',
                           'security' or 'codeql', and optionally 'klee' keys

        Returns:
            RewardResult
        """
        # Parse compilation
        compilation = analysis_result.get('compilation', {})
        compiles = compilation.get('compiles', False)

        # Parse test execution
        test_exec = analysis_result.get('test_execution', {})
        tests_passed = test_exec.get('passed', 0)
        tests_total = test_exec.get('total_tests', 0)

        # If no test info, check semantic error flag
        if tests_total == 0:
            has_semantic_error = test_exec.get('has_semantic_error', None)
            if has_semantic_error is not None:
                tests_passed = 0 if has_semantic_error else 1
                tests_total = 1

        # Parse security findings
        security = analysis_result.get('security', analysis_result.get('codeql', {}))
        codeql_findings = []
        if isinstance(security, dict):
            security_issues = security.get('security_issues', [])
            if isinstance(security_issues, list):
                codeql_findings = [
                    {'rule_id': issue} if isinstance(issue, str)
                    else issue
                    for issue in security_issues
                ]
            elif security.get('has_security_issues', False):
                # Generic security issue
                codeql_findings = [{'rule_id': 'unknown'}]

        # Parse KLEE bugs
        klee = analysis_result.get('klee', {})
        klee_bugs = []
        if isinstance(klee, dict):
            klee_bugs = klee.get('bugs', [])
        elif isinstance(klee, list):
            klee_bugs = klee

        return self.compute_reward(
            compiles=compiles,
            tests_passed=tests_passed,
            tests_total=tests_total,
            codeql_findings=codeql_findings,
            klee_bugs=klee_bugs,
            include_breakdown=include_breakdown,
        )

    def compute_advantage(
        self,
        new_reward: float,
        baseline_reward: float
    ) -> float:
        """
        Compute advantage: A = R(y') - R(y)

        Args:
            new_reward: Reward for new sample
            baseline_reward: Baseline reward (previous sample or value estimate)

        Returns:
            Advantage value
        """
        return new_reward - baseline_reward

    def batch_compute_rewards(
        self,
        analysis_results: List[Dict]
    ) -> List[RewardResult]:
        """
        Compute rewards for a batch of analysis results.

        Args:
            analysis_results: List of analysis result dictionaries

        Returns:
            List of RewardResult objects
        """
        return [
            self.compute_from_analysis_result(result)
            for result in analysis_results
        ]

    def get_reward_stats(
        self,
        rewards: List[RewardResult]
    ) -> Dict:
        """
        Compute statistics over a batch of rewards.

        Args:
            rewards: List of RewardResult objects

        Returns:
            Dictionary with mean, std, min, max for each component
        """
        if not rewards:
            return {}

        rfuncs = [r.rfunc for r in rewards]
        rsecs = [r.rsec for r in rewards]
        totals = [r.reward for r in rewards]

        def stats(values):
            n = len(values)
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / n
            std = math.sqrt(variance)
            return {
                'mean': mean,
                'std': std,
                'min': min(values),
                'max': max(values),
            }

        return {
            'rfunc': stats(rfuncs),
            'rsec': stats(rsecs),
            'total': stats(totals),
            'count': len(rewards),
        }
