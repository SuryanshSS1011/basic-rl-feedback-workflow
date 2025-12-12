"""
KLEE Integration for Symbolic Execution

Provides symbolic execution capabilities for finding bugs in generated C code.
KLEE explores all possible execution paths to find:
- Buffer overflows
- Null pointer dereferences
- Division by zero
- Memory leaks
- Use-after-free

Note: KLEE is computationally expensive, so it's primarily used for
offline evaluation rather than online RL training.

IMPORTANT: KLEE only works on C/C++ code. For Python code analysis,
use the scoring_agent.py with language="python" which uses AST parsing
and regex-based security checks instead.
"""

import os
import re
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class KLEEBug:
    """A bug found by KLEE"""

    bug_type: str  # ptr, free, overflow, div, assert, abort
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    test_case: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'type': self.bug_type,
            'message': self.message,
            'file': self.file_path,
            'line': self.line_number,
            'test_case': self.test_case,
        }


@dataclass
class KLEEResult:
    """Result of KLEE analysis"""

    success: bool
    bugs: List[KLEEBug]
    paths_explored: int
    instructions_executed: int
    execution_time: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'bugs': [b.to_dict() for b in self.bugs],
            'paths_explored': self.paths_explored,
            'instructions_executed': self.instructions_executed,
            'execution_time': self.execution_time,
            'error': self.error_message,
        }


class KLEERunner:
    """
    Runs KLEE symbolic execution on C code.

    Requirements:
    - KLEE installed (https://klee.github.io/)
    - LLVM/Clang for compiling to LLVM IR

    Usage:
        runner = KLEERunner()
        result = runner.analyze(code)
        for bug in result.bugs:
            print(f"Found {bug.bug_type}: {bug.message}")
    """

    # Error patterns in KLEE output
    ERROR_PATTERNS = {
        'ptr': [
            re.compile(r'memory error.*ptr', re.IGNORECASE),
            re.compile(r'invalid memory access', re.IGNORECASE),
            re.compile(r'null pointer', re.IGNORECASE),
        ],
        'free': [
            re.compile(r'double free', re.IGNORECASE),
            re.compile(r'invalid free', re.IGNORECASE),
            re.compile(r'free of alloca', re.IGNORECASE),
        ],
        'overflow': [
            re.compile(r'buffer overflow', re.IGNORECASE),
            re.compile(r'out of bound', re.IGNORECASE),
            re.compile(r'heap-buffer-overflow', re.IGNORECASE),
            re.compile(r'stack-buffer-overflow', re.IGNORECASE),
        ],
        'div': [
            re.compile(r'division by zero', re.IGNORECASE),
            re.compile(r'divide by zero', re.IGNORECASE),
        ],
        'assert': [
            re.compile(r'assertion fail', re.IGNORECASE),
            re.compile(r'ASSERTION FAIL', re.IGNORECASE),
        ],
        'abort': [
            re.compile(r'abort', re.IGNORECASE),
            re.compile(r'ABORT', re.IGNORECASE),
        ],
        'memory': [
            re.compile(r'memory leak', re.IGNORECASE),
            re.compile(r'use after free', re.IGNORECASE),
            re.compile(r'uninitialized', re.IGNORECASE),
        ],
    }

    def __init__(
        self,
        klee_path: str = "klee",
        clang_path: str = "clang",
        timeout: int = 60,
        max_time: int = 30,
        search_strategy: str = "random-path",
        temp_dir: Optional[Path] = None,
    ):
        """
        Initialize KLEE runner.

        Args:
            klee_path: Path to KLEE executable
            clang_path: Path to clang for LLVM IR compilation
            timeout: Overall timeout for KLEE execution
            max_time: KLEE's internal time limit per path
            search_strategy: KLEE search strategy (dfs, bfs, random-path, etc.)
            temp_dir: Temporary directory for work files
        """
        self.klee_path = klee_path
        self.clang_path = clang_path
        self.timeout = timeout
        self.max_time = max_time
        self.search_strategy = search_strategy
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "klee_analysis"

    def is_available(self) -> bool:
        """Check if KLEE is installed and available"""
        try:
            result = subprocess.run(
                [self.klee_path, "--version"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def analyze(
        self,
        code: str,
        entry_function: str = "main",
        symbolic_args: Optional[List[str]] = None,
        language: str = "c",
    ) -> KLEEResult:
        """
        Analyze C code with KLEE.

        Args:
            code: C source code
            entry_function: Entry function for analysis
            symbolic_args: Arguments to make symbolic
            language: Code language ("c" or "python"). Python returns empty result.

        Returns:
            KLEEResult with found bugs
        """
        import time
        start_time = time.time()

        # KLEE only works on C/C++ code
        if language.lower() == "python":
            return KLEEResult(
                success=True,
                bugs=[],
                paths_explored=0,
                instructions_executed=0,
                execution_time=0.0,
                error_message="KLEE not applicable for Python code",
            )

        # Create work directory
        work_dir = self.temp_dir / f"klee_{os.getpid()}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Write code to file
            source_file = work_dir / "code.c"
            with open(source_file, 'w') as f:
                f.write(self._prepare_code(code))

            # Compile to LLVM IR
            bc_file = work_dir / "code.bc"
            compile_result = self._compile_to_llvm(source_file, bc_file)

            if not compile_result[0]:
                return KLEEResult(
                    success=False,
                    bugs=[],
                    paths_explored=0,
                    instructions_executed=0,
                    execution_time=time.time() - start_time,
                    error_message=f"Compilation failed: {compile_result[1]}",
                )

            # Run KLEE
            klee_output_dir = work_dir / "klee-out"
            klee_result = self._run_klee(bc_file, klee_output_dir)

            if not klee_result[0]:
                return KLEEResult(
                    success=False,
                    bugs=[],
                    paths_explored=0,
                    instructions_executed=0,
                    execution_time=time.time() - start_time,
                    error_message=f"KLEE failed: {klee_result[1]}",
                )

            # Parse results
            bugs = self._parse_klee_output(klee_output_dir, klee_result[1])
            stats = self._parse_klee_stats(klee_output_dir)

            return KLEEResult(
                success=True,
                bugs=bugs,
                paths_explored=stats.get('paths', 0),
                instructions_executed=stats.get('instructions', 0),
                execution_time=time.time() - start_time,
            )

        finally:
            # Cleanup
            try:
                shutil.rmtree(work_dir)
            except OSError:
                pass

    def _prepare_code(self, code: str) -> str:
        """Prepare code for KLEE analysis"""
        # Add KLEE headers if not present
        if "#include <klee/klee.h>" not in code:
            code = "#include <klee/klee.h>\n" + code

        return code

    def _compile_to_llvm(
        self,
        source_file: Path,
        output_file: Path,
    ) -> Tuple[bool, str]:
        """Compile C code to LLVM bitcode"""
        try:
            result = subprocess.run(
                [
                    self.clang_path,
                    "-emit-llvm",
                    "-c",
                    "-g",  # Debug info
                    "-O0",  # No optimization for better analysis
                    "-Xclang", "-disable-O0-optnone",
                    str(source_file),
                    "-o", str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return False, result.stderr

            return True, ""

        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except FileNotFoundError:
            return False, f"Clang not found: {self.clang_path}"

    def _run_klee(
        self,
        bc_file: Path,
        output_dir: Path,
    ) -> Tuple[bool, str]:
        """Run KLEE on LLVM bitcode"""
        try:
            result = subprocess.run(
                [
                    self.klee_path,
                    f"--output-dir={output_dir}",
                    f"--max-time={self.max_time}",
                    f"--search={self.search_strategy}",
                    "--emit-all-errors",
                    "--only-output-states-covering-new",
                    "--max-memory=2048",  # 2GB limit
                    str(bc_file),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # KLEE may return non-zero but still produce useful output
            return True, result.stderr + result.stdout

        except subprocess.TimeoutExpired:
            return False, "KLEE execution timeout"
        except FileNotFoundError:
            return False, f"KLEE not found: {self.klee_path}"

    def _parse_klee_output(
        self,
        output_dir: Path,
        klee_output: str,
    ) -> List[KLEEBug]:
        """Parse KLEE output and error files"""
        bugs = []

        # Parse stderr/stdout for error messages
        for line in klee_output.split('\n'):
            bug = self._classify_error(line)
            if bug:
                bugs.append(bug)

        # Parse .err files in output directory
        if output_dir.exists():
            for err_file in output_dir.glob("*.err"):
                try:
                    with open(err_file) as f:
                        content = f.read()
                    bug = self._parse_err_file(err_file.name, content)
                    if bug and bug not in bugs:
                        bugs.append(bug)
                except OSError:
                    pass

        return bugs

    def _classify_error(self, line: str) -> Optional[KLEEBug]:
        """Classify an error line by type"""
        for bug_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(line):
                    return KLEEBug(
                        bug_type=bug_type,
                        message=line.strip(),
                    )
        return None

    def _parse_err_file(self, filename: str, content: str) -> Optional[KLEEBug]:
        """Parse a KLEE .err file"""
        # Extract bug type from filename (e.g., test000001.ptr.err)
        parts = filename.split('.')
        if len(parts) >= 2:
            bug_type = parts[-2]  # e.g., "ptr", "free", "div"
        else:
            bug_type = "unknown"

        # Extract first line as message
        message = content.split('\n')[0].strip() if content else filename

        # Try to extract file/line info
        file_match = re.search(r'File:\s*(.+)', content)
        line_match = re.search(r'Line:\s*(\d+)', content)

        return KLEEBug(
            bug_type=bug_type,
            message=message,
            file_path=file_match.group(1) if file_match else None,
            line_number=int(line_match.group(1)) if line_match else None,
        )

    def _parse_klee_stats(self, output_dir: Path) -> Dict[str, int]:
        """Parse KLEE run.stats file"""
        stats = {'paths': 0, 'instructions': 0}

        stats_file = output_dir / "run.stats"
        if not stats_file.exists():
            return stats

        try:
            with open(stats_file) as f:
                for line in f:
                    if line.startswith("NumQueries"):
                        # Parse CSV-like format
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            stats['queries'] = int(parts[1])
        except (OSError, ValueError):
            pass

        # Count test cases as proxy for paths
        if output_dir.exists():
            stats['paths'] = len(list(output_dir.glob("*.ktest")))

        return stats

    def analyze_batch(
        self,
        codes: List[str],
        max_workers: int = 4,
    ) -> List[KLEEResult]:
        """
        Analyze multiple code samples in parallel.

        Args:
            codes: List of C code samples
            max_workers: Maximum parallel KLEE processes

        Returns:
            List of KLEEResults
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(codes)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.analyze, code): i
                for i, code in enumerate(codes)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = KLEEResult(
                        success=False,
                        bugs=[],
                        paths_explored=0,
                        instructions_executed=0,
                        execution_time=0,
                        error_message=str(e),
                    )

        return results


def analyze_with_klee(
    code: str,
    timeout: int = 60,
    language: str = "c",
) -> Tuple[List[Dict], bool]:
    """
    Convenience function to analyze code with KLEE.

    Args:
        code: C source code (Python returns empty result)
        timeout: Timeout in seconds
        language: Code language ("c" or "python")

    Returns:
        Tuple of (list of bug dicts, success bool)
    """
    # KLEE only works on C/C++
    if language.lower() == "python":
        return [], True

    runner = KLEERunner(timeout=timeout)

    if not runner.is_available():
        return [], False

    result = runner.analyze(code, language=language)

    return [b.to_dict() for b in result.bugs], result.success


def quick_klee_check(code: str, language: str = "c") -> int:
    """
    Quick KLEE check returning number of bugs found.

    Args:
        code: C source code (Python returns 0)
        language: Code language ("c" or "python")

    Returns:
        Number of bugs found (0 if KLEE unavailable, failed, or Python)
    """
    # KLEE only works on C/C++
    if language.lower() == "python":
        return 0

    bugs, success = analyze_with_klee(code, timeout=30, language=language)
    return len(bugs) if success else 0
