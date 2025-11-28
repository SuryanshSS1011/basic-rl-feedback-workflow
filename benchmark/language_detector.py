#!/usr/bin/env python3
"""
Language detection module for multi-language code analysis.
Detects programming language from code content and provides appropriate toolchain.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional


class LanguageDetector:
    """Detect programming language from code and provide language-specific toolchain."""

    # Python indicators (weighted by specificity)
    PYTHON_PATTERNS = [
        (r'^def\s+\w+\s*\(', 10),           # Function definition
        (r'^import\s+\w+', 8),               # Import statement
        (r'^from\s+\w+\s+import', 8),        # From import
        (r'^class\s+\w+.*:', 10),            # Class definition
        (r'print\s*\(', 5),                  # Print function
        (r'if\s+.*:\s*$', 3),                # If with colon
        (r'for\s+\w+\s+in\s+', 5),           # For-in loop
        (r'elif\s+', 8),                     # Python-specific elif
        (r':\s*$', 2),                       # Line ending with colon
        (r'^\s{4}\w', 2),                    # 4-space indentation
        (r'True|False|None', 3),             # Python boolean/None
        (r'lambda\s+\w+\s*:', 6),            # Lambda expression
        (r'list\(|dict\(|set\(|tuple\(', 4), # Python type constructors
        (r'range\(|len\(|str\(|int\(', 3),  # Common Python built-ins
        (r'self\.\w+', 5),                   # Self reference
        (r'__\w+__', 6),                     # Dunder methods
        (r'\.append\(|\.extend\(', 4),       # List methods
        (r'except\s+\w+:', 6),               # Exception handling
    ]

    # C/C++ indicators (weighted by specificity)
    C_PATTERNS = [
        (r'#include\s*<', 10),               # Include directive
        (r'#include\s*"', 10),               # Include with quotes
        (r'int\s+main\s*\(', 10),            # Main function
        (r'void\s+\w+\s*\(', 6),             # Void function
        (r'printf\s*\(', 8),                 # Printf call
        (r'scanf\s*\(', 8),                  # Scanf call
        (r'return\s+0\s*;', 5),              # Return 0
        (r';\s*$', 2),                       # Line ending with semicolon
        (r'\{|\}', 1),                       # Braces
        (r'int\s+\w+\s*=', 4),               # Int declaration
        (r'char\s+\w+', 5),                  # Char declaration
        (r'float\s+\w+|double\s+\w+', 5),    # Float/double
        (r'sizeof\s*\(', 6),                 # Sizeof operator
        (r'malloc\s*\(|free\s*\(', 8),       # Memory management
        (r'struct\s+\w+', 7),                # Struct definition
        (r'\*\w+\s*=', 4),                   # Pointer assignment
        (r'NULL', 5),                        # NULL constant
        (r'#define\s+', 7),                  # Macro definition
        (r'&&|\|\|', 2),                     # Logical operators
        (r'->(\w+)', 5),                     # Arrow operator
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self._python_patterns = [
            (re.compile(pattern, re.MULTILINE), weight)
            for pattern, weight in self.PYTHON_PATTERNS
        ]
        self._c_patterns = [
            (re.compile(pattern, re.MULTILINE), weight)
            for pattern, weight in self.C_PATTERNS
        ]

    def detect(self, code: str) -> str:
        """
        Detect programming language from code content.

        Args:
            code: Source code string

        Returns:
            'python', 'c', or 'unknown'
        """
        if not code or not code.strip():
            return 'unknown'

        python_score = self._calculate_score(code, self._python_patterns)
        c_score = self._calculate_score(code, self._c_patterns)

        # Require minimum score threshold
        min_threshold = 5

        if python_score >= min_threshold and python_score > c_score:
            return 'python'
        elif c_score >= min_threshold and c_score > python_score:
            return 'c'
        elif python_score >= min_threshold:
            return 'python'
        elif c_score >= min_threshold:
            return 'c'
        else:
            return 'unknown'

    def _calculate_score(
        self,
        code: str,
        patterns: List[Tuple[re.Pattern, int]]
    ) -> int:
        """Calculate weighted score for a set of patterns."""
        score = 0
        for pattern, weight in patterns:
            matches = pattern.findall(code)
            if matches:
                # Add weight for each match, with diminishing returns
                score += weight * min(len(matches), 3)
        return score

    def get_file_extension(self, language: str) -> str:
        """
        Get appropriate file extension for language.

        Args:
            language: Detected language ('python', 'c', 'unknown')

        Returns:
            File extension including dot (e.g., '.py', '.c')
        """
        extensions = {
            'python': '.py',
            'c': '.c',
            'cpp': '.cpp',
            'unknown': '.txt'
        }
        return extensions.get(language, '.txt')

    def get_compilation_command(
        self,
        language: str,
        source: Path,
        output: Optional[Path] = None
    ) -> Optional[List[str]]:
        """
        Get compilation command for language.

        Args:
            language: Detected language
            source: Source file path
            output: Output binary path (optional)

        Returns:
            List of command arguments, or None if not applicable
        """
        if output is None:
            output = source.with_suffix('.out')

        if language == 'c':
            return ['gcc', '-o', str(output), str(source), '-lm']
        elif language == 'cpp':
            return ['g++', '-o', str(output), str(source)]
        elif language == 'python':
            # Python doesn't need compilation, return None
            return None
        else:
            return None

    def get_execution_command(
        self,
        language: str,
        source_or_binary: Path
    ) -> List[str]:
        """
        Get execution command for language.

        Args:
            language: Detected language
            source_or_binary: Path to source (Python) or compiled binary (C/C++)

        Returns:
            List of command arguments for execution
        """
        if language == 'python':
            return ['python3', str(source_or_binary)]
        elif language in ('c', 'cpp'):
            # Assume binary is at same path with .out extension
            binary = source_or_binary.with_suffix('.out')
            return [str(binary)]
        else:
            # Try running as executable
            return [str(source_or_binary)]

    def get_syntax_check_command(
        self,
        language: str,
        source: Path
    ) -> Optional[List[str]]:
        """
        Get syntax checking command for language.

        Args:
            language: Detected language
            source: Source file path

        Returns:
            List of command arguments for syntax checking
        """
        if language == 'python':
            return ['python3', '-m', 'py_compile', str(source)]
        elif language == 'c':
            return ['gcc', '-fsyntax-only', '-Wall', str(source)]
        elif language == 'cpp':
            return ['g++', '-fsyntax-only', '-Wall', str(source)]
        else:
            return None

    def get_security_scanner(self, language: str) -> str:
        """
        Get appropriate security scanner for language.

        Args:
            language: Detected language

        Returns:
            Name of security scanner to use
        """
        if language == 'python':
            return 'bandit'
        elif language in ('c', 'cpp'):
            return 'codeql'
        else:
            return 'none'


def detect_language(code: str) -> str:
    """
    Convenience function to detect language from code.

    Args:
        code: Source code string

    Returns:
        Detected language ('python', 'c', or 'unknown')
    """
    detector = LanguageDetector()
    return detector.detect(code)


if __name__ == "__main__":
    # Test the language detector
    test_cases = [
        # Python code
        ("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
""", "python"),

        # C code
        ("""
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    printf("%d\\n", factorial(5));
    return 0;
}
""", "c"),

        # Ambiguous
        ("x = 5", "python"),  # More Python-like
        ("", "unknown"),
    ]

    detector = LanguageDetector()

    print("Language Detection Tests")
    print("=" * 50)

    for code, expected in test_cases:
        result = detector.detect(code)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] Expected: {expected}, Got: {result}")
        if result != expected:
            print(f"  Code preview: {code[:50]}...")

    print("\n" + "=" * 50)
    print("Tests completed")
