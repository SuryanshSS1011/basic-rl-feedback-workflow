#!/usr/bin/env python3
"""
Multi-language code cleaner for LLM-generated content.
Extracts valid code from LLM output, removing explanatory text and artifacts.
Supports Python and C/C++.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

from .language_detector import LanguageDetector, detect_language


class CodeCleaner:
    """Clean LLM-generated code for multiple languages."""

    def __init__(self):
        self.language_detector = LanguageDetector()

    def clean(
        self,
        raw_code: str,
        expected_language: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Clean LLM-generated code.

        Args:
            raw_code: Raw LLM output
            expected_language: Expected language (if known)

        Returns:
            Tuple of (cleaned_code, detected_language)
        """
        # First, extract code from markdown blocks if present
        extracted = self._extract_from_markdown(raw_code)

        # Detect language if not specified
        if expected_language:
            language = expected_language
        else:
            language = self.language_detector.detect(extracted)

        # Apply language-specific cleaning
        if language == 'python':
            cleaned = self._clean_python(extracted)
        elif language in ('c', 'cpp'):
            cleaned = self._clean_c(extracted)
        else:
            # Default to generic cleaning
            cleaned = self._clean_generic(extracted)

        return cleaned, language

    def _extract_from_markdown(self, content: str) -> str:
        """Extract code from markdown code blocks."""
        # Look for code blocks with language specifier
        patterns = [
            r'```(?:python|py)\n(.*?)```',
            r'```(?:c|cpp|c\+\+)\n(.*?)```',
            r'```\n(.*?)```',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the first (or longest) code block
                return max(matches, key=len)

        # No code blocks found, remove any stray ``` markers
        content = re.sub(r'```[a-zA-Z]*\n?', '', content)
        return content.strip()

    def _clean_python(self, code: str) -> str:
        """
        Clean Python code from LLM output.

        Handles:
        - Multiple function definitions (keeps relevant ones)
        - Test code and examples
        - Explanatory comments at the end
        """
        lines = code.split('\n')
        cleaned_lines = []
        in_function = False
        function_indent = 0
        seen_functions = set()

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip common LLM artifacts
            if any(skip in stripped.lower() for skip in [
                'here is', 'here\'s', 'example:', 'output:', 'note:',
                'explanation:', 'this code', 'the above', '# test',
                'if __name__'  # Often included but may not be needed
            ]):
                # If this is a standalone comment/text, skip it
                if stripped.startswith('#') or not stripped.startswith(('def ', 'class ', 'import ', 'from ')):
                    continue

            # Track function definitions
            if stripped.startswith('def '):
                func_name_match = re.match(r'def\s+(\w+)', stripped)
                if func_name_match:
                    func_name = func_name_match.group(1)
                    # Skip duplicate functions
                    if func_name in seen_functions:
                        continue
                    seen_functions.add(func_name)

                in_function = True
                function_indent = len(line) - len(line.lstrip())

            # Track class definitions
            if stripped.startswith('class '):
                in_function = True
                function_indent = len(line) - len(line.lstrip())

            # Check if we've exited the function/class
            if in_function and line.strip() and not line.startswith(' ' * (function_indent + 1)):
                if not stripped.startswith(('def ', 'class ', '@')):
                    in_function = False

            cleaned_lines.append(line)

        # Remove trailing empty lines and comments
        while cleaned_lines and (not cleaned_lines[-1].strip() or cleaned_lines[-1].strip().startswith('#')):
            cleaned_lines.pop()

        return '\n'.join(cleaned_lines)

    def _clean_c(self, code: str) -> str:
        """
        Clean C/C++ code from LLM output.

        Handles:
        - Multiple main functions (keeps first complete one)
        - Explanatory comments
        - Duplicate code blocks
        """
        lines = code.split('\n')
        cleaned_lines = []

        main_started = False
        brace_count = 0
        main_complete = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines at the beginning
            if not cleaned_lines and not stripped:
                continue

            # Skip explanatory text that's not code
            if any(skip in stripped.lower() for skip in [
                'here is', 'here\'s', 'example:', 'output:', 'note:',
                'explanation:', 'this code', 'the above'
            ]):
                if stripped.startswith('//') or not any(c in stripped for c in '{}();'):
                    continue

            # Skip duplicate main functions after the first one is complete
            if main_complete and 'int main(' in line:
                continue

            # Detect start of main function
            if 'int main(' in line or 'void main(' in line:
                main_started = True

            # Track braces when in main
            if main_started and not main_complete:
                brace_count += line.count('{') - line.count('}')

                # Main function complete when braces balance
                if brace_count == 0 and '}' in line and main_started:
                    main_complete = True

            cleaned_lines.append(line)

            # Stop if main is complete and we hit explanatory content
            if main_complete and stripped.startswith('//') and any(
                skip in stripped.lower() for skip in ['note', 'this', 'the above', 'output']
            ):
                break

        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        return '\n'.join(cleaned_lines)

    def _clean_generic(self, code: str) -> str:
        """Generic code cleaning for unknown languages."""
        lines = code.split('\n')

        # Remove leading/trailing empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        return '\n'.join(lines)

    def clean_file(
        self,
        input_path: Path,
        output_path: Path,
        expected_language: Optional[str] = None
    ) -> str:
        """
        Clean code from input file and save to output file.

        Args:
            input_path: Path to raw code file
            output_path: Path to save cleaned code
            expected_language: Expected language (if known)

        Returns:
            Detected language
        """
        with open(input_path, 'r') as f:
            raw_code = f.read()

        cleaned_code, language = self.clean(raw_code, expected_language)

        with open(output_path, 'w') as f:
            f.write(cleaned_code)

        return language


def clean_code(
    raw_code: str,
    expected_language: Optional[str] = None
) -> Tuple[str, str]:
    """
    Convenience function to clean code.

    Args:
        raw_code: Raw LLM-generated code
        expected_language: Expected language (optional)

    Returns:
        Tuple of (cleaned_code, detected_language)
    """
    cleaner = CodeCleaner()
    return cleaner.clean(raw_code, expected_language)


# For backward compatibility with existing code
def clean_c_code(input_file: str, output_file: str) -> None:
    """
    Clean C code from file (backward compatible with original clean_code.py).

    Args:
        input_file: Path to input file
        output_file: Path to output file
    """
    cleaner = CodeCleaner()
    cleaner.clean_file(Path(input_file), Path(output_file), expected_language='c')
    print(f"Code cleaned: {input_file} -> {output_file}")


def clean_python_code(input_file: str, output_file: str) -> None:
    """
    Clean Python code from file.

    Args:
        input_file: Path to input file
        output_file: Path to output file
    """
    cleaner = CodeCleaner()
    cleaner.clean_file(Path(input_file), Path(output_file), expected_language='python')
    print(f"Code cleaned: {input_file} -> {output_file}")


if __name__ == "__main__":
    # Test the code cleaner
    test_cases = [
        # Python with markdown
        ("""Here's a solution:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

This calculates the factorial recursively.
""", "python"),

        # C with explanation
        ("""#include <stdio.h>

int main() {
    int n = 5;
    printf("%d\\n", n);
    return 0;
}

// Note: This is a simple program
// Output: 5
""", "c"),
    ]

    cleaner = CodeCleaner()

    print("Code Cleaning Tests")
    print("=" * 50)

    for raw, expected_lang in test_cases:
        cleaned, detected = cleaner.clean(raw)
        print(f"\nExpected language: {expected_lang}")
        print(f"Detected language: {detected}")
        print(f"Cleaned code:\n{cleaned[:200]}...")
        print("-" * 50)
