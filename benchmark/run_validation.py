#!/usr/bin/env python3
"""
Quick validation script to test the benchmark pipeline.
Runs a minimal test with a few prompts to verify everything works.
"""

import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_validation():
    """Run a minimal validation of the benchmark pipeline."""

    print("=" * 60)
    print("BENCHMARK VALIDATION TEST")
    print("=" * 60)

    # Test 1: Import all modules
    print("\n[1/6] Testing module imports...")
    try:
        from benchmark.language_detector import LanguageDetector, detect_language
        from benchmark.code_cleaner import CodeCleaner
        from benchmark.test_executor import TestExecutor, FailureType
        from benchmark.dataset_loader import DatasetLoader
        from benchmark.analyze_multi import CodeAnalyzer
        from benchmark.compute_metrics import MetricsComputer
        print("  ✓ All modules imported successfully")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

    # Test 2: Language detection
    print("\n[2/6] Testing language detection...")
    detector = LanguageDetector()

    python_code = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
    c_code = "#include <stdio.h>\nint main() { printf(\"Hello\"); return 0; }"

    py_result = detector.detect(python_code)
    c_result = detector.detect(c_code)

    if py_result == 'python' and c_result == 'c':
        print(f"  ✓ Python detection: {py_result}")
        print(f"  ✓ C detection: {c_result}")
    else:
        print(f"  ✗ Detection failed: Python={py_result}, C={c_result}")
        return False

    # Test 3: Code cleaning
    print("\n[3/6] Testing code cleaning...")
    cleaner = CodeCleaner()

    raw_code = """Here's the solution:

```python
def hello():
    print("Hello, World!")
```

This prints hello world.
"""
    cleaned, lang = cleaner.clean(raw_code)
    if "def hello" in cleaned and lang == 'python':
        print(f"  ✓ Code cleaned successfully (detected: {lang})")
    else:
        print(f"  ✗ Code cleaning failed")
        return False

    # Test 4: Test executor (with simple Python code)
    print("\n[4/6] Testing test executor...")
    import tempfile

    test_code = """
n = int(input())
print(n * 2)
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_path = Path(f.name)

    try:
        executor = TestExecutor(timeout_seconds=5)
        test_cases = {
            'inputs': ['5\n', '10\n'],
            'outputs': ['10\n', '20\n']
        }
        result = executor.run_all_tests(temp_path, test_cases, language='python')

        print(f"  ✓ Tests run: {result.total_tests}")
        print(f"  ✓ Tests passed: {result.passed}")
        print(f"  ✓ Tests failed: {result.failed}")
        print(f"  ✓ Has semantic error: {result.has_semantic_error}")
    finally:
        temp_path.unlink()

    # Test 5: Dataset loader (just check it initializes)
    print("\n[5/6] Testing dataset loader...")
    try:
        loader = DatasetLoader(cache_dir="./benchmark/validation_cache")
        print("  ✓ Dataset loader initialized")

        # Try loading a small sample from APPS_Plus
        print("  Loading 2 samples from APPS_Plus...")
        prompts = loader.load_apps_plus(limit=2)
        if prompts:
            print(f"  ✓ Loaded {len(prompts)} prompts")
            print(f"  ✓ First prompt has test_cases: {'test_cases' in prompts[0]}")
            if prompts[0].get('test_cases'):
                inputs = prompts[0]['test_cases'].get('inputs', [])
                outputs = prompts[0]['test_cases'].get('outputs', [])
                print(f"  ✓ Test cases: {len(inputs)} inputs, {len(outputs)} outputs")
        else:
            print("  ⚠ No prompts loaded (may need network)")
    except Exception as e:
        print(f"  ⚠ Dataset loading skipped: {e}")

    # Test 6: Check GPU availability
    print("\n[6/6] Checking compute resources...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ CUDA available: {gpu_name} ({gpu_mem:.1f} GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✓ MPS (Apple Silicon) available")
        else:
            print("  ⚠ No GPU detected (will use CPU - slower)")
    except ImportError:
        print("  ⚠ PyTorch not installed")

    # Check CodeQL
    codeql_paths = [
        "/opt/codeql/codeql",
        f"/scratch/{os.getlogin()}/codeql/codeql" if os.getlogin() else None,
        "/usr/local/bin/codeql"
    ]
    codeql_found = False
    for path in codeql_paths:
        if path and os.path.exists(path):
            print(f"  ✓ CodeQL found: {path}")
            codeql_found = True
            break
    if not codeql_found:
        print("  ⚠ CodeQL not found (C/C++ security analysis will be skipped)")

    # Check Bandit
    import shutil
    if shutil.which('bandit'):
        print("  ✓ Bandit available for Python security analysis")
    else:
        print("  ⚠ Bandit not found (install with: pip install bandit)")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\nThe benchmark pipeline is ready to run!")
    print("\nNext steps:")
    print("1. For a quick local test:")
    print("   python -m benchmark.batch_processor --stage all")
    print("\n2. For GCP deployment:")
    print("   - Create VM with GPU")
    print("   - Run cloud/gcp/setup.sh")
    print("   - Run the benchmark")

    return True


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
