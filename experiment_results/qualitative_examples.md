# SecureCodeRL: Qualitative Examples

This document provides concrete code examples demonstrating the key findings of the SecureCodeRL project.

---

## Example 1: The "Missing Print" Bug

**Problem:** This pervasive pattern demonstrates why models fail most tests even when the code appears correct.

### Problem Statement (apps_plus_4799)
```
Input: n players, k cards each, card values, favorite numbers, joy levels
Output: Maximum total joy level
```

### Generated Code (StarCoder2-7B)
```python
# What the model generated:
n, k = map(int, input().split())
c = [int(x) for x in input().split()]
f = [int(x) for x in input().split()]
h = [int(x) for x in input().split()]

# ... rest is commented out or missing ...
# NO PRINT STATEMENT!
```

### Analysis
| Property | Status |
|----------|--------|
| Syntax Valid | Yes |
| Runs Without Error | Yes |
| Produces Output | **NO** |
| Tests Pass | 0% |

### Partial Credit Score
- **Binary Reward**: 0.0 (tests fail)
- **Partial Credit**: 0.4 (Stage 2: runs without error)

### Why This Matters
The model read all inputs correctly but never called `print()`. Under binary rewards:
- Reward = 0.0 (no gradient for learning)
- Model cannot distinguish this from syntax errors

Under partial credit:
- Reward = 0.4 (recognizes partial progress)
- Model learns it needs to add output

---

## Example 2: Security Vulnerability (eval injection)

**Problem:** Models sometimes generate code with serious security vulnerabilities.

### Problem Statement (apps_plus_131 - Zozonacci Sequence)
```
Compute Fibonacci-like sequences based on pattern abbreviations.
```

### Generated Code (DeepSeek-6.7B)
```python
def zozonacci(p, n):
    if not n or not p:
        return []
    if n < 4:
        return [0, 0, 0, 1][:n]

    pattern = [0, 0, 0, 1]

    for _ in range(3, n):
        abbrev = p[len(pattern) % len(p)]

        if abbrev == "fib":
            formula = "a[n] = a[n-1] + a[n-2]"
        elif abbrev == "jac":
            formula = "a[n] = a[n-1] + 2 * a[n-2]"
        # ... more cases ...

        # SECURITY VULNERABILITY!
        pattern.append(eval(formula))  # <-- HIGH severity

    return pattern
```

### Bandit Analysis
```
>> Issue: [B307:eval] Use of eval() detected.
   Severity: HIGH   Confidence: HIGH
   Location: generated_code.py:101
   More Info: https://bandit.readthedocs.io/en/latest/
```

### Security Reward Calculation
```
Rsec = 1 - (0.3 × 1)  = 0.7  (HIGH severity × 1 finding)
```

### Combined Reward
```
R = 0.6 × Rfunc + 0.4 × Rsec
R = 0.6 × 0.2 + 0.4 × 0.7 = 0.40
```

The security penalty reduces the reward, teaching the model to avoid `eval()`.

---

## Example 3: Partial Credit Stages (Constructed Example)

This example shows how the same problem solution progresses through partial credit stages.

### Problem: Double a Number
```
Input: Single integer n
Output: n * 2
```

### Stage 0: Syntax Error (Rfunc = 0.0)
```python
def double(
    n = int(input()
    print(n * 2
```
- Missing closing parentheses
- Cannot parse

### Stage 1: Valid Syntax (Rfunc = 0.2)
```python
def double():
    n = int(input())
    n * 2
```
- Parses correctly
- Expression evaluated but discarded

### Stage 2: Runs Without Error (Rfunc = 0.4)
```python
def double():
    n = int(input())
    result = n * 2

double()
```
- Computes correctly
- Stores result but doesn't output

### Stage 3: Produces Output (Rfunc = 0.6)
```python
def double():
    n = int(input())
    print("Result:", n * 2)

double()
```
- Produces output
- Wrong format (includes "Result:")

### Stage 5: All Tests Pass (Rfunc = 1.0)
```python
n = int(input())
print(n * 2)
```
- Correct and minimal
- All tests pass

---

## Example 4: Model Comparison (SFT vs PPO-continue)

### Results Summary (from VM experiments)

| Model | Syntax Valid | Test Pass | Security | Combined R |
|-------|-------------|-----------|----------|------------|
| SFT Baseline | 45% | 0% | 100% | 0.40 |
| PPO-simple (binary) | 15% | 0% | 100% | 0.40 |
| PPO-fresh | 25% | 0% | 100% | 0.40 |
| **PPO-continue** | **60%** | **5%** | **100%** | **0.41** |

### Key Observations

1. **PPO-simple degraded**: Binary rewards caused model collapse (45% → 15%)
   - The model learned to "game" simple metrics
   - Lost actual code generation capability

2. **PPO-continue recovered**: Partial credit enabled learning
   - Started from "degraded" PPO-simple checkpoint
   - Partial credit provided gradient for improvement
   - Achieved best results: 60% syntax, 5% test pass

3. **Security maintained**: All models kept 100% compliance
   - Joint optimization works
   - Rsec prevents security regressions

### Why Continue > Fresh?

PPO-simple, despite appearing worse on metrics, had:
- Explored the action space during binary training
- Learned something about code structure
- Just needed the right reward signal

Partial credit then provided:
- Gradient for incremental improvement
- Signal for "almost correct" solutions
- Path from syntax → runtime → output → tests

---

## Example 5: Correct Code (Baseline Comparison)

### Problem: Jon Snow's Stewards (apps_plus_349)
```
Count stewards with strength strictly between min and max.
```

### Generated Code (DeepSeek-6.7B) - CORRECT
```python
if __name__ == "__main__":
    n = int(input())
    a = list(map(int, input().split()))
    a.sort()
    ans = 0
    for i in range(1, n - 1):
        if a[i] > a[0] and a[i] < a[-1]:
            ans += 1
    print(ans)  # <-- Correctly outputs result
```

### Analysis
| Property | Status |
|----------|--------|
| Syntax Valid | Yes |
| Runs Without Error | Yes |
| Produces Output | Yes |
| Correct Logic | Yes |
| Tests Pass | Yes |

### Partial Credit Score: Rfunc = 1.0

This example shows what correct code looks like - it includes the crucial `print()` statement.

---

## Summary: Why Partial Credit Works

| Problem | Binary Reward | Partial Credit |
|---------|---------------|----------------|
| Missing print | 0.0 | 0.4 (runs) |
| Wrong output | 0.0 | 0.6 (has output) |
| Security issue | N/A | Rsec penalty |
| Partial correct | 0.0 | 0.6 + 0.4×(pass%) |

Partial credit provides:
1. **Continuous gradient** - Every improvement rewarded
2. **Staged learning** - Syntax → Runtime → Output → Correctness
3. **Missing print fix** - Stage 3 specifically rewards output
4. **No reward hacking** - Must make actual progress

---

## Data Sources

- **Benchmark Results**: `benchmark/full_results/` (Phase 1)
- **Training Prompts**: `data/prompts/ppo_prompts_with_tests.json`
- **Experiment Logs**: `results_from_vm/experiment_log.txt`
- **Evaluation Results**: `results_from_vm/evaluation_results.json`
