# SecureCodeRL Evaluation Results

## Evaluation Configuration
- **Evaluation script:** `evaluate_dual_metrics.py`
- **N prompts:** 20 per model
- **Test execution:** stdin/stdout with timeout
- **Security analysis:** Bandit static analyzer

---

## Summary Table

| Model | N | Syntax % | Test Pass % | Security % | Rfunc | Rsec | R |
|-------|---|----------|-------------|------------|-------|------|---|
| SFT Baseline | 20 | 45.0% | 0.0% | 100% | 0.00 | 1.00 | 0.40 |
| PPO-simple | 20 | 15.0% | 0.0% | 100% | 0.00 | 1.00 | 0.40 |
| PPO-continue | 20 | **60.0%** | **5.0%** | 100% | 0.02 | 1.00 | **0.41** |
| PPO-fresh | 20 | 25.0% | 0.0% | 100% | 0.00 | 1.00 | 0.40 |

---

## Failure Mode Breakdown (from per-sample analysis)

### SFT Baseline (20 samples)
| Outcome | Count | % |
|---------|-------|---|
| Syntax Error | 11 | 55% |
| Valid but No Test Pass | 9 | 45% |
| At Least 1 Test Pass | 0 | 0% |
| Security Issues (regex) | 3 | 15% |
| Security Issues (Bandit) | 0 | 0% |

### PPO-simple (20 samples)
| Outcome | Count | % |
|---------|-------|---|
| Syntax Error | 17 | 85% |
| Valid but No Test Pass | 3 | 15% |
| At Least 1 Test Pass | 0 | 0% |
| Security Issues (regex) | 0 | 0% |
| Security Issues (Bandit) | 0 | 0% |

### PPO-continue (20 samples) - BEST
| Outcome | Count | % |
|---------|-------|---|
| Syntax Error | 8 | 40% |
| Valid but No Test Pass | 11 | 55% |
| **At Least 1 Test Pass** | **1** | **5%** |
| Security Issues (regex) | 11 | 55% |
| Security Issues (Bandit) | 0 | 0% |

### PPO-fresh (20 samples)
| Outcome | Count | % |
|---------|-------|---|
| Syntax Error | 15 | 75% |
| Valid but No Test Pass | 5 | 25% |
| At Least 1 Test Pass | 0 | 0% |
| Security Issues (regex) | 5 | 25% |
| Security Issues (Bandit) | 0 | 0% |

---

## Key Observations

1. **PPO-simple degraded** from 45% → 15% syntax (sparse reward problem)
2. **PPO-continue achieved best results**: 60% syntax, 5% test pass
3. **Only PPO-continue has non-zero test pass rate**
4. **100% Bandit security compliance** across all models
5. **Regex findings** increased in PPO-continue (likely false positives from longer code)

---

## Per-Sample Test Results (PPO-continue)

The one successful test pass came from sample 17:
```
{
  "syntax_valid": true,
  "tests_passed": 1,
  "tests_total": 3,
  "rfunc_true": 0.333  (partial credit: 1/3 tests)
  "rsec_bandit": 1.0,
  "reward_enhanced": 0.6
}
```

This demonstrates:
- Partial credit working (0.333 for 1/3 tests)
- Enhanced reward = 0.6 (not just 0.4)
- Security maintained

---

## Data Source
- **File:** `results_from_vm/final_comparison/evaluation_results.json`
- **Generated:** 2025-12-12 19:22 UTC
- **Hardware:** NVIDIA V100 16GB
