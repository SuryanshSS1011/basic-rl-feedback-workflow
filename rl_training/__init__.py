"""
RL Training Module for Secure Code Generation

This module implements reinforcement learning training for code generation LLMs,
using diagnostic feedback from compilers, test execution, and security analyzers.

Components:
- config: Training hyperparameters and configuration
- reward_calculator: Compute R = α·Rfunc + β·Rsec
- scoring_agent: Hybrid rules + prompted LLM scorer
- security_weights: CVSS/CWE severity database
- data_converter: Convert benchmark results to training format
- sft_trainer: Supervised fine-tuning / behavior cloning (Phase 1)
- ppo_trainer: Proximal Policy Optimization (Phase 2)
- klee_integration: Symbolic execution for bug detection

Training Pipeline:
    1. Run benchmark to generate code samples with analysis
    2. Phase 1 (SFT): python train_sft.py --benchmark_dir ./benchmark/full_results
    3. Phase 2 (PPO): python train_ppo.py --sft_checkpoint ./checkpoints/sft/best
"""

from .config import (
    RLConfig,
    TrainingConfig,
    RewardConfig,
    PPOConfig,
    SFTConfig,
    ModelConfig,
)
from .reward_calculator import RewardCalculator, RewardResult, RewardBreakdown
from .security_weights import SecurityWeights, VulnerabilityInfo
from .scoring_agent import ScoringAgent, ScoringResult
from .data_converter import (
    BenchmarkConverter,
    TrainingDataset,
    TrainingSample,
    prepare_sft_data,
)

__all__ = [
    # Config
    'RLConfig',
    'TrainingConfig',
    'RewardConfig',
    'PPOConfig',
    'SFTConfig',
    'ModelConfig',
    # Reward
    'RewardCalculator',
    'RewardResult',
    'RewardBreakdown',
    # Security
    'SecurityWeights',
    'VulnerabilityInfo',
    # Scoring
    'ScoringAgent',
    'ScoringResult',
    # Data
    'BenchmarkConverter',
    'TrainingDataset',
    'TrainingSample',
    'prepare_sft_data',
]

__version__ = '0.1.0'
