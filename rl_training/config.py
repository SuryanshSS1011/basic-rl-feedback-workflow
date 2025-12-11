"""
Configuration for RL Training

Defines hyperparameters for:
- Reward computation (α, β weights)
- PPO training (learning rate, batch size, etc.)
- Model settings (LoRA, gradient checkpointing)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path


@dataclass
class RewardConfig:
    """Configuration for reward computation: R = α·Rfunc + β·Rsec"""

    # Reward weights (FIXED across all experiments)
    alpha: float = 0.6  # Weight for functional correctness
    beta: float = 0.4   # Weight for security reward

    # Security reward formula: 'exp' for exp(-V), 'linear' for 1 - min(V, 1)
    rsec_formula: Literal['exp', 'linear'] = 'exp'

    # KLEE bug severity weight (included in V calculation)
    klee_bug_weight: float = 0.8

    # Compilation failure penalty (Rfunc = 0 if code doesn't compile)
    compilation_failure_rfunc: float = 0.0

    def __post_init__(self):
        assert abs(self.alpha + self.beta - 1.0) < 1e-6, "α + β must equal 1"
        assert 0 <= self.alpha <= 1, "α must be in [0, 1]"
        assert 0 <= self.beta <= 1, "β must be in [0, 1]"


@dataclass
class PPOConfig:
    """Configuration for Proximal Policy Optimization"""

    # Learning rate
    learning_rate: float = 1e-5

    # Batch sizes
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # PPO hyperparameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5

    # KL divergence
    target_kl: Optional[float] = 0.01
    kl_penalty_coefficient: float = 0.1

    # Advantage computation
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning (Behavior Cloning)"""

    # Learning rate
    learning_rate: float = 2e-5

    # Batch size
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Training
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Data filtering
    min_reward_threshold: float = 0.5  # Only use samples with R > threshold
    select_best_per_prompt: bool = True  # Use only best output per prompt


@dataclass
class ModelConfig:
    """Configuration for policy model with LoRA"""

    # Supported models
    SUPPORTED_MODELS = {
        'deepseek-1.3b': 'deepseek-ai/deepseek-coder-1.3b-instruct',
        'deepseek-6.7b': 'deepseek-ai/deepseek-coder-6.7b-base',
    }

    # Model selection
    model_name: str = 'deepseek-1.3b'

    # LoRA configuration
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Memory optimization
    use_gradient_checkpointing: bool = True
    mixed_precision: Literal['no', 'fp16', 'bf16'] = 'fp16'

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

    @property
    def model_path(self) -> str:
        return self.SUPPORTED_MODELS.get(self.model_name, self.model_name)


@dataclass
class TrainingConfig:
    """Complete training configuration"""

    # Sub-configurations
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Training phases
    run_sft: bool = True   # Phase 1: Behavior cloning
    run_ppo: bool = True   # Phase 2: Online PPO

    # Paths
    benchmark_results_dir: Path = field(
        default_factory=lambda: Path("./benchmark/full_results")
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: Path("./checkpoints")
    )

    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    use_wandb: bool = False
    wandb_project: str = "secure-code-rl"

    # Training duration
    total_episodes: int = 10000
    eval_episodes: int = 100

    # Device
    device: str = "cuda"

    def __post_init__(self):
        self.benchmark_results_dir = Path(self.benchmark_results_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)


@dataclass
class RLConfig:
    """
    Main configuration class - backward compatible alias for TrainingConfig
    """

    # Reward weights
    alpha: float = 0.6
    beta: float = 0.4
    rsec_formula: Literal['exp', 'linear'] = 'exp'
    klee_bug_weight: float = 0.8

    # Model
    model_name: str = 'deepseek-1.3b'
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # PPO
    learning_rate: float = 1e-5
    batch_size: int = 4
    ppo_epochs: int = 4
    clip_range: float = 0.2

    # Paths
    benchmark_results_dir: str = "./benchmark/full_results"
    checkpoint_dir: str = "./checkpoints"

    def to_training_config(self) -> TrainingConfig:
        """Convert to full TrainingConfig"""
        return TrainingConfig(
            reward=RewardConfig(
                alpha=self.alpha,
                beta=self.beta,
                rsec_formula=self.rsec_formula,
                klee_bug_weight=self.klee_bug_weight,
            ),
            ppo=PPOConfig(
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                ppo_epochs=self.ppo_epochs,
                clip_range=self.clip_range,
            ),
            model=ModelConfig(
                model_name=self.model_name,
                use_peft=self.use_peft,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
            ),
            benchmark_results_dir=Path(self.benchmark_results_dir),
            checkpoint_dir=Path(self.checkpoint_dir),
        )
