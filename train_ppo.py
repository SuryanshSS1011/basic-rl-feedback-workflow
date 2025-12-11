#!/usr/bin/env python3
"""
Phase 2: Online PPO Training with Tool Feedback

Train the policy model using Proximal Policy Optimization with
rewards from compiler, tests, and security tools.

Usage:
    python train_ppo.py --sft_checkpoint ./checkpoints/sft/best

    # With custom settings
    python train_ppo.py \
        --sft_checkpoint ./checkpoints/sft/best \
        --prompts_file ./data/prompts.txt \
        --episodes 1000 \
        --alpha 0.6 --beta 0.4
"""

import argparse
from pathlib import Path
from typing import List

from rl_training.config import (
    TrainingConfig,
    RewardConfig,
    PPOConfig,
    ModelConfig,
)
from rl_training.ppo_trainer import PPOTrainer, train_ppo_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: PPO Training with Tool Feedback"
    )

    # Checkpoints
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        required=True,
        help="Path to SFT checkpoint from Phase 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/ppo",
        help="Output directory for PPO checkpoints",
    )

    # Data
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="File with prompts for RL training (one per line)",
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="./benchmark/full_results",
        help="Benchmark directory to extract prompts from",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=1000,
        help="Maximum number of prompts to use",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-1.3b",
        choices=["deepseek-1.3b", "deepseek-6.7b"],
        help="Model architecture (must match SFT checkpoint)",
    )

    # Reward weights
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Weight for functional correctness (Rfunc)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
        help="Weight for security reward (Rsec)",
    )
    parser.add_argument(
        "--rsec_formula",
        type=str,
        default="exp",
        choices=["exp", "linear"],
        help="Formula for Rsec: 'exp' for exp(-V), 'linear' for 1-min(V,1)",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Rollout batch size",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=2,
        help="Mini-batch size for PPO updates",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=4,
        help="Number of PPO epochs per batch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.2,
        help="PPO clipping range",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=0.01,
        help="Target KL divergence for early stopping",
    )
    parser.add_argument(
        "--kl_penalty",
        type=float,
        default=0.1,
        help="KL penalty coefficient",
    )

    # Generation
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )

    # LoRA
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )

    # Logging
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N episodes",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Save checkpoint every N episodes",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )

    return parser.parse_args()


def load_prompts(
    prompts_file: str = None,
    benchmark_dir: str = None,
    max_prompts: int = 1000,
) -> List[str]:
    """Load prompts from file or extract from benchmark results"""

    prompts = []

    # Try loading from file first
    if prompts_file:
        prompts_path = Path(prompts_file)
        if prompts_path.exists():
            with open(prompts_path) as f:
                prompts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(prompts)} prompts from {prompts_file}")

    # Extract from benchmark results if no file
    if not prompts and benchmark_dir:
        benchmark_path = Path(benchmark_dir)
        if benchmark_path.exists():
            # Find all prompt.txt files
            for prompt_file in benchmark_path.glob("**/prompt.txt"):
                try:
                    with open(prompt_file) as f:
                        prompt = f.read().strip()
                    if prompt:
                        prompts.append(prompt)
                except OSError:
                    pass

            # Deduplicate
            prompts = list(set(prompts))
            print(f"Extracted {len(prompts)} unique prompts from benchmark results")

    # Limit number of prompts
    if len(prompts) > max_prompts:
        import random
        random.shuffle(prompts)
        prompts = prompts[:max_prompts]
        print(f"Limited to {max_prompts} prompts")

    return prompts


def main():
    args = parse_args()

    print("=" * 60)
    print("Phase 2: PPO Training with Tool Feedback")
    print("=" * 60)

    # Validate SFT checkpoint
    sft_path = Path(args.sft_checkpoint)
    if not sft_path.exists():
        print(f"\nError: SFT checkpoint not found at {sft_path}")
        print("Run Phase 1 (train_sft.py) first.")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print("\nLoading prompts...")
    prompts = load_prompts(
        prompts_file=args.prompts_file,
        benchmark_dir=args.benchmark_dir,
        max_prompts=args.max_prompts,
    )

    if not prompts:
        print("\nError: No prompts found.")
        print("Provide --prompts_file or ensure --benchmark_dir contains prompt.txt files")
        return 1

    # Create configuration
    reward_config = RewardConfig(
        alpha=args.alpha,
        beta=args.beta,
        rsec_formula=args.rsec_formula,
    )

    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        kl_penalty_coefficient=args.kl_penalty,
    )

    model_config = ModelConfig(
        model_name=args.model,
        use_peft=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    config = TrainingConfig(
        reward=reward_config,
        ppo=ppo_config,
        model=model_config,
        checkpoint_dir=output_dir,
        total_episodes=args.episodes,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
    )

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  SFT checkpoint: {sft_path}")
    print(f"  Model: {model_config.model_path}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Reward weights: α={args.alpha}, β={args.beta}")
    print(f"  Rsec formula: {args.rsec_formula}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  PPO epochs: {args.ppo_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Clip range: {args.clip_range}")
    print(f"  Output: {output_dir}")

    # Create trainer
    print("\nInitializing PPO trainer...")
    trainer = PPOTrainer(config, prompts)

    # Load SFT checkpoint
    print(f"Loading SFT checkpoint from: {sft_path}")
    trainer.load_sft_checkpoint(sft_path)

    # Train
    print("\nStarting PPO training...")
    print("(This will run compiler/test feedback for each generation)")
    print()

    metrics = trainer.train()

    # Print results
    print("\n" + "=" * 60)
    print("PPO Training Complete!")
    print("=" * 60)
    print(f"Best mean reward: {metrics.get('best_mean_reward', 'N/A'):.4f}")
    print(f"Final mean reward: {metrics.get('final_reward', 'N/A'):.4f}")
    print(f"\nCheckpoints saved to: {output_dir}")
    print(f"Best model: {output_dir / 'ppo' / 'best'}")
    print(f"Final model: {output_dir / 'ppo' / 'final'}")

    # Save training summary
    import json
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'sft_checkpoint': str(sft_path),
            'model': args.model,
            'episodes': args.episodes,
            'alpha': args.alpha,
            'beta': args.beta,
            'best_mean_reward': metrics.get('best_mean_reward'),
            'final_reward': metrics.get('final_reward'),
        }, f, indent=2)
    print(f"\nTraining summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    exit(main())
