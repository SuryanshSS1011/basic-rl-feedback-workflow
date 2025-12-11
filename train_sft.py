#!/usr/bin/env python3
"""
Phase 1: Supervised Fine-Tuning (Behavior Cloning)

Train the policy model on high-reward samples from benchmark results.

Usage:
    python train_sft.py --benchmark_dir ./benchmark/full_results --output_dir ./checkpoints/sft

    # With custom settings
    python train_sft.py \
        --model deepseek-1.3b \
        --min_reward 0.6 \
        --epochs 3 \
        --batch_size 4
"""

import argparse
from pathlib import Path

from rl_training.config import (
    TrainingConfig,
    RewardConfig,
    SFTConfig,
    ModelConfig,
)
from rl_training.data_converter import BenchmarkConverter, prepare_sft_data
from rl_training.sft_trainer import SFTTrainer, SFTTrainingArgs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: SFT / Behavior Cloning Training"
    )

    # Data paths
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="./benchmark/full_results",
        help="Path to benchmark results directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/sft",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/sft",
        help="Directory for processed training data",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-1.3b",
        choices=["deepseek-1.3b", "deepseek-6.7b"],
        help="Model to train",
    )

    # Data filtering
    parser.add_argument(
        "--min_reward",
        type=float,
        default=0.5,
        help="Minimum reward threshold for training samples",
    )
    parser.add_argument(
        "--no_select_best",
        action="store_true",
        help="Don't select best completion per prompt (use all samples)",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )

    # LoRA
    parser.add_argument(
        "--no_peft",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
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

    # Mixed precision
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 training",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 training (overrides fp16)",
    )

    # Logging
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Evaluate every N steps",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Phase 1: Supervised Fine-Tuning (Behavior Cloning)")
    print("=" * 60)

    # Create directories
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    benchmark_dir = Path(args.benchmark_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check benchmark results exist
    if not benchmark_dir.exists():
        print(f"\nError: Benchmark results not found at {benchmark_dir}")
        print("Run the benchmark first to generate training data.")
        return 1

    # Prepare training data
    print(f"\nPreparing training data from: {benchmark_dir}")

    train_path, val_path = prepare_sft_data(
        results_dir=benchmark_dir,
        output_path=data_dir,
        min_reward=args.min_reward,
        select_best=not args.no_select_best,
        train_ratio=0.9,
        seed=42,
    )

    # Load datasets
    from rl_training.data_converter import TrainingDataset

    train_dataset = TrainingDataset.from_jsonl(train_path)
    val_dataset = TrainingDataset.from_jsonl(val_path)

    if len(train_dataset) == 0:
        print("\nError: No training samples after filtering.")
        print(f"Try lowering --min_reward (current: {args.min_reward})")
        return 1

    # Configure model
    model_config = ModelConfig(
        model_name=args.model,
        use_peft=not args.no_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        mixed_precision='bf16' if args.bf16 else ('fp16' if args.fp16 else 'no'),
    )

    # Configure training
    training_args = SFTTrainingArgs(
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
    )

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Model: {model_config.model_path}")
    print(f"  LoRA: {'enabled' if model_config.use_peft else 'disabled'}")
    if model_config.use_peft:
        print(f"    - rank: {model_config.lora_r}")
        print(f"    - alpha: {model_config.lora_alpha}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output: {output_dir}")

    # Create trainer
    trainer = SFTTrainer(
        model_config=model_config,
        training_args=training_args,
    )

    # Train
    print("\nStarting training...")
    metrics = trainer.train(train_dataset, val_dataset)

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {metrics.get('final_loss', 'N/A'):.4f}")
    if metrics.get('best_val_loss'):
        print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
    print(f"Total steps: {metrics.get('total_steps', 'N/A')}")
    print(f"\nCheckpoints saved to: {output_dir}")
    print(f"Best model: {output_dir / 'best'}")
    print(f"Final model: {output_dir / 'final'}")

    print("\nNext step: Run Phase 2 (PPO) training:")
    print(f"  python train_ppo.py --sft_checkpoint {output_dir / 'best'}")

    return 0


if __name__ == "__main__":
    exit(main())
