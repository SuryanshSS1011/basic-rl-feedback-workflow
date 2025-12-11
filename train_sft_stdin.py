#!/usr/bin/env python3
"""
SFT Training on Stdin-Style Subset

Trains DeepSeek-1.3B on the pre-filtered stdin-style subset using LoRA.

Usage:
    python train_sft_stdin.py

    # With custom settings
    python train_sft_stdin.py --epochs 3 --batch_size 4 --learning_rate 2e-5
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")


# Configuration
DEFAULT_CONFIG = {
    'model_name': 'deepseek-ai/deepseek-coder-1.3b-instruct',
    'data_dir': './data/sft',
    'output_dir': './checkpoints/sft_stdin',
    'epochs': 3,
    'batch_size': 4,
    'gradient_accumulation': 4,
    'learning_rate': 2e-5,
    'max_seq_length': 2048,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'fp16': True,
    'logging_steps': 10,
    'save_steps': 200,
    'eval_steps': 100,
}


class CodeDataset(Dataset):
    """PyTorch Dataset for code generation training"""

    def __init__(self, data_path: Path, tokenizer: Any, max_length: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSONL data
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Format as instruction-following
        prompt = sample['prompt']
        completion = sample['completion']

        formatted_prompt = f"### Instruction:\nComplete the following Python code:\n\n{prompt}\n\n### Response:\n"
        full_text = f"{formatted_prompt}{completion}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()

        # Mask prompt tokens (only train on completion)
        prompt_tokens = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prompt_len = len(prompt_tokens)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training on Stdin Subset")

    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG['model_name'])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG['data_dir'])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG['output_dir'])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument("--gradient_accumulation", type=int, default=DEFAULT_CONFIG['gradient_accumulation'])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_CONFIG['max_seq_length'])
    parser.add_argument("--lora_r", type=int, default=DEFAULT_CONFIG['lora_r'])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_CONFIG['lora_alpha'])
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--fp16", action="store_true", default=DEFAULT_CONFIG['fp16'])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_CONFIG['logging_steps'])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG['save_steps'])
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_CONFIG['eval_steps'])
    parser.add_argument("--dry_run", action="store_true", help="Check setup without training")

    return parser.parse_args()


def check_dependencies():
    """Check if required dependencies are available"""
    missing = []

    if not TORCH_AVAILABLE:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import peft
    except ImportError:
        missing.append("peft")

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True


def main():
    args = parse_args()

    print("=" * 70)
    print("SFT Training on Stdin-Style Subset")
    print("=" * 70)

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check data exists
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        print(f"\nError: Training data not found at {train_path}")
        print("Run prepare_training_data.py first.")
        return 1

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  LoRA: {'disabled' if args.no_lora else f'r={args.lora_r}, alpha={args.lora_alpha}'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation})")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  Precision: {'bf16' if args.bf16 else 'fp16' if args.fp16 else 'fp32'}")

    if args.dry_run:
        print("\n[DRY RUN] Setup check complete. Exiting without training.")
        return 0

    # Import training dependencies
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device and dtype
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        device_map = "auto"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS has limited dtype support
        device_map = None  # Manual device placement for MPS
        print("Using Apple MPS (Metal) acceleration")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        device_map = None
        print("Warning: No GPU available, using CPU (slow)")

    # Load model
    print(f"Loading model: {args.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    # Move to device if not using device_map
    if device_map is None:
        model = model.to(device)

    # Apply LoRA
    if not args.no_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=DEFAULT_CONFIG['lora_dropout'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Enable input gradients for LoRA training
    model.enable_input_require_grads()

    # Disable gradient checkpointing (V100 has enough VRAM)
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Create datasets
    print(f"\nLoading datasets...")
    train_dataset = CodeDataset(train_path, tokenizer, args.max_seq_length)
    val_dataset = CodeDataset(val_path, tokenizer, args.max_seq_length) if val_path.exists() else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # Setup optimizer and scheduler
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    num_warmup_steps = int(num_training_steps * 0.1)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Get device (use the one we determined earlier)

    # Training loop
    print(f"\nStarting training...")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset) if val_dataset else 0}")
    print(f"  Total steps: {num_training_steps}")
    print()

    model.train()
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation

            loss.backward()
            epoch_loss += loss.item() * args.gradient_accumulation
            num_batches += 1

            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}/{args.epochs} | Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

                # Evaluation
                if val_loader and global_step % args.eval_steps == 0:
                    model.eval()
                    val_loss = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            outputs = model(**val_batch)
                            val_loss += outputs.loss.item()
                            val_batches += 1
                    val_loss /= val_batches
                    print(f"  Validation loss: {val_loss:.4f}")
                    model.train()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # Save best model
                        best_dir = output_dir / "best"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        print(f"  New best model saved!")

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}\n")

    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training info
    with open(output_dir / "training_info.json", "w") as f:
        json.dump({
            'model_name': args.model_name,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'lora_r': args.lora_r if not args.no_lora else None,
            'final_loss': avg_epoch_loss,
            'best_val_loss': best_val_loss,
            'total_steps': global_step,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset) if val_dataset else 0,
        }, f, indent=2)

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final loss: {avg_epoch_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"\nNext step: Run PPO training with:")
    print(f"  python train_ppo.py --sft_checkpoint {output_dir / 'best'}")

    return 0


if __name__ == "__main__":
    exit(main())
