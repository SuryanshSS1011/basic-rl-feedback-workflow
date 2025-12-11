"""
Supervised Fine-Tuning (SFT) Trainer

Phase 1: Behavior Cloning
Trains the policy model on high-reward samples from benchmark results.

Uses LoRA/PEFT for memory-efficient fine-tuning.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import SFTConfig, ModelConfig, TrainingConfig
from .data_converter import TrainingDataset, TrainingSample


@dataclass
class SFTTrainingArgs:
    """Arguments for SFT training"""

    output_dir: Path
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    fp16: bool = True
    bf16: bool = False

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


class CodeSFTDataset(Dataset):
    """PyTorch Dataset for code generation SFT"""

    def __init__(
        self,
        samples: List[TrainingSample],
        tokenizer: Any,
        max_length: int = 2048
    ):
        """
        Initialize dataset.

        Args:
            samples: List of TrainingSample objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Format as instruction-following
        full_text = self._format_sample(sample)

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels (same as input_ids for causal LM)
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()

        # Mask prompt tokens (only train on completion)
        prompt_text = self._format_prompt(sample.prompt)
        prompt_tokens = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prompt_len = len(prompt_tokens)

        # Set prompt tokens to -100 (ignored in loss)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt part"""
        return f"### Instruction:\nComplete the following code:\n\n{prompt}\n\n### Response:\n"

    def _format_sample(self, sample: TrainingSample) -> str:
        """Format full sample (prompt + completion)"""
        prompt_formatted = self._format_prompt(sample.prompt)
        return f"{prompt_formatted}{sample.completion}"


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for code generation.

    Phase 1 of the RL pipeline: Train on high-reward demonstrations.

    Usage:
        trainer = SFTTrainer(
            model_config=ModelConfig(),
            training_args=SFTTrainingArgs(output_dir="./sft_output")
        )
        trainer.train(train_dataset, val_dataset)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_args: SFTTrainingArgs,
        sft_config: Optional[SFTConfig] = None,
    ):
        """
        Initialize SFT trainer.

        Args:
            model_config: Model configuration (includes LoRA settings)
            training_args: Training arguments
            sft_config: Additional SFT configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for training. "
                "Install with: pip install torch"
            )

        self.model_config = model_config
        self.training_args = training_args
        self.sft_config = sft_config or SFTConfig()

        self.model = None
        self.tokenizer = None
        self.optimizer = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model_and_tokenizer(self):
        """Load model with LoRA and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.model_config.model_path

        print(f"Loading model: {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        dtype = torch.float16 if self.training_args.fp16 else torch.float32
        if self.training_args.bf16:
            dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )

        # Apply LoRA if configured
        if self.model_config.use_peft:
            self._apply_lora()

        # Gradient checkpointing
        if self.model_config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _apply_lora(self):
        """Apply LoRA adapters to model"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
                target_modules=self.model_config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )

            self.model = get_peft_model(self.model, lora_config)

            # Print trainable parameters
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

        except ImportError:
            raise ImportError(
                "PEFT is required for LoRA. Install with: pip install peft"
            )

    def _create_optimizer(self):
        """Create AdamW optimizer"""
        from torch.optim import AdamW

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay,
        )

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup"""
        from transformers import get_linear_schedule_with_warmup

        num_warmup_steps = int(
            num_training_steps * self.training_args.warmup_ratio
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(
        self,
        train_dataset: TrainingDataset,
        val_dataset: Optional[TrainingDataset] = None,
    ) -> Dict[str, float]:
        """
        Train the model using SFT.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)

        Returns:
            Dictionary of training metrics
        """
        # Load model
        self._load_model_and_tokenizer()

        # Create PyTorch datasets
        train_ds = CodeSFTDataset(
            samples=train_dataset.samples,
            tokenizer=self.tokenizer,
            max_length=self.training_args.max_seq_length,
        )

        val_ds = None
        if val_dataset:
            val_ds = CodeSFTDataset(
                samples=val_dataset.samples,
                tokenizer=self.tokenizer,
                max_length=self.training_args.max_seq_length,
            )

        # Create data loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.training_args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = None
        if val_ds:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.training_args.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Setup optimizer and scheduler
        num_training_steps = (
            len(train_loader)
            * self.training_args.num_epochs
            // self.training_args.gradient_accumulation_steps
        )
        self._create_optimizer()
        self._create_scheduler(num_training_steps)

        # Training loop
        print(f"\nStarting SFT training:")
        print(f"  - Train samples: {len(train_ds)}")
        print(f"  - Validation samples: {len(val_ds) if val_ds else 0}")
        print(f"  - Epochs: {self.training_args.num_epochs}")
        print(f"  - Batch size: {self.training_args.batch_size}")
        print(f"  - Gradient accumulation: {self.training_args.gradient_accumulation_steps}")
        print(f"  - Learning rate: {self.training_args.learning_rate}")
        print()

        metrics = self._training_loop(train_loader, val_loader)

        # Save final model
        self._save_checkpoint("final")

        return metrics

    def _training_loop(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
    ) -> Dict[str, float]:
        """Main training loop"""

        self.model.train()
        global_step = 0
        total_loss = 0.0
        best_val_loss = float("inf")
        metrics_history = []

        for epoch in range(self.training_args.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / self.training_args.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                epoch_loss += loss.item() * self.training_args.gradient_accumulation_steps
                num_batches += 1

                # Optimizer step
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.training_args.logging_steps == 0:
                        avg_loss = epoch_loss / num_batches
                        lr = self.scheduler.get_last_lr()[0]
                        print(
                            f"Epoch {epoch+1}/{self.training_args.num_epochs} | "
                            f"Step {global_step} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e}"
                        )

                    # Evaluation
                    if val_loader and global_step % self.training_args.eval_steps == 0:
                        val_loss = self._evaluate(val_loader)
                        print(f"  Validation loss: {val_loss:.4f}")
                        self.model.train()

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self._save_checkpoint("best")

                    # Save checkpoint
                    if global_step % self.training_args.save_steps == 0:
                        self._save_checkpoint(f"step_{global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}\n")
            metrics_history.append({
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
            })

        return {
            "final_loss": avg_epoch_loss,
            "best_val_loss": best_val_loss if val_loader else None,
            "total_steps": global_step,
            "history": metrics_history,
        }

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        output_dir = self.training_args.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model_config.use_peft:
            # Save only LoRA weights
            self.model.save_pretrained(output_dir)
        else:
            # Save full model
            self.model.save_pretrained(output_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # Save training args
        with open(output_dir / "training_args.json", "w") as f:
            json.dump({
                "learning_rate": self.training_args.learning_rate,
                "batch_size": self.training_args.batch_size,
                "num_epochs": self.training_args.num_epochs,
            }, f, indent=2)

        print(f"Checkpoint saved to: {output_dir}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint"""
        if self.model is None:
            self._load_model_and_tokenizer()

        if self.model_config.use_peft:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model.base_model.model,
                checkpoint_path,
            )
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

        print(f"Loaded checkpoint from: {checkpoint_path}")


def train_sft_from_config(config: TrainingConfig) -> Dict[str, float]:
    """
    Run SFT training from a TrainingConfig.

    Args:
        config: Full training configuration

    Returns:
        Training metrics
    """
    from .data_converter import BenchmarkConverter

    # Convert benchmark results to training data
    converter = BenchmarkConverter(
        results_dir=config.benchmark_results_dir,
        config=config.reward,
    )

    print("Loading benchmark results...")
    dataset = converter.convert_all()
    print(f"Loaded {len(dataset)} samples")

    # Filter and prepare data
    dataset = dataset.filter_by_reward(config.sft.min_reward_threshold)
    print(f"After reward filter: {len(dataset)} samples")

    if config.sft.select_best_per_prompt:
        dataset = dataset.select_best_per_prompt()
        print(f"After selecting best per prompt: {len(dataset)} samples")

    # Split into train/val
    dataset = dataset.shuffle(seed=42)
    train_dataset, val_dataset = dataset.split(train_ratio=0.9)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create trainer
    training_args = SFTTrainingArgs(
        output_dir=config.checkpoint_dir / "sft",
        num_epochs=config.sft.num_epochs,
        batch_size=config.sft.batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        learning_rate=config.sft.learning_rate,
        warmup_ratio=config.sft.warmup_ratio,
        weight_decay=config.sft.weight_decay,
        fp16=(config.model.mixed_precision == 'fp16'),
        bf16=(config.model.mixed_precision == 'bf16'),
    )

    trainer = SFTTrainer(
        model_config=config.model,
        training_args=training_args,
        sft_config=config.sft,
    )

    # Train
    metrics = trainer.train(train_dataset, val_dataset)

    return metrics
