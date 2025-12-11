"""
Proximal Policy Optimization (PPO) Trainer

Phase 2: Online RL Training with Tool Feedback

Implements PPO with:
- Policy model (πθ) generates code
- Scoring agent (πϕ) provides rewards from compiler/tests/security tools
- PPO updates to maximize R = α·Rfunc + β·Rsec

Reference: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import PPOConfig, ModelConfig, TrainingConfig, RewardConfig
from .reward_calculator import RewardCalculator, RewardResult
from .scoring_agent import ScoringAgent, ScoringResult


@dataclass
class RolloutSample:
    """A single rollout sample from policy"""

    prompt: str
    completion: str
    log_probs: List[float]
    reward: float
    rfunc: float
    rsec: float
    value: float = 0.0  # From value head
    advantage: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'prompt': self.prompt,
            'completion': self.completion,
            'reward': self.reward,
            'rfunc': self.rfunc,
            'rsec': self.rsec,
            'advantage': self.advantage,
        }


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout experience"""

    samples: List[RolloutSample] = field(default_factory=list)

    def add(self, sample: RolloutSample):
        self.samples.append(sample)

    def clear(self):
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages (simplified for single-step rewards)"""
        # For code generation, each completion is a single "episode"
        # Advantage = reward - baseline
        rewards = [s.reward for s in self.samples]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        for sample in self.samples:
            sample.advantage = sample.reward - mean_reward

    def normalize_advantages(self):
        """Normalize advantages to zero mean, unit variance"""
        if not self.samples:
            return

        advantages = [s.advantage for s in self.samples]
        mean = sum(advantages) / len(advantages)
        variance = sum((a - mean) ** 2 for a in advantages) / len(advantages)
        std = variance ** 0.5 + 1e-8

        for sample in self.samples:
            sample.advantage = (sample.advantage - mean) / std

    def get_batch(self, batch_size: int):
        """Get random mini-batches for training"""
        import random
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            yield [self.samples[i] for i in batch_indices]


class PolicyModel:
    """
    Wrapper for policy model with LoRA.

    Handles generation and log probability computation.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        device: str = "cuda"
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.model_config = model_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.ref_model = None  # Frozen reference for KL

    def load(self, checkpoint_path: Optional[Path] = None):
        """Load model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.model_config.model_path

        print(f"Loading policy model: {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Apply LoRA if configured
        if self.model_config.use_peft:
            self._apply_lora(checkpoint_path)

        # Load frozen reference model for KL penalty
        self._load_reference_model(model_path)

        if self.model_config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _apply_lora(self, checkpoint_path: Optional[Path] = None):
        """Apply LoRA adapters"""
        from peft import LoraConfig, get_peft_model, TaskType

        if checkpoint_path and checkpoint_path.exists():
            # Load existing LoRA weights
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            print(f"Loaded LoRA weights from: {checkpoint_path}")
        else:
            # Initialize new LoRA
            lora_config = LoraConfig(
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
                target_modules=self.model_config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _load_reference_model(self, model_path: str):
        """Load frozen reference model for KL computation"""
        from transformers import AutoModelForCausalLM

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[str, List[float]]:
        """
        Generate completion and return log probabilities.

        Returns:
            Tuple of (completion_text, log_probs_per_token)
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        # Generate with scores
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract generated tokens
        generated_ids = outputs.sequences[0, prompt_len:]
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probabilities
        log_probs = []
        for i, score in enumerate(outputs.scores):
            probs = F.softmax(score[0], dim=-1)
            token_id = generated_ids[i].item()
            log_prob = torch.log(probs[token_id] + 1e-10).item()
            log_probs.append(log_prob)

        return completion, log_probs

    def compute_log_probs(
        self,
        prompt: str,
        completion: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities for a given completion.

        Returns:
            Tuple of (policy_log_probs, reference_log_probs)
        """
        full_text = prompt + completion

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Policy log probs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, prompt_len-1:-1, :]
            probs = F.softmax(logits, dim=-1)
            target_ids = inputs["input_ids"][:, prompt_len:]

            policy_log_probs = torch.log(
                probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) + 1e-10
            )

        # Reference log probs
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
            ref_probs = F.softmax(ref_logits, dim=-1)
            ref_log_probs = torch.log(
                ref_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) + 1e-10
            )

        return policy_log_probs, ref_log_probs

    def save(self, path: Path):
        """Save model checkpoint"""
        path.mkdir(parents=True, exist_ok=True)

        if self.model_config.use_peft:
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

        self.tokenizer.save_pretrained(path)


class ToolFeedbackCollector:
    """
    Collects feedback from compilation, testing, and security tools.

    Wraps the scoring agent with actual tool execution.
    """

    def __init__(
        self,
        reward_config: RewardConfig,
        use_llm_for_security: bool = False,  # Disable LLM by default for speed
        temp_dir: Optional[Path] = None,
    ):
        self.scoring_agent = ScoringAgent(
            config=reward_config,
            use_llm_for_security=use_llm_for_security,
        )
        self.reward_calculator = RewardCalculator(config=reward_config)
        self.temp_dir = temp_dir or Path("/tmp/ppo_training")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_reward(self, code: str, test_cases: Optional[Dict] = None) -> RewardResult:
        """
        Get reward for generated code by running tools.

        Args:
            code: Generated code
            test_cases: Optional test cases to run

        Returns:
            RewardResult with rfunc and rsec
        """
        import subprocess
        import tempfile
        import os

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.c',
            dir=self.temp_dir,
            delete=False
        ) as f:
            f.write(code)
            code_file = f.name

        try:
            # 1. Compile with GCC
            gcc_result = subprocess.run(
                ['gcc', '-c', '-Wall', '-Wextra', code_file, '-o', '/dev/null'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            gcc_output = gcc_result.stderr

            # Parse compilation
            compilation = self.scoring_agent.rules_engine.parse_gcc_output(gcc_output)

            # 2. Run tests if provided
            test_output = {'passed': 0, 'failed': 0, 'total_tests': 0}
            if test_cases and compilation.compiles:
                test_output = self._run_tests(code_file, test_cases)

            # 3. Security analysis (simplified - no CodeQL in online loop)
            # In production, you'd integrate CodeQL or use static heuristics
            codeql_findings = self._quick_security_check(code)

            # Compute reward
            result = self.reward_calculator.compute_reward(
                compiles=compilation.compiles,
                tests_passed=test_output.get('passed', 0),
                tests_total=test_output.get('total_tests', 0),
                codeql_findings=codeql_findings,
                klee_bugs=[],  # KLEE too slow for online RL
            )

            return result

        except subprocess.TimeoutExpired:
            # Compilation timeout
            return RewardResult(reward=0.0, rfunc=0.0, rsec=1.0)

        except Exception as e:
            print(f"Error getting reward: {e}")
            return RewardResult(reward=0.0, rfunc=0.0, rsec=1.0)

        finally:
            # Cleanup
            try:
                os.unlink(code_file)
            except OSError:
                pass

    def _run_tests(self, code_file: str, test_cases: Dict) -> Dict:
        """Run test cases against compiled code"""
        # Simplified - would need actual test runner
        return {'passed': 0, 'failed': 0, 'total_tests': 0}

    def _quick_security_check(self, code: str) -> List[Dict]:
        """Quick regex-based security checks for online RL"""
        import re

        findings = []

        # Check for dangerous patterns
        patterns = [
            (r'gets\s*\(', 'cpp/dangerous-gets', 'Use of dangerous gets()'),
            (r'strcpy\s*\(', 'cpp/strcpy-no-bounds', 'strcpy without bounds check'),
            (r'sprintf\s*\(', 'cpp/sprintf-overflow', 'sprintf without size limit'),
            (r'system\s*\(', 'cpp/command-injection', 'Potential command injection'),
            (r'exec[vl]?[pe]?\s*\(', 'cpp/command-injection', 'Potential command injection'),
            (r'\bfree\b.*\bfree\b', 'cpp/double-free', 'Potential double free'),
        ]

        for pattern, rule_id, message in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                findings.append({'rule_id': rule_id, 'message': message})

        return findings


class PPOTrainer:
    """
    PPO Trainer for code generation.

    Phase 2: Online RL with tool feedback.

    Usage:
        trainer = PPOTrainer(config)
        trainer.load_sft_checkpoint(sft_path)
        trainer.train(prompts)
    """

    def __init__(
        self,
        config: TrainingConfig,
        prompts: Optional[List[str]] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            config: Full training configuration
            prompts: List of prompts for RL training
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config
        self.prompts = prompts or []

        self.policy = PolicyModel(config.model, device=config.device)
        self.feedback = ToolFeedbackCollector(
            reward_config=config.reward,
            use_llm_for_security=False,
        )

        self.optimizer = None
        self.global_step = 0
        self.best_mean_reward = -float('inf')

    def load_sft_checkpoint(self, checkpoint_path: Path):
        """Load policy from SFT checkpoint"""
        self.policy.load(checkpoint_path)
        self._setup_optimizer()

    def _setup_optimizer(self):
        """Setup AdamW optimizer"""
        from torch.optim import AdamW

        self.optimizer = AdamW(
            self.policy.model.parameters(),
            lr=self.config.ppo.learning_rate,
        )

    def train(
        self,
        num_episodes: Optional[int] = None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run PPO training loop.

        Args:
            num_episodes: Number of episodes to train
            prompts: Prompts to train on (overrides constructor prompts)

        Returns:
            Training metrics
        """
        prompts = prompts or self.prompts
        num_episodes = num_episodes or self.config.total_episodes

        if not prompts:
            raise ValueError("No prompts provided for training")

        print(f"\nStarting PPO training:")
        print(f"  - Prompts: {len(prompts)}")
        print(f"  - Episodes: {num_episodes}")
        print(f"  - Batch size: {self.config.ppo.batch_size}")
        print(f"  - PPO epochs: {self.config.ppo.ppo_epochs}")
        print()

        metrics_history = []
        prompt_idx = 0

        for episode in range(num_episodes):
            # Collect rollouts
            rollout_buffer = self._collect_rollouts(
                prompts=prompts,
                num_samples=self.config.ppo.batch_size,
                start_idx=prompt_idx,
            )
            prompt_idx = (prompt_idx + self.config.ppo.batch_size) % len(prompts)

            # Compute advantages
            rollout_buffer.compute_advantages(
                gamma=self.config.ppo.gamma,
                gae_lambda=self.config.ppo.gae_lambda,
            )

            if self.config.ppo.normalize_advantages:
                rollout_buffer.normalize_advantages()

            # PPO update
            metrics = self._ppo_update(rollout_buffer)
            metrics['episode'] = episode + 1
            metrics_history.append(metrics)

            self.global_step += 1

            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                print(
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Reward: {metrics['mean_reward']:.4f} | "
                    f"Rfunc: {metrics['mean_rfunc']:.4f} | "
                    f"Rsec: {metrics['mean_rsec']:.4f} | "
                    f"Loss: {metrics['policy_loss']:.4f}"
                )

            # Save checkpoint
            if (episode + 1) % self.config.save_interval == 0:
                self._save_checkpoint(f"episode_{episode+1}")

            # Track best
            if metrics['mean_reward'] > self.best_mean_reward:
                self.best_mean_reward = metrics['mean_reward']
                self._save_checkpoint("best")

        # Final save
        self._save_checkpoint("final")

        return {
            'best_mean_reward': self.best_mean_reward,
            'final_reward': metrics_history[-1]['mean_reward'] if metrics_history else 0,
            'history': metrics_history,
        }

    def _collect_rollouts(
        self,
        prompts: List[str],
        num_samples: int,
        start_idx: int = 0,
    ) -> RolloutBuffer:
        """Collect rollout samples from policy"""

        buffer = RolloutBuffer()

        for i in range(num_samples):
            prompt_idx = (start_idx + i) % len(prompts)
            prompt = prompts[prompt_idx]

            # Format prompt
            formatted_prompt = (
                f"### Instruction:\nComplete the following code:\n\n"
                f"{prompt}\n\n### Response:\n"
            )

            # Generate completion
            completion, log_probs = self.policy.generate(
                prompt=formatted_prompt,
                max_new_tokens=self.config.model.max_new_tokens,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
            )

            # Get reward from tools
            reward_result = self.feedback.get_reward(completion)

            sample = RolloutSample(
                prompt=formatted_prompt,
                completion=completion,
                log_probs=log_probs,
                reward=reward_result.reward,
                rfunc=reward_result.rfunc,
                rsec=reward_result.rsec,
            )

            buffer.add(sample)

        return buffer

    def _ppo_update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Perform PPO policy update"""

        self.policy.model.train()

        total_policy_loss = 0.0
        total_kl = 0.0
        num_updates = 0

        for ppo_epoch in range(self.config.ppo.ppo_epochs):
            for batch in buffer.get_batch(self.config.ppo.mini_batch_size):
                # Compute current and old log probs
                batch_loss = 0.0
                batch_kl = 0.0

                for sample in batch:
                    # Get log probs
                    policy_log_probs, ref_log_probs = self.policy.compute_log_probs(
                        sample.prompt, sample.completion
                    )

                    # Old log probs (from generation)
                    old_log_probs = torch.tensor(
                        sample.log_probs[:policy_log_probs.shape[1]],
                        device=policy_log_probs.device,
                    )

                    # Ratio
                    ratio = torch.exp(policy_log_probs - old_log_probs).mean()

                    # Clipped objective
                    advantage = sample.advantage
                    clip_range = self.config.ppo.clip_range

                    unclipped = ratio * advantage
                    clipped = torch.clamp(
                        ratio, 1 - clip_range, 1 + clip_range
                    ) * advantage

                    policy_loss = -torch.min(unclipped, clipped)

                    # KL penalty
                    kl = (policy_log_probs - ref_log_probs).mean()
                    kl_penalty = self.config.ppo.kl_penalty_coefficient * kl

                    batch_loss += policy_loss + kl_penalty
                    batch_kl += kl.item()

                # Average over batch
                batch_loss = batch_loss / len(batch)

                # Backward
                self.optimizer.zero_grad()
                batch_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.model.parameters(),
                    self.config.ppo.max_grad_norm,
                )

                self.optimizer.step()

                total_policy_loss += batch_loss.item()
                total_kl += batch_kl / len(batch)
                num_updates += 1

                # Early stopping on KL
                if self.config.ppo.target_kl and batch_kl / len(batch) > self.config.ppo.target_kl:
                    break

        # Compute metrics
        rewards = [s.reward for s in buffer.samples]
        rfuncs = [s.rfunc for s in buffer.samples]
        rsecs = [s.rsec for s in buffer.samples]

        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'kl': total_kl / max(num_updates, 1),
            'mean_reward': sum(rewards) / len(rewards),
            'mean_rfunc': sum(rfuncs) / len(rfuncs),
            'mean_rsec': sum(rsecs) / len(rsecs),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
        }

    def _save_checkpoint(self, name: str):
        """Save checkpoint"""
        path = self.config.checkpoint_dir / "ppo" / name
        self.policy.save(path)

        # Save training state
        state = {
            'global_step': self.global_step,
            'best_mean_reward': self.best_mean_reward,
        }
        with open(path / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Saved checkpoint: {path}")


def train_ppo_from_config(
    config: TrainingConfig,
    sft_checkpoint: Path,
    prompts: List[str],
) -> Dict[str, Any]:
    """
    Run PPO training from config.

    Args:
        config: Training configuration
        sft_checkpoint: Path to SFT checkpoint
        prompts: Training prompts

    Returns:
        Training metrics
    """
    trainer = PPOTrainer(config, prompts)
    trainer.load_sft_checkpoint(sft_checkpoint)
    return trainer.train()
