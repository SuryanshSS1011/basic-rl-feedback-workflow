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
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable log operation"""
    return torch.log(torch.clamp(x, min=eps))


def check_tensor_valid(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        warnings.warn(f"{name} contains NaN values")
        return False
    if torch.isinf(tensor).any():
        warnings.warn(f"{name} contains Inf values")
        return False
    return True


def sanitize_tensor(tensor: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    """Replace NaN and Inf values with default"""
    tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, default), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.full_like(tensor, default), tensor)
    return tensor

from .config import PPOConfig, ModelConfig, TrainingConfig, RewardConfig
from .reward_calculator import RewardCalculator, RewardResult
from .scoring_agent import ScoringAgent, ScoringResult
from .bandit_runner import BanditRunner, BanditFinding


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
        device: str = "cuda",
        use_reference_model: bool = True,  # Enable reference model for KL penalty
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.model_config = model_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.ref_model = None  # Frozen reference for KL (optional)
        self.use_reference_model = use_reference_model

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

        # Load frozen reference model for KL penalty (optional - saves ~3GB VRAM)
        if self.use_reference_model:
            self._load_reference_model(model_path)
        else:
            print("Skipping reference model to save VRAM (KL penalty disabled)")

        # Enable input gradients for LoRA training
        self.model.enable_input_require_grads()

        # Make LoRA parameters trainable
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        # Disable gradient checkpointing (causes issues with LoRA)
        # if self.model_config.use_gradient_checkpointing:
        #     self.model.gradient_checkpointing_enable()

    def _apply_lora(self, checkpoint_path: Optional[Path] = None):
        """Apply LoRA adapters"""
        from peft import LoraConfig, get_peft_model, TaskType

        if checkpoint_path and checkpoint_path.exists():
            # Load existing LoRA weights
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
                is_trainable=True,  # Critical: make LoRA trainable
            )
            print(f"Loaded LoRA weights from: {checkpoint_path}")

            # Explicitly enable training for LoRA parameters
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
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

        # Ensure LoRA parameters are trainable
        lora_params = 0
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                lora_params += param.numel()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"LoRA parameters: {lora_params:,}")

    def _load_reference_model(self, model_path: str):
        """Load frozen reference model for KL computation in 8-bit to save memory"""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        try:
            # Try loading in 8-bit to save VRAM
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
            )
            print("Reference model loaded in 8-bit quantization")
        except Exception as e:
            warnings.warn(f"8-bit loading failed ({e}), using float16")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
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
        max_retries: int = 3,
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

        # Try generation with fallback
        for attempt in range(max_retries):
            try:
                # Keep temperature stable - don't reduce too much
                current_temp = max(0.5, temperature - (attempt * 0.1))

                # Generate with scores
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=10,  # Force at least 10 tokens
                        temperature=current_temp,
                        top_p=top_p,
                        top_k=50,  # Add top_k for stability
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Discourage repetition
                    )

                # Extract generated tokens
                generated_ids = outputs.sequences[0, prompt_len:]

                if len(generated_ids) == 0:
                    warnings.warn(f"Attempt {attempt + 1}: No tokens generated")
                    continue

                completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Check for meaningful completion
                if len(completion.strip()) < 5:
                    warnings.warn(f"Attempt {attempt + 1}: Very short completion: '{completion[:50]}'")
                    if attempt < max_retries - 1:
                        continue

                # Compute log probabilities with stability checks
                log_probs = []
                for i, score in enumerate(outputs.scores):
                    if i >= len(generated_ids):
                        break

                    # Clamp scores to prevent numerical issues
                    score_clamped = torch.clamp(score[0].float(), min=-100, max=100)

                    # Use log_softmax for numerical stability
                    log_probs_all = F.log_softmax(score_clamped, dim=-1)
                    token_id = generated_ids[i].item()
                    log_prob = log_probs_all[token_id].item()

                    # Sanity check
                    if not (torch.isnan(torch.tensor(log_prob)) or torch.isinf(torch.tensor(log_prob))):
                        log_probs.append(log_prob)
                    else:
                        log_probs.append(-5.0)  # Default for invalid

                if len(log_probs) > 0:
                    # Clear memory after successful generation
                    del outputs
                    torch.cuda.empty_cache()
                    return completion, log_probs

            except RuntimeError as e:
                if "CUDA" in str(e) or "probability" in str(e) or "out of memory" in str(e).lower():
                    warnings.warn(f"Generation attempt {attempt + 1} failed: {e}")
                    torch.cuda.empty_cache()
                else:
                    raise

        # If all retries failed, return placeholder with penalty
        warnings.warn("All generation attempts failed")
        return "// Failed to generate", [-5.0]

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
        if not completion:
            # Empty completion - return dummy tensors
            device = self.device
            dummy = torch.tensor([[-10.0]], device=device, requires_grad=True)
            dummy_ref = torch.tensor([[-10.0]], device=device)
            return dummy, dummy_ref

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

        # Ensure we have completion tokens
        total_len = inputs["input_ids"].shape[1]
        if prompt_len >= total_len:
            # No completion tokens after prompt
            device = self.device
            dummy = torch.tensor([[-10.0]], device=device, requires_grad=True)
            dummy_ref = torch.tensor([[-10.0]], device=device)
            return dummy, dummy_ref

        target_ids = inputs["input_ids"][:, prompt_len:]

        # Policy log probs (WITH gradients for training)
        self.model.train()  # Ensure training mode for gradients
        outputs = self.model(**inputs)

        # Clamp logits for numerical stability
        logits = outputs.logits[:, prompt_len-1:-1, :]
        logits = torch.clamp(logits.float(), min=-100, max=100)

        # Use log_softmax for numerical stability (more stable than softmax + log)
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Gather log probs for target tokens
        policy_log_probs = log_probs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Sanitize any remaining NaN/Inf
        policy_log_probs = sanitize_tensor(policy_log_probs, default=-10.0)

        # Reference log probs (no gradients) - only if reference model is loaded
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(**inputs)
                ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
                ref_logits = torch.clamp(ref_logits.float(), min=-100, max=100)
                ref_log_probs_all = F.log_softmax(ref_logits, dim=-1)
                ref_log_probs = ref_log_probs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                ref_log_probs = sanitize_tensor(ref_log_probs, default=-10.0)
        else:
            # No reference model - use policy log probs as reference (no KL penalty)
            ref_log_probs = policy_log_probs.detach().clone()

        # Clear CUDA cache to free memory
        torch.cuda.empty_cache()

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

    Supports:
    - Bandit integration for comprehensive Python security analysis
    - Regex fallback for fast online training
    - Optional test execution for true Rfunc computation
    """

    def __init__(
        self,
        reward_config: RewardConfig,
        use_llm_for_security: bool = False,  # Disable LLM by default for speed
        use_bandit: bool = False,  # Use Bandit for comprehensive security analysis
        temp_dir: Optional[Path] = None,
    ):
        self.scoring_agent = ScoringAgent(
            config=reward_config,
            use_llm_for_security=use_llm_for_security,
        )
        self.reward_calculator = RewardCalculator(config=reward_config)
        self.temp_dir = temp_dir or Path("/tmp/ppo_training")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Bandit runner (falls back to regex if Bandit unavailable)
        self.bandit_runner = BanditRunner(use_bandit=use_bandit)
        self.use_bandit = use_bandit

    def get_reward(self, code: str, test_cases: Optional[Dict] = None) -> RewardResult:
        """
        Get reward for generated Python code.

        Args:
            code: Generated Python code
            test_cases: Optional test cases to run

        Returns:
            RewardResult with rfunc and rsec
        """
        import subprocess
        import tempfile
        import os
        import ast

        # Quick check: is there any actual code?
        if not code or len(code.strip()) < 10:
            return RewardResult(reward=0.0, rfunc=0.0, rsec=1.0)

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',  # Python file
            dir=self.temp_dir,
            delete=False
        ) as f:
            f.write(code)
            code_file = f.name

        try:
            # 1. Check Python syntax using AST parse
            syntax_valid = False
            try:
                ast.parse(code)
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False

            # 2. Also try py_compile for additional validation
            if syntax_valid:
                compile_result = subprocess.run(
                    ['python', '-m', 'py_compile', code_file],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                syntax_valid = (compile_result.returncode == 0)

            # 3. Security analysis for Python
            # Use Bandit if enabled, otherwise fall back to quick regex check
            if self.use_bandit:
                bandit_findings = self.bandit_runner.analyze(code)
                rsec = self.bandit_runner.compute_rsec(
                    bandit_findings,
                    formula=self.reward_calculator.config.rsec_formula
                )
                security_findings = [f.to_dict() for f in bandit_findings]
            else:
                security_findings = self._quick_python_security_check(code)
                # Security reward: penalize dangerous patterns
                num_findings = len(security_findings)
                if num_findings == 0:
                    rsec = 1.0
                else:
                    # Exponential decay based on findings
                    rsec = max(0.0, 1.0 - (0.2 * num_findings))

            # Compute functional reward with PARTIAL CREDIT
            # If test cases provided, run them for true Rfunc with partial credit
            # Otherwise, use syntax validity as fallback (can't evaluate without tests)
            if syntax_valid and test_cases and test_cases.get('inputs'):
                # Partial credit system: 0.2 (syntax) + 0.2 (runs) + 0.2 (output) + 0.4*(pass rate)
                rfunc = self._run_tests_for_rfunc(code_file, test_cases)
            else:
                # Fallback when no tests available: syntax valid = 1.0, invalid = 0.0
                # (Can't apply partial credit without tests to run)
                rfunc = 1.0 if syntax_valid else 0.0

            # Combined reward
            alpha = self.reward_calculator.config.alpha
            beta = self.reward_calculator.config.beta
            reward = alpha * rfunc + beta * rsec

            return RewardResult(reward=reward, rfunc=rfunc, rsec=rsec)

        except subprocess.TimeoutExpired:
            return RewardResult(reward=0.0, rfunc=0.0, rsec=1.0)

        except Exception as e:
            # Don't print every error - too noisy
            return RewardResult(reward=0.0, rfunc=0.0, rsec=1.0)

        finally:
            # Cleanup
            try:
                os.unlink(code_file)
            except OSError:
                pass

    def _quick_python_security_check(self, code: str) -> List[Dict]:
        """Quick security checks for Python code"""
        import re

        findings = []

        # Dangerous Python patterns
        patterns = [
            (r'\beval\s*\(', 'python/code-injection', 'Use of eval()'),
            (r'\bexec\s*\(', 'python/code-injection', 'Use of exec()'),
            (r'\b__import__\s*\(', 'python/dynamic-import', 'Dynamic import'),
            (r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True',
             'python/shell-injection', 'Shell injection risk'),
            (r'\bos\.system\s*\(', 'python/command-injection', 'Command injection'),
            (r'\bos\.popen\s*\(', 'python/command-injection', 'Command injection'),
            (r'\bpickle\.loads?\s*\(', 'python/unsafe-deserialization', 'Unsafe pickle'),
            (r'\byaml\.load\s*\([^)]*\)', 'python/unsafe-yaml', 'Unsafe YAML load'),
            (r'input\s*\(\s*\)', 'python/user-input', 'Unvalidated user input'),
        ]

        for pattern, rule_id, message in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                findings.append({'rule_id': rule_id, 'message': message})

        return findings

    def _run_tests_for_rfunc(self, code_file: str, test_cases: Dict) -> float:
        """
        Run test cases with PARTIAL CREDIT scoring.

        This addresses the sparse reward problem by giving graduated rewards
        instead of binary pass/fail.

        Scoring:
        - 0.0: Syntax error (handled before this method is called)
        - 0.2: Valid syntax (base credit for reaching this method)
        - 0.4: Code runs without runtime error
        - 0.6: Code produces some output
        - 0.6 + 0.4*(passed/total): Partial test matches
        - 1.0: All tests pass

        Args:
            code_file: Path to Python code file
            test_cases: Dict with 'inputs' and 'outputs' lists

        Returns:
            Rfunc in [0.2, 1.0] with partial credit
        """
        import subprocess

        inputs = test_cases.get('inputs', [])
        outputs = test_cases.get('outputs', [])

        if not inputs or not outputs:
            return 0.2  # Syntax valid, no tests = base credit only

        num_tests = min(len(inputs), len(outputs))
        passed = 0
        ran_without_error = 0
        produced_output = 0

        for i in range(num_tests):
            test_input = inputs[i]
            expected_output = outputs[i]

            try:
                # Run the code with test input
                result = subprocess.run(
                    ['python', code_file],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=5,  # 5 second timeout per test
                )

                # Check if ran without error (returncode 0)
                if result.returncode == 0:
                    ran_without_error += 1

                # Get actual output
                actual_output = result.stdout.strip()
                expected_clean = expected_output.strip()

                # Check if produced any output
                if actual_output:
                    produced_output += 1

                # Check if output matches expected
                if actual_output == expected_clean:
                    passed += 1

            except subprocess.TimeoutExpired:
                # Timeout = no credit for this test
                pass
            except Exception:
                # Any error = no credit for this test
                pass

        # Compute partial credit score
        if num_tests == 0:
            return 0.2  # Base syntax credit

        # Base: 0.2 for valid syntax (we got here, so syntax is valid)
        score = 0.2

        # +0.2 if any test ran without runtime error
        if ran_without_error > 0:
            score += 0.2

        # +0.2 if any test produced output
        if produced_output > 0:
            score += 0.2

        # +0.4 scaled by test pass rate
        score += 0.4 * (passed / num_tests)

        return min(score, 1.0)

    def _run_tests(self, code_file: str, test_cases: Dict) -> Dict:
        """Run test cases against Python code (legacy interface)"""
        rfunc = self._run_tests_for_rfunc(code_file, test_cases)
        num_tests = min(
            len(test_cases.get('inputs', [])),
            len(test_cases.get('outputs', []))
        )
        passed = int(rfunc * num_tests)
        return {
            'passed': passed,
            'failed': num_tests - passed,
            'total_tests': num_tests
        }


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
        use_bandit: bool = False,
        use_llm_for_security: bool = False,
    ):
        """
        Initialize PPO trainer.

        Args:
            config: Full training configuration
            prompts: List of prompts for RL training
            use_bandit: Use Bandit for comprehensive security analysis
            use_llm_for_security: Use LLM for ambiguous security findings
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config
        self.prompts = prompts or []

        self.policy = PolicyModel(config.model, device=config.device)
        self.feedback = ToolFeedbackCollector(
            reward_config=config.reward,
            use_llm_for_security=use_llm_for_security,
            use_bandit=use_bandit,
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
        prompts: List,
        num_samples: int,
        start_idx: int = 0,
    ) -> RolloutBuffer:
        """Collect rollout samples from policy with error handling.

        Args:
            prompts: List of prompts (can be strings or dicts with 'prompt' and 'test_cases')
            num_samples: Number of samples to collect
            start_idx: Starting index in prompts list

        Returns:
            RolloutBuffer with collected samples
        """

        buffer = RolloutBuffer()
        failed_generations = 0

        for i in range(num_samples):
            prompt_idx = (start_idx + i) % len(prompts)
            prompt_item = prompts[prompt_idx]

            # Handle both string prompts and dict format with test cases
            if isinstance(prompt_item, dict):
                prompt = prompt_item.get('prompt', '')
                test_cases = prompt_item.get('test_cases', {'inputs': [], 'outputs': []})
            else:
                prompt = prompt_item
                test_cases = {'inputs': [], 'outputs': []}

            # Format prompt
            formatted_prompt = (
                f"### Instruction:\nComplete the following code:\n\n"
                f"{prompt}\n\n### Response:\n"
            )

            try:
                # Generate completion with retries
                completion, log_probs = self.policy.generate(
                    prompt=formatted_prompt,
                    max_new_tokens=self.config.model.max_new_tokens,
                    temperature=self.config.model.temperature,
                    top_p=self.config.model.top_p,
                )

                # Skip if generation failed
                if not completion or len(log_probs) == 0:
                    failed_generations += 1
                    warnings.warn(f"Empty generation for sample {i}, skipping")
                    continue

                # Get reward from tools (with optional test cases for true Rfunc)
                reward_result = self.feedback.get_reward(completion, test_cases)

                sample = RolloutSample(
                    prompt=formatted_prompt,
                    completion=completion,
                    log_probs=log_probs,
                    reward=reward_result.reward,
                    rfunc=reward_result.rfunc,
                    rsec=reward_result.rsec,
                )

                buffer.add(sample)

                # Clear memory after each sample
                torch.cuda.empty_cache()

            except Exception as e:
                failed_generations += 1
                warnings.warn(f"Error generating sample {i}: {e}")
                torch.cuda.empty_cache()
                continue

        if failed_generations > 0:
            warnings.warn(f"Failed to generate {failed_generations}/{num_samples} samples")

        # Ensure we have at least one sample
        if len(buffer) == 0:
            warnings.warn("No valid samples collected, adding dummy sample")
            buffer.add(RolloutSample(
                prompt="",
                completion="",
                log_probs=[-10.0],
                reward=0.0,
                rfunc=0.0,
                rsec=1.0,
            ))

        return buffer

    def _ppo_update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Perform PPO policy update with numerical stability"""

        # Clear cache before update
        torch.cuda.empty_cache()

        self.policy.model.train()

        total_policy_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        skipped_samples = 0

        for ppo_epoch in range(self.config.ppo.ppo_epochs):
            for batch in buffer.get_batch(self.config.ppo.mini_batch_size):
                # Compute current and old log probs
                batch_loss = torch.tensor(0.0, device=self.policy.device, requires_grad=True)
                batch_kl = 0.0
                valid_samples = 0

                for sample in batch:
                    try:
                        # Skip empty completions
                        if not sample.completion or len(sample.log_probs) == 0:
                            skipped_samples += 1
                            continue

                        # Get log probs
                        policy_log_probs, ref_log_probs = self.policy.compute_log_probs(
                            sample.prompt, sample.completion
                        )

                        # Align all tensors to the same length
                        policy_len = policy_log_probs.shape[1] if policy_log_probs.dim() > 1 else policy_log_probs.shape[0]
                        ref_len = ref_log_probs.shape[1] if ref_log_probs.dim() > 1 else ref_log_probs.shape[0]
                        old_len = len(sample.log_probs)

                        min_len = min(policy_len, ref_len, old_len)

                        if min_len == 0:
                            skipped_samples += 1
                            continue

                        # Old log probs (from generation)
                        old_log_probs = torch.tensor(
                            sample.log_probs[:min_len],
                            device=policy_log_probs.device,
                            dtype=torch.float32,
                        )

                        # Align policy and ref log probs
                        if policy_log_probs.dim() > 1:
                            policy_log_probs_aligned = policy_log_probs[:, :min_len].squeeze(0)
                        else:
                            policy_log_probs_aligned = policy_log_probs[:min_len]

                        if ref_log_probs.dim() > 1:
                            ref_log_probs_aligned = ref_log_probs[:, :min_len].squeeze(0)
                        else:
                            ref_log_probs_aligned = ref_log_probs[:min_len]

                        # Compute ratio with numerical stability
                        log_ratio = policy_log_probs_aligned - old_log_probs
                        log_ratio = torch.clamp(log_ratio, min=-20, max=20)  # Prevent exp overflow
                        ratio = torch.exp(log_ratio).mean()

                        # Clamp ratio to prevent extreme values
                        ratio = torch.clamp(ratio, min=0.01, max=100.0)

                        # Clipped objective
                        advantage = sample.advantage
                        clip_range = self.config.ppo.clip_range

                        unclipped = ratio * advantage
                        clipped = torch.clamp(
                            ratio, 1 - clip_range, 1 + clip_range
                        ) * advantage

                        policy_loss = -torch.min(unclipped, clipped)

                        # KL penalty (using aligned tensors)
                        kl = (policy_log_probs_aligned - ref_log_probs_aligned).mean()
                        kl = torch.clamp(kl, min=-10, max=10)  # Prevent extreme KL
                        kl_penalty = self.config.ppo.kl_penalty_coefficient * kl

                        # Accumulate loss
                        sample_loss = policy_loss + kl_penalty

                        # Check for valid loss
                        if torch.isnan(sample_loss) or torch.isinf(sample_loss):
                            warnings.warn(f"Invalid loss detected, skipping sample")
                            skipped_samples += 1
                            continue

                        batch_loss = batch_loss + sample_loss
                        batch_kl += kl.item()
                        valid_samples += 1

                    except Exception as e:
                        warnings.warn(f"Error processing sample: {e}")
                        skipped_samples += 1
                        # Clear CUDA cache on OOM
                        if "CUDA out of memory" in str(e):
                            torch.cuda.empty_cache()
                        continue

                # Skip batch if no valid samples
                if valid_samples == 0:
                    continue

                # Average over valid samples
                batch_loss = batch_loss / valid_samples

                # Backward pass
                self.optimizer.zero_grad()

                try:
                    batch_loss.backward()

                    # Check for NaN gradients
                    has_nan_grad = False
                    for param in self.policy.model.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break

                    if has_nan_grad:
                        warnings.warn("NaN gradients detected, skipping update")
                        self.optimizer.zero_grad()
                        continue

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.model.parameters(),
                        self.config.ppo.max_grad_norm,
                    )

                    self.optimizer.step()

                    total_policy_loss += batch_loss.item()
                    total_kl += batch_kl / valid_samples
                    num_updates += 1

                    # Clear cache after each batch update
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    warnings.warn(f"Backward pass failed: {e}")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue

                # Early stopping on KL
                if self.config.ppo.target_kl and batch_kl / valid_samples > self.config.ppo.target_kl:
                    break

        if skipped_samples > 0:
            warnings.warn(f"Skipped {skipped_samples} samples due to errors")

        # Compute metrics
        rewards = [s.reward for s in buffer.samples]
        rfuncs = [s.rfunc for s in buffer.samples]
        rsecs = [s.rsec for s in buffer.samples]

        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'kl': total_kl / max(num_updates, 1),
            'mean_reward': sum(rewards) / len(rewards) if rewards else 0.0,
            'mean_rfunc': sum(rfuncs) / len(rfuncs) if rfuncs else 0.0,
            'mean_rsec': sum(rsecs) / len(rsecs) if rsecs else 1.0,
            'min_reward': min(rewards) if rewards else 0.0,
            'max_reward': max(rewards) if rewards else 0.0,
            'skipped_samples': skipped_samples,
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
