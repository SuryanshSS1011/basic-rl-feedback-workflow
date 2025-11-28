#!/usr/bin/env python3
"""
Multi-model code generator for benchmarking.
Generates code using multiple LLMs (StarCoder2, DeepSeek, CodeLlama, WizardCoder).
"""

import torch
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time


class MultiModelRunner:
    """Generate code with multiple LLM models."""

    SUPPORTED_MODELS = {
        'starcoder2': 'bigcode/starcoder2-7b',
        'deepseek': 'deepseek-ai/deepseek-coder-6.7b-base',
        'codellama': 'codellama/CodeLlama-7b-hf',
        'wizardcoder': 'WizardLM/WizardCoder-15B-V1.0',
        'deepseek-small': 'deepseek-ai/deepseek-coder-1.3b-instruct',  # For testing
    }

    def __init__(
        self,
        cache_dir: str,
        results_dir: str = "./benchmark/results",
        device: Optional[str] = None,
        checkpoint_interval: int = 10
    ):
        self.cache_dir = cache_dir
        self.results_dir = results_dir
        self.checkpoint_interval = checkpoint_interval

        # Checkpoint directory
        self.checkpoint_dir = Path(results_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect best device: CUDA > MPS > CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        print(f"Using device: {self.device}")

    def _save_checkpoint(self, model_name: str, prompt_idx: int, results: List[Dict]):
        """
        Save checkpoint for resumable execution.

        Args:
            model_name: Name of the model being processed
            prompt_idx: Index of last processed prompt
            results: Results collected so far
        """
        checkpoint = {
            'model': model_name,
            'last_prompt_idx': prompt_idx,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        checkpoint_path = self.checkpoint_dir / f"{model_name}_checkpoint.json"

        # Atomic write - write to temp file then rename
        temp_path = checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        temp_path.rename(checkpoint_path)

    def _load_checkpoint(self, model_name: str) -> Optional[Dict]:
        """
        Load checkpoint if it exists.

        Args:
            model_name: Name of the model

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{model_name}_checkpoint.json"

        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                print(f"Found checkpoint for {model_name} at prompt {checkpoint['last_prompt_idx']}")
                return checkpoint
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                return None
        return None

    def _clear_checkpoint(self, model_name: str):
        """Remove checkpoint file after successful completion."""
        checkpoint_path = self.checkpoint_dir / f"{model_name}_checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Cleared checkpoint for {model_name}")

    def _create_prompt(self, prompt_text: str, language: str = 'c') -> str:
        """
        Format prompt for code generation based on target language.

        Args:
            prompt_text: The task description
            language: Target language ('c', 'python')

        Returns:
            Formatted prompt string
        """
        if language == 'python':
            return f"# Task: {prompt_text}\n# Write a Python program to solve this:\n\n"
        else:
            # Default to C
            return f"// Task: {prompt_text}\n// Write a C program to solve this:\n#include <stdio.h>\n\nint main() {{\n"

    def _create_c_prompt(self, prompt_text: str) -> str:
        """Format prompt for C code generation (backward compatible)."""
        return self._create_prompt(prompt_text, 'c')

    def generate_with_model(
        self,
        model_name: str,
        prompts: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        resume: bool = True
    ) -> List[Dict]:
        """
        Generate code for all prompts using specified model with checkpointing.

        Args:
            model_name: Key from SUPPORTED_MODELS or full HuggingFace model path
            prompts: List of prompt dictionaries
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            resume: Whether to resume from checkpoint if available

        Returns:
            List of results with generated code and metadata
        """
        # Check for existing checkpoint
        start_idx = 0
        results = []

        if resume:
            checkpoint = self._load_checkpoint(model_name)
            if checkpoint:
                start_idx = checkpoint['last_prompt_idx'] + 1
                results = checkpoint['results']
                print(f"Resuming from prompt {start_idx} ({len(results)} results already collected)")

        # Skip if already completed
        if start_idx >= len(prompts):
            print(f"Model {model_name} already completed all {len(prompts)} prompts")
            return results

        # Get model path
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        print(f"\n{'='*60}")
        print(f"Loading model: {model_path}")
        print(f"Processing prompts {start_idx} to {len(prompts) - 1}")
        print(f"{'='*60}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )

            # Add padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    cache_dir=self.cache_dir,
                    device_map="auto"
                )
            elif self.device == "mps":
                # MPS (Apple Silicon) - use float16 for better performance
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    cache_dir=self.cache_dir
                )
                model = model.to(self.device)
            else:
                # CPU fallback
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=self.cache_dir
                )
                model = model.to(self.device)

            print(f"Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return results

        # Generate code for remaining prompts
        remaining_prompts = prompts[start_idx:]

        for i, prompt_dict in enumerate(tqdm(remaining_prompts, desc=f"Generating with {model_name}")):
            actual_idx = start_idx + i

            try:
                # Get target language from prompt
                language = prompt_dict.get('language', 'c')

                # Format prompt based on language
                prompt_text = self._create_prompt(prompt_dict['prompt'], language)

                # Tokenize
                inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)

                # Generate
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )
                generation_time = time.time() - start_time

                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                code = generated_text[len(prompt_text):].strip()
                full_code = f"{prompt_text}{code}"

                # Save result
                result = {
                    'prompt_id': prompt_dict['id'],
                    'dataset': prompt_dict['dataset'],
                    'original_prompt': prompt_dict['prompt'],
                    'language': language,
                    'model': model_name,
                    'model_path': model_path,
                    'generated_code': full_code,
                    'generation_time': generation_time,
                    'tokens_generated': len(tokenizer.encode(code)),
                    'parameters': {
                        'max_new_tokens': max_new_tokens,
                        'temperature': temperature,
                        'top_k': top_k,
                        'top_p': top_p
                    }
                }

                results.append(result)

                # Save individual result
                self._save_result(result, language)

            except Exception as e:
                print(f"\nError generating code for prompt {prompt_dict['id']}: {e}")
                results.append({
                    'prompt_id': prompt_dict['id'],
                    'dataset': prompt_dict['dataset'],
                    'model': model_name,
                    'error': str(e),
                    'generated_code': None
                })

            # Save checkpoint at intervals
            if (actual_idx + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(model_name, actual_idx, results)
                print(f"\n[Checkpoint saved at prompt {actual_idx + 1}]")

        # Clean up model from memory
        del model
        del tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

        # Clear checkpoint on successful completion
        self._clear_checkpoint(model_name)

        print(f"\nCompleted generation with {model_name}: {len(results)} results")
        return results

    def _save_result(self, result: Dict, language: str = 'c'):
        """
        Save individual generation result to file.

        Args:
            result: Result dictionary
            language: Programming language for file extension
        """
        # Create directory structure: results/{model}/{dataset}/{prompt_id}/
        output_dir = Path(self.results_dir) / result['model'] / result['dataset'] / result['prompt_id']
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension based on language
        ext = '.py' if language == 'python' else '.c'

        # Save generated code
        code_path = output_dir / f"generated_code{ext}"
        if result.get('generated_code'):
            with open(code_path, 'w') as f:
                f.write(result['generated_code'])

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        metadata = {k: v for k, v in result.items() if k != 'generated_code'}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def run_benchmark(
        self,
        models: List[str],
        prompts: List[Dict],
        **generation_kwargs
    ) -> Dict[str, List[Dict]]:
        """
        Run code generation benchmark across multiple models.

        Args:
            models: List of model names to benchmark
            prompts: List of prompts to use
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary mapping model names to their results
        """
        all_results = {}

        for model_name in models:
            print(f"\n{'#'*60}")
            print(f"# Benchmarking Model: {model_name}")
            print(f"{'#'*60}")

            results = self.generate_with_model(
                model_name=model_name,
                prompts=prompts,
                **generation_kwargs
            )

            all_results[model_name] = results

            # Save checkpoint
            checkpoint_path = Path(self.results_dir) / "summary" / f"{model_name}_results.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Checkpoint saved: {checkpoint_path}")

        return all_results


if __name__ == "__main__":
    # Test the runner with a small sample
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Create test prompt
    test_prompts = [
        {
            'id': 'test_001',
            'prompt': 'Write a C program that calculates the factorial of a number',
            'language': 'C',
            'dataset': 'test'
        }
    ]

    # Use smaller model for testing
    cache_dir = "/scratch/{}/hf_cache".format(os.getlogin())
    runner = MultiModelRunner(cache_dir=cache_dir)

    # Test with small model
    results = runner.generate_with_model(
        model_name='deepseek-small',
        prompts=test_prompts,
        max_new_tokens=256
    )

    print("\nTest completed!")
    print(f"Generated {len(results)} results")
