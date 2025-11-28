#!/usr/bin/env python3
"""
Batch processor for full dataset benchmarking.
Orchestrates code generation, analysis, and metrics computation
with proper memory management and checkpointing.
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .dataset_loader import DatasetLoader
from .multi_model_runner import MultiModelRunner
from .analyze_multi import CodeAnalyzer
from .compute_metrics import MetricsComputer


class BatchProcessor:
    """Orchestrate full benchmark with memory management."""

    def __init__(
        self,
        config_path: str = "./benchmark/config_benchmark.json",
        results_dir: str = "./benchmark/results",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize batch processor.

        Args:
            config_path: Path to benchmark configuration
            results_dir: Directory for results
            cache_dir: HuggingFace cache directory
        """
        self.config_path = Path(config_path)
        self.results_dir = Path(results_dir)

        # Auto-detect cache directory
        if cache_dir is None:
            username = os.getlogin()
            # Try /scratch first (for HPC), fall back to local
            scratch_cache = f"/scratch/{username}/hf_cache"
            if os.path.exists("/scratch"):
                self.cache_dir = scratch_cache
            else:
                # Use local cache in home directory
                self.cache_dir = os.path.expanduser("~/.cache/huggingface")
        else:
            self.cache_dir = cache_dir

        # Load configuration
        self.config = self._load_config()

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.results_dir / "progress.json"

    def _load_config(self) -> Dict:
        """Load benchmark configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "models": ["deepseek-small"],
                "datasets": {
                    "xlcost": {"enabled": True, "limit": None},
                    "apps_plus": {"enabled": True, "limit": None},
                    "question_prompts": {"enabled": False, "limit": None}
                },
                "generation": {
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95
                },
                "analysis": {
                    "test_timeout": 5,
                    "max_tests_per_sample": 10,
                    "codeql_timeout": 180
                },
                "checkpointing": {
                    "enabled": True,
                    "interval": 10
                }
            }

    def _save_progress(self, stage: str, model: str = None, details: Dict = None):
        """Save progress for resumability."""
        progress = {
            'stage': stage,
            'model': model,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'details': details or {}
        }

        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def _load_progress(self) -> Optional[Dict]:
        """Load progress if exists."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return None

    def _clear_gpu_memory(self):
        """Clear GPU memory between models."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def load_all_prompts(self) -> List[Dict]:
        """Load prompts from all enabled datasets."""
        loader = DatasetLoader(cache_dir=str(self.results_dir / "dataset_cache"))
        all_prompts = []

        datasets_config = self.config.get('datasets', {})

        # Load xlcost
        if datasets_config.get('xlcost', {}).get('enabled', True):
            limit = datasets_config['xlcost'].get('limit')
            prompts = loader.load_xlcost(limit=limit)
            all_prompts.extend(prompts)
            print(f"Loaded {len(prompts)} prompts from xlcost")

        # Load APPS_Plus
        if datasets_config.get('apps_plus', {}).get('enabled', True):
            limit = datasets_config['apps_plus'].get('limit')
            prompts = loader.load_apps_plus(limit=limit)
            all_prompts.extend(prompts)
            print(f"Loaded {len(prompts)} prompts from APPS_Plus")

        # Load question prompts
        if datasets_config.get('question_prompts', {}).get('enabled', False):
            limit = datasets_config['question_prompts'].get('limit')
            prompts = loader.load_question_prompts(limit=limit)
            all_prompts.extend(prompts)
            print(f"Loaded {len(prompts)} prompts from question_prompts")

        print(f"\nTotal prompts loaded: {len(all_prompts)}")
        return all_prompts

    def run_generation(self, prompts: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Run code generation for all models.

        Args:
            prompts: List of prompts with test cases

        Returns:
            Dict mapping model names to results
        """
        # Handle both list and dict format for models
        models_config = self.config.get('models', ['deepseek-small'])
        if isinstance(models_config, dict):
            models = list(models_config.keys())
        else:
            models = models_config

        gen_config = self.config.get('generation', {})
        checkpoint_config = self.config.get('checkpointing', {})

        all_results = {}

        runner = MultiModelRunner(
            cache_dir=self.cache_dir,
            results_dir=str(self.results_dir),
            checkpoint_interval=checkpoint_config.get('interval', 10)
        )

        for model in models:
            print(f"\n{'#' * 60}")
            print(f"# Processing Model: {model}")
            print(f"{'#' * 60}")

            self._save_progress('generation', model)

            # Generate code
            results = runner.generate_with_model(
                model_name=model,
                prompts=prompts,
                max_new_tokens=gen_config.get('max_tokens', gen_config.get('max_new_tokens', 512)),
                temperature=gen_config.get('temperature', 0.7),
                top_k=gen_config.get('top_k', 50),
                top_p=gen_config.get('top_p', 0.95),
                resume=checkpoint_config.get('enabled', True)
            )

            all_results[model] = results

            # Clear GPU memory before next model
            self._clear_gpu_memory()

            print(f"Generated {len(results)} samples for {model}")

        return all_results

    def run_analysis(self, prompts: List[Dict]) -> List[Dict]:
        """
        Run analysis on all generated code.

        Args:
            prompts: List of prompts with test cases

        Returns:
            List of analysis results
        """
        # Handle both list and dict format for models
        models_config = self.config.get('models', ['deepseek-small'])
        if isinstance(models_config, dict):
            models = list(models_config.keys())
        else:
            models = models_config

        analysis_config = self.config.get('analysis', {})

        self._save_progress('analysis')

        analyzer = CodeAnalyzer(
            results_dir=str(self.results_dir),
            test_timeout=analysis_config.get('test_timeout', 5),
            max_tests_per_sample=analysis_config.get('max_tests_per_sample', 10)
        )

        # Build prompts map for test case lookup
        prompts_map = {p['id']: p for p in prompts}

        # Run analysis
        results = analyzer.batch_analyze(models, prompts_map=prompts_map)

        # Save results
        summary_dir = self.results_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        analysis_path = summary_dir / "analysis_results.json"
        with open(analysis_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nAnalysis complete: {len(results)} samples")
        return results

    def run_metrics(self) -> None:
        """Compute and save metrics."""
        self._save_progress('metrics')

        computer = MetricsComputer(results_dir=str(self.results_dir))
        computer.run()

    def run_full_benchmark(self) -> None:
        """
        Run the complete benchmark pipeline.

        Stages:
        1. Load prompts from all datasets
        2. Generate code with all models
        3. Analyze generated code
        4. Compute metrics and generate reports
        """
        start_time = time.time()

        print("=" * 60)
        print("FULL BENCHMARK PIPELINE")
        print("=" * 60)
        print(f"Models: {self.config.get('models', [])}")
        print(f"Results directory: {self.results_dir}")
        print("=" * 60)

        # Stage 1: Load prompts
        print("\n[Stage 1/4] Loading prompts...")
        prompts = self.load_all_prompts()

        if not prompts:
            print("No prompts loaded. Exiting.")
            return

        # Save prompts for reference
        prompts_path = self.results_dir / "prompts.json"
        with open(prompts_path, 'w') as f:
            json.dump(prompts, f, indent=2)

        # Stage 2: Generate code
        print("\n[Stage 2/4] Generating code...")
        self.run_generation(prompts)

        # Stage 3: Analyze code
        print("\n[Stage 3/4] Analyzing generated code...")
        self.run_analysis(prompts)

        # Stage 4: Compute metrics
        print("\n[Stage 4/4] Computing metrics...")
        self.run_metrics()

        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed / 60:.1f} minutes")
        print(f"Results saved to: {self.results_dir}")
        print("=" * 60)

        self._save_progress('complete', details={'elapsed_minutes': elapsed / 60})

    def estimate_runtime(self, num_prompts: int, num_models: int) -> float:
        """
        Estimate runtime based on typical generation speeds.

        Args:
            num_prompts: Number of prompts to process
            num_models: Number of models to run

        Returns:
            Estimated runtime in minutes
        """
        # Typical estimates (adjust based on hardware)
        avg_gen_time_per_prompt = 5  # seconds
        model_load_time = 120  # seconds
        analysis_time_per_sample = 10  # seconds

        gen_time = num_prompts * num_models * avg_gen_time_per_prompt
        load_time = num_models * model_load_time
        analysis_time = num_prompts * num_models * analysis_time_per_sample

        total_seconds = gen_time + load_time + analysis_time
        return total_seconds / 60


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full benchmark pipeline")
    parser.add_argument(
        "--config", "-c",
        default="./benchmark/config_benchmark.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--results", "-r",
        default="./benchmark/results",
        help="Results directory"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "analyze", "metrics"],
        default="all",
        help="Which stage to run"
    )

    args = parser.parse_args()

    processor = BatchProcessor(
        config_path=args.config,
        results_dir=args.results
    )

    if args.stage == "all":
        processor.run_full_benchmark()
    elif args.stage == "generate":
        prompts = processor.load_all_prompts()
        processor.run_generation(prompts)
    elif args.stage == "analyze":
        prompts = processor.load_all_prompts()
        processor.run_analysis(prompts)
    elif args.stage == "metrics":
        processor.run_metrics()
