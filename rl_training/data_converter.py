"""
Data Converter for RL Training

Converts benchmark results into training-ready format for SFT and PPO.

Supports:
- Loading benchmark results from JSON files
- Converting to prompt-completion pairs with rewards
- Filtering by reward threshold
- Selecting best completions per prompt
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
import random

from .reward_calculator import RewardCalculator, RewardResult
from .config import RewardConfig


@dataclass
class TrainingSample:
    """A single training sample with prompt, completion, and reward"""

    prompt_id: str
    prompt: str
    completion: str
    reward: float
    rfunc: float
    rsec: float

    # Metadata
    model: str
    dataset: str
    compiles: bool
    tests_passed: int
    tests_total: int
    has_security_issues: bool

    def to_dict(self) -> Dict:
        return {
            'prompt_id': self.prompt_id,
            'prompt': self.prompt,
            'completion': self.completion,
            'reward': self.reward,
            'rfunc': self.rfunc,
            'rsec': self.rsec,
            'model': self.model,
            'dataset': self.dataset,
            'compiles': self.compiles,
            'tests_passed': self.tests_passed,
            'tests_total': self.tests_total,
            'has_security_issues': self.has_security_issues,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingSample':
        return cls(
            prompt_id=data['prompt_id'],
            prompt=data['prompt'],
            completion=data['completion'],
            reward=data['reward'],
            rfunc=data['rfunc'],
            rsec=data['rsec'],
            model=data.get('model', 'unknown'),
            dataset=data.get('dataset', 'unknown'),
            compiles=data.get('compiles', True),
            tests_passed=data.get('tests_passed', 0),
            tests_total=data.get('tests_total', 0),
            has_security_issues=data.get('has_security_issues', False),
        )


@dataclass
class TrainingDataset:
    """Collection of training samples with filtering and selection"""

    samples: List[TrainingSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[TrainingSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> TrainingSample:
        return self.samples[idx]

    def add(self, sample: TrainingSample):
        """Add a sample to the dataset"""
        self.samples.append(sample)

    def filter_by_reward(self, min_reward: float) -> 'TrainingDataset':
        """Filter samples by minimum reward threshold"""
        filtered = [s for s in self.samples if s.reward >= min_reward]
        return TrainingDataset(samples=filtered)

    def filter_compiling(self) -> 'TrainingDataset':
        """Keep only samples that compile"""
        filtered = [s for s in self.samples if s.compiles]
        return TrainingDataset(samples=filtered)

    def filter_passing_tests(self, min_ratio: float = 0.5) -> 'TrainingDataset':
        """Keep samples where at least min_ratio tests pass"""
        filtered = []
        for s in self.samples:
            if s.tests_total == 0:
                filtered.append(s)  # No tests = assume OK
            elif s.tests_passed / s.tests_total >= min_ratio:
                filtered.append(s)
        return TrainingDataset(samples=filtered)

    def select_best_per_prompt(self) -> 'TrainingDataset':
        """Select the best completion for each unique prompt"""
        best_by_prompt: Dict[str, TrainingSample] = {}

        for sample in self.samples:
            key = sample.prompt_id
            if key not in best_by_prompt or sample.reward > best_by_prompt[key].reward:
                best_by_prompt[key] = sample

        return TrainingDataset(samples=list(best_by_prompt.values()))

    def shuffle(self, seed: Optional[int] = None) -> 'TrainingDataset':
        """Shuffle samples"""
        samples = list(self.samples)
        if seed is not None:
            random.seed(seed)
        random.shuffle(samples)
        return TrainingDataset(samples=samples)

    def split(self, train_ratio: float = 0.9) -> Tuple['TrainingDataset', 'TrainingDataset']:
        """Split into train/validation sets"""
        n_train = int(len(self.samples) * train_ratio)
        return (
            TrainingDataset(samples=self.samples[:n_train]),
            TrainingDataset(samples=self.samples[n_train:]),
        )

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        if not self.samples:
            return {'count': 0}

        rewards = [s.reward for s in self.samples]
        rfuncs = [s.rfunc for s in self.samples]
        rsecs = [s.rsec for s in self.samples]

        compiling = sum(1 for s in self.samples if s.compiles)
        with_security = sum(1 for s in self.samples if s.has_security_issues)

        return {
            'count': len(self.samples),
            'reward_mean': sum(rewards) / len(rewards),
            'reward_min': min(rewards),
            'reward_max': max(rewards),
            'rfunc_mean': sum(rfuncs) / len(rfuncs),
            'rsec_mean': sum(rsecs) / len(rsecs),
            'compiling_ratio': compiling / len(self.samples),
            'security_issues_ratio': with_security / len(self.samples),
            'unique_prompts': len(set(s.prompt_id for s in self.samples)),
        }

    def to_jsonl(self, path: Path):
        """Save to JSONL format"""
        with open(path, 'w') as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + '\n')

    @classmethod
    def from_jsonl(cls, path: Path) -> 'TrainingDataset':
        """Load from JSONL format"""
        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(TrainingSample.from_dict(json.loads(line)))
        return cls(samples=samples)


class BenchmarkConverter:
    """
    Converts benchmark results to training format.

    Expected benchmark structure:
        benchmark/full_results/{model}/{dataset}/{dataset}_{id}/
            - prompt.txt
            - generated_code.py
            - analysis.json

    Usage:
        converter = BenchmarkConverter(results_dir="./benchmark/full_results")
        dataset = converter.convert_all()
        dataset = dataset.filter_by_reward(0.5).select_best_per_prompt()
    """

    def __init__(
        self,
        results_dir: Path,
        config: Optional[RewardConfig] = None
    ):
        """
        Initialize converter.

        Args:
            results_dir: Path to benchmark results directory
            config: Reward configuration
        """
        self.results_dir = Path(results_dir)
        self.reward_calculator = RewardCalculator(config=config)

    def _load_sample(
        self,
        sample_dir: Path,
        model: str,
        dataset: str
    ) -> Optional[TrainingSample]:
        """Load a single sample from benchmark output directory"""

        prompt_file = sample_dir / "prompt.txt"
        code_file = sample_dir / "generated_code.py"
        analysis_file = sample_dir / "analysis.json"

        # Check required files exist
        if not prompt_file.exists() or not code_file.exists():
            return None

        # Load prompt
        with open(prompt_file) as f:
            prompt = f.read().strip()

        # Load generated code
        with open(code_file) as f:
            completion = f.read()

        # Load analysis if available
        analysis = {}
        if analysis_file.exists():
            try:
                with open(analysis_file) as f:
                    analysis = json.load(f)
            except json.JSONDecodeError:
                pass

        # Compute reward
        reward_result = self.reward_calculator.compute_from_analysis_result(analysis)

        # Extract metadata
        compilation = analysis.get('compilation', {})
        test_exec = analysis.get('test_execution', {})
        security = analysis.get('security', analysis.get('codeql', {}))

        compiles = compilation.get('compiles', True)
        tests_passed = test_exec.get('passed', 0)
        tests_total = test_exec.get('total_tests', 0)
        has_security_issues = False

        if isinstance(security, dict):
            has_security_issues = security.get('has_security_issues', False)
            if not has_security_issues:
                issues = security.get('security_issues', [])
                has_security_issues = len(issues) > 0 if isinstance(issues, list) else False

        # Extract prompt_id from directory name
        prompt_id = sample_dir.name

        return TrainingSample(
            prompt_id=prompt_id,
            prompt=prompt,
            completion=completion,
            reward=reward_result.reward,
            rfunc=reward_result.rfunc,
            rsec=reward_result.rsec,
            model=model,
            dataset=dataset,
            compiles=compiles,
            tests_passed=tests_passed,
            tests_total=tests_total,
            has_security_issues=has_security_issues,
        )

    def convert_model_dataset(
        self,
        model: str,
        dataset: str,
        limit: Optional[int] = None
    ) -> TrainingDataset:
        """
        Convert results for a specific model and dataset.

        Args:
            model: Model name (e.g., 'starcoder2', 'deepseek')
            dataset: Dataset name (e.g., 'apps_plus')
            limit: Maximum number of samples to load

        Returns:
            TrainingDataset with converted samples
        """
        model_dir = self.results_dir / model / dataset

        if not model_dir.exists():
            return TrainingDataset()

        samples = []
        count = 0

        # Find all sample directories
        for sample_dir in sorted(model_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            sample = self._load_sample(sample_dir, model, dataset)
            if sample is not None:
                samples.append(sample)
                count += 1

                if limit and count >= limit:
                    break

        return TrainingDataset(samples=samples)

    def convert_all(
        self,
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        limit_per_model: Optional[int] = None
    ) -> TrainingDataset:
        """
        Convert all benchmark results.

        Args:
            models: List of models to include (None = all)
            datasets: List of datasets to include (None = all)
            limit_per_model: Max samples per model

        Returns:
            Combined TrainingDataset
        """
        all_samples = []

        if not self.results_dir.exists():
            return TrainingDataset()

        # Find all models
        available_models = [
            d.name for d in self.results_dir.iterdir()
            if d.is_dir()
        ]

        if models:
            available_models = [m for m in available_models if m in models]

        for model in available_models:
            model_dir = self.results_dir / model

            # Find all datasets for this model
            available_datasets = [
                d.name for d in model_dir.iterdir()
                if d.is_dir()
            ]

            if datasets:
                available_datasets = [d for d in available_datasets if d in datasets]

            model_samples = []
            for dataset in available_datasets:
                ds = self.convert_model_dataset(model, dataset)
                model_samples.extend(ds.samples)

            # Apply limit per model
            if limit_per_model and len(model_samples) > limit_per_model:
                # Sort by reward and take best
                model_samples.sort(key=lambda s: s.reward, reverse=True)
                model_samples = model_samples[:limit_per_model]

            all_samples.extend(model_samples)

        return TrainingDataset(samples=all_samples)

    def get_available_models(self) -> List[str]:
        """List available models in results directory"""
        if not self.results_dir.exists():
            return []
        return [d.name for d in self.results_dir.iterdir() if d.is_dir()]

    def get_available_datasets(self, model: str) -> List[str]:
        """List available datasets for a model"""
        model_dir = self.results_dir / model
        if not model_dir.exists():
            return []
        return [d.name for d in model_dir.iterdir() if d.is_dir()]


def prepare_sft_data(
    results_dir: Path,
    output_path: Path,
    min_reward: float = 0.5,
    select_best: bool = True,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[Path, Path]:
    """
    Prepare data for SFT training.

    Args:
        results_dir: Benchmark results directory
        output_path: Output directory for processed data
        min_reward: Minimum reward threshold
        select_best: Select best completion per prompt
        train_ratio: Train/validation split ratio
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_path, val_path)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert benchmark results
    converter = BenchmarkConverter(results_dir)
    dataset = converter.convert_all()

    print(f"Loaded {len(dataset)} samples from benchmark results")

    # Filter by reward
    dataset = dataset.filter_by_reward(min_reward)
    print(f"After reward filter (>= {min_reward}): {len(dataset)} samples")

    # Select best per prompt
    if select_best:
        dataset = dataset.select_best_per_prompt()
        print(f"After selecting best per prompt: {len(dataset)} samples")

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    train_ds, val_ds = dataset.split(train_ratio)

    # Save
    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    train_ds.to_jsonl(train_path)
    val_ds.to_jsonl(val_path)

    print(f"Saved {len(train_ds)} train samples to {train_path}")
    print(f"Saved {len(val_ds)} validation samples to {val_path}")

    # Print stats
    print("\nDataset statistics:")
    stats = dataset.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return train_path, val_path
