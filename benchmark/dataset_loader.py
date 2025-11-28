#!/usr/bin/env python3
"""
Dataset loader for multi-LLM benchmarking.
Loads and standardizes prompts from multiple sources:
1. QuestionPromptForLLMs (Google Docs export)
2. codeparrot/xlcost-text-to-code (HuggingFace)
3. APPS_Plus (GitHub JSON)
"""

import json
import os
from typing import List, Dict, Optional
from datasets import load_dataset
import requests


class DatasetLoader:
    """Load and standardize code generation prompts from multiple sources."""

    def __init__(self, cache_dir: str = "./benchmark/dataset_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load_apps_plus(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Load APPS_Plus dataset from GitHub.
        URL: https://raw.githubusercontent.com/Ablustrund/APPS_Plus/refs/heads/main/data/v1/data.json
        """
        url = "https://raw.githubusercontent.com/Ablustrund/APPS_Plus/refs/heads/main/data/v1/data.json"
        cache_path = os.path.join(self.cache_dir, "apps_plus.json")

        # Try loading from cache first
        if os.path.exists(cache_path):
            print(f"Loading APPS_Plus from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                data = json.load(f)
        else:
            print(f"Downloading APPS_Plus from: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Cache the data
            with open(cache_path, 'w') as f:
                json.dump(data, f)

        # Standardize format
        prompts = []
        items = data if isinstance(data, list) else [data]

        for i, item in enumerate(items[:limit] if limit else items):
            # APPS_Plus typically has 'question' and 'solutions' fields
            prompt_text = item.get('question', item.get('prompt', item.get('description', '')))

            # Extract test cases (inputs/outputs for semantic error detection)
            test_inputs = item.get('inputs', item.get('input', []))
            test_outputs = item.get('outputs', item.get('output', []))

            # Ensure inputs/outputs are lists
            if isinstance(test_inputs, str):
                test_inputs = [test_inputs]
            if isinstance(test_outputs, str):
                test_outputs = [test_outputs]

            prompts.append({
                'id': f"apps_plus_{i}",
                'prompt': prompt_text,
                'language': 'python',  # APPS_Plus contains Python problems
                'dataset': 'apps_plus',
                'expected_output': item.get('solutions', None),
                'test_cases': {
                    'inputs': test_inputs,
                    'outputs': test_outputs
                },
                'difficulty': item.get('difficulty', 'unknown'),
                'problem_id': item.get('id', item.get('problem_id', i))
            })

        print(f"Loaded {len(prompts)} prompts from APPS_Plus")
        return prompts

    def load_xlcost(self, limit: Optional[int] = None, split: str = "train") -> List[Dict]:
        """
        Load xlcost-text-to-code dataset from HuggingFace.
        Focus on C/C++ examples.
        """
        print(f"Loading xlcost dataset (split: {split})...")

        try:
            # Load C++ program level dataset with trust_remote_code
            dataset = load_dataset(
                "codeparrot/xlcost-text-to-code",
                "C++-program-level",
                split=split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

            prompts = []
            items = list(dataset)[:limit] if limit else list(dataset)

            for i, item in enumerate(items):
                # xlcost has 'text' (description) and 'code' fields
                prompts.append({
                    'id': f"xlcost_cpp_{i}",
                    'prompt': item.get('text', ''),
                    'language': 'c',  # Lowercase for consistency
                    'dataset': 'xlcost',
                    'expected_output': item.get('code', None),
                    'test_cases': {
                        'inputs': [],  # xlcost doesn't have test cases
                        'outputs': []
                    },
                    'difficulty': 'unknown'
                })

            print(f"Loaded {len(prompts)} prompts from xlcost C++ dataset")
            return prompts

        except Exception as e:
            print(f"Warning: Could not load xlcost dataset: {e}")
            print("Continuing with other datasets...")
            return []

    def load_question_prompts(self, file_path: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Load QuestionPromptForLLMs dataset.
        Expected format: JSON/CSV file with prompts.
        User needs to export from Google Docs.
        """
        if file_path is None:
            file_path = os.path.join(self.cache_dir, "question_prompts.json")

        if not os.path.exists(file_path):
            print(f"QuestionPromptForLLMs not found at {file_path}")
            print("Please download from: https://docs.google.com/document/d/1Lo6QMD1trXL8OlNPf50zY4rwvUDYu1xzK4sgEj8GrlA/edit")
            print("Export as .json and save to:", file_path)
            return []

        print(f"Loading QuestionPromptForLLMs from: {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            prompts = []
            items = data if isinstance(data, list) else [data]

            for i, item in enumerate(items[:limit] if limit else items):
                # Adapt to actual structure of the document
                prompt_text = item.get('prompt', item.get('question', item.get('text', '')))

                # Get language, normalize to lowercase
                lang = item.get('language', 'c').lower()
                if lang in ('c++', 'cpp'):
                    lang = 'c'

                prompts.append({
                    'id': f"question_prompt_{i}",
                    'prompt': prompt_text,
                    'language': lang,
                    'dataset': 'question_prompts',
                    'expected_output': item.get('expected_code', None),
                    'test_cases': {
                        'inputs': item.get('test_inputs', []),
                        'outputs': item.get('test_outputs', [])
                    },
                    'difficulty': item.get('difficulty', 'unknown')
                })

            print(f"Loaded {len(prompts)} prompts from QuestionPromptForLLMs")
            return prompts

        except Exception as e:
            print(f"Error loading QuestionPromptForLLMs: {e}")
            return []

    def load_all(self, limit_per_dataset: Optional[int] = None) -> List[Dict]:
        """Load prompts from all available datasets."""
        all_prompts = []

        # Load each dataset
        all_prompts.extend(self.load_xlcost(limit=limit_per_dataset))
        all_prompts.extend(self.load_apps_plus(limit=limit_per_dataset))
        all_prompts.extend(self.load_question_prompts(limit=limit_per_dataset))

        print(f"\nTotal prompts loaded: {len(all_prompts)}")
        return all_prompts

    def save_prompts(self, prompts: List[Dict], output_path: str):
        """Save standardized prompts to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        print(f"Saved {len(prompts)} prompts to: {output_path}")


if __name__ == "__main__":
    # Test the dataset loader
    loader = DatasetLoader()

    # Load small sample from each dataset
    print("Testing dataset loader with small samples...\n")
    prompts = loader.load_all(limit_per_dataset=5)

    # Save to file
    output_path = "./benchmark/dataset_cache/all_prompts.json"
    loader.save_prompts(prompts, output_path)

    # Display sample
    if prompts:
        print("\nSample prompt:")
        print(json.dumps(prompts[0], indent=2))
