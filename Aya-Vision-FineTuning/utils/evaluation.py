"""
Evaluation Utilities for Aya Vision Fine-tuning

This module provides comprehensive evaluation tools for assessing the performance
of fine-tuned Aya Vision models on various vision-language tasks.
"""

import os
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import pandas as pd


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: Dict[str, Any]
    samples: List[Dict[str, Any]]


class AyaVisionEvaluator:
    """
    Main evaluator for Aya Vision models.
    Provides various evaluation metrics and benchmarks.
    """

    def __init__(self, model, processor, device: str = "auto"):
        """
        Initialize the evaluator.

        Args:
            model: The Aya Vision model (base or fine-tuned)
            processor: The Aya Vision processor
            device: Device to run evaluation on
        """
        self.model = model
        self.processor = processor
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device if needed
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)

        self.model.eval()

    def generate_caption(self, image: Union[Image.Image, str],
                        prompt: str = "Describe this image in detail.",
                        max_new_tokens: int = 300,
                        temperature: float = 0.3) -> str:
        """
        Generate caption for a single image.

        Args:
            image: PIL Image or path to image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated caption
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Format messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": prompt}
            ]
        }]

        try:
            # Process input
            inputs = self.processor.apply_chat_template(
                messages,
                padding=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate
            with torch.no_grad():
                gen_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode response
            response = self.processor.tokenizer.decode(
                gen_tokens[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Error generating caption"

    def evaluate_captioning(self, dataset: Dataset,
                          num_samples: Optional[int] = None,
                          prompts: Optional[List[str]] = None) -> EvaluationResult:
        """
        Evaluate image captioning performance.

        Args:
            dataset: Dataset with images and ground truth captions
            num_samples: Number of samples to evaluate (None for all)
            prompts: List of prompts to use

        Returns:
            Evaluation result
        """
        if prompts is None:
            prompts = [
                "Describe this image in detail.",
                "What do you see in this image?",
                "Please provide a detailed caption for this image."
            ]

        # Sample dataset if needed
        if num_samples and len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            eval_dataset = dataset.select(indices)
        else:
            eval_dataset = dataset

        results = []
        total_samples = len(eval_dataset)

        print(f"Evaluating captioning on {total_samples} samples...")

        for i, example in enumerate(eval_dataset):
            if i % 10 == 0:
                print(f"Progress: {i}/{total_samples}")

            # Get image and ground truth
            image = example.get("image")
            ground_truth = example.get("caption", example.get("english_caption", ""))

            if not image or not ground_truth:
                continue

            # Generate caption with random prompt
            prompt = random.choice(prompts)
            generated = self.generate_caption(image, prompt)

            # Store result
            results.append({
                "image_id": i,
                "ground_truth": ground_truth,
                "generated": generated,
                "prompt": prompt
            })

        # Calculate metrics
        metrics = self._calculate_captioning_metrics(results)

        return EvaluationResult(
            metric_name="captioning",
            score=metrics["average_score"],
            details=metrics,
            samples=results[:10]  # Store first 10 samples
        )

    def evaluate_vqa(self, dataset: Dataset,
                    num_samples: Optional[int] = None) -> EvaluationResult:
        """
        Evaluate Visual Question Answering performance.

        Args:
            dataset: Dataset with images, questions, and answers
            num_samples: Number of samples to evaluate

        Returns:
            Evaluation result
        """
        # Sample dataset if needed
        if num_samples and len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            eval_dataset = dataset.select(indices)
        else:
            eval_dataset = dataset

        results = []
        total_samples = len(eval_dataset)

        print(f"Evaluating VQA on {total_samples} samples...")

        for i, example in enumerate(eval_dataset):
            if i % 10 == 0:
                print(f"Progress: {i}/{total_samples}")

            # Get image, question, and answer
            image = example.get("image")
            question = example.get("question", "What do you see?")
            ground_truth = example.get("answer", "")

            if not image:
                continue

            # Generate answer
            generated = self.generate_caption(image, question)

            # Store result
            results.append({
                "image_id": i,
                "question": question,
                "ground_truth": ground_truth,
                "generated": generated
            })

        # Calculate metrics
        metrics = self._calculate_vqa_metrics(results)

        return EvaluationResult(
            metric_name="vqa",
            score=metrics["accuracy"],
            details=metrics,
            samples=results[:10]
        )

    def evaluate_cultural_understanding(self, dataset: Dataset,
                                      num_samples: Optional[int] = None) -> EvaluationResult:
        """
        Evaluate cultural understanding for African contexts.

        Args:
            dataset: Dataset with culturally relevant images
            num_samples: Number of samples to evaluate

        Returns:
            Evaluation result
        """
        cultural_prompts = [
            "Describe the cultural elements visible in this image.",
            "What cultural context can you identify in this image?",
            "Describe any traditional clothing, architecture, or customs in this image.",
            "What story does this image tell about African culture?"
        ]

        # Sample dataset if needed
        if num_samples and len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            eval_dataset = dataset.select(indices)
        else:
            eval_dataset = dataset

        results = []
        total_samples = len(eval_dataset)

        print(f"Evaluating cultural understanding on {total_samples} samples...")

        for i, example in enumerate(eval_dataset):
            if i % 10 == 0:
                print(f"Progress: {i}/{total_samples}")

            image = example.get("image")
            if not image:
                continue

            # Test multiple cultural prompts
            for prompt in cultural_prompts[:2]:  # Use first 2 prompts
                generated = self.generate_caption(image, prompt)

                results.append({
                    "image_id": i,
                    "prompt": prompt,
                    "generated": generated,
                    "cultural_keywords": self._extract_cultural_keywords(generated)
                })

        # Calculate cultural understanding metrics
        metrics = self._calculate_cultural_metrics(results)

        return EvaluationResult(
            metric_name="cultural_understanding",
            score=metrics["cultural_score"],
            details=metrics,
            samples=results[:10]
        )

    def compare_models(self, other_evaluator: 'AyaVisionEvaluator',
                      dataset: Dataset,
                      num_samples: int = 50) -> Dict[str, Any]:
        """
        Compare this model with another model.

        Args:
            other_evaluator: Another evaluator to compare with
            dataset: Dataset for comparison
            num_samples: Number of samples to use

        Returns:
            Comparison results
        """
        # Sample dataset
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            eval_dataset = dataset.select(indices)
        else:
            eval_dataset = dataset

        prompts = ["Describe this image in detail.", "What do you see in this image?"]
        comparisons = []

        print(f"Comparing models on {len(eval_dataset)} samples...")

        for i, example in enumerate(eval_dataset):
            image = example.get("image")
            if not image:
                continue

            prompt = random.choice(prompts)

            # Generate with both models
            response1 = self.generate_caption(image, prompt)
            response2 = other_evaluator.generate_caption(image, prompt)

            comparisons.append({
                "image_id": i,
                "prompt": prompt,
                "model1_response": response1,
                "model2_response": response2,
                "ground_truth": example.get("caption", "")
            })

        # Calculate comparison metrics
        metrics = self._calculate_comparison_metrics(comparisons)

        return {
            "total_comparisons": len(comparisons),
            "metrics": metrics,
            "samples": comparisons[:5]
        }

    def benchmark_speed(self, dataset: Dataset,
                       num_samples: int = 20) -> Dict[str, float]:
        """
        Benchmark model inference speed.

        Args:
            dataset: Dataset for benchmarking
            num_samples: Number of samples to benchmark

        Returns:
            Speed metrics
        """
        # Sample dataset
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            eval_dataset = dataset.select(indices)
        else:
            eval_dataset = dataset

        times = []

        print(f"Benchmarking speed on {len(eval_dataset)} samples...")

        for example in eval_dataset:
            image = example.get("image")
            if not image:
                continue

            start_time = time.time()
            _ = self.generate_caption(image, "Describe this image.")
            end_time = time.time()

            times.append(end_time - start_time)

        return {
            "avg_time_per_sample": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_time": np.sum(times),
            "samples_per_second": 1.0 / np.mean(times)
        }

    def _calculate_captioning_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate captioning evaluation metrics."""
        if not results:
            return {"average_score": 0.0}

        # Simple metrics based on length and keyword matching
        scores = []
        for result in results:
            gt = result["ground_truth"].lower()
            gen = result["generated"].lower()

            # Basic similarity score (can be enhanced with BLEU, ROUGE, etc.)
            score = self._calculate_similarity_score(gt, gen)
            scores.append(score)

        return {
            "average_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "total_samples": len(results)
        }

    def _calculate_vqa_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate VQA evaluation metrics."""
        if not results:
            return {"accuracy": 0.0}

        correct = 0
        total = 0

        for result in results:
            if result["ground_truth"]:
                total += 1
                # Simple exact match (can be enhanced)
                if result["ground_truth"].lower().strip() in result["generated"].lower():
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct_answers": correct,
            "total_questions": total
        }

    def _calculate_cultural_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate cultural understanding metrics."""
        cultural_keywords = [
            "traditional", "culture", "cultural", "heritage", "tribal", "ceremonial",
            "ritual", "community", "village", "indigenous", "authentic", "historic",
            "african", "native", "ancestral", "custom", "celebration", "festival"
        ]

        cultural_mentions = 0
        total_responses = len(results)

        for result in results:
            generated = result["generated"].lower()
            if any(keyword in generated for keyword in cultural_keywords):
                cultural_mentions += 1

        cultural_score = cultural_mentions / total_responses if total_responses > 0 else 0.0

        return {
            "cultural_score": cultural_score,
            "cultural_mentions": cultural_mentions,
            "total_responses": total_responses,
            "cultural_keywords_found": cultural_mentions
        }

    def _calculate_comparison_metrics(self, comparisons: List[Dict]) -> Dict[str, Any]:
        """Calculate model comparison metrics."""
        # Simple preference scoring (in practice, use human evaluation)
        model1_better = 0
        model2_better = 0
        ties = 0

        for comp in comparisons:
            len1 = len(comp["model1_response"])
            len2 = len(comp["model2_response"])

            # Simple heuristic: longer responses might be better (not always true)
            if abs(len1 - len2) < 10:  # Similar length
                ties += 1
            elif len1 > len2:
                model1_better += 1
            else:
                model2_better += 1

        total = len(comparisons)
        return {
            "model1_win_rate": model1_better / total if total > 0 else 0,
            "model2_win_rate": model2_better / total if total > 0 else 0,
            "tie_rate": ties / total if total > 0 else 0,
            "total_comparisons": total
        }

    def _calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate simple similarity score between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _extract_cultural_keywords(self, text: str) -> List[str]:
        """Extract cultural keywords from text."""
        cultural_keywords = [
            "traditional", "culture", "cultural", "heritage", "tribal", "ceremonial",
            "ritual", "community", "village", "indigenous", "authentic", "historic",
            "african", "native", "ancestral", "custom", "celebration", "festival"
        ]

        found_keywords = []
        text_lower = text.lower()

        for keyword in cultural_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return found_keywords


class BenchmarkRunner:
    """
    Runner for standardized benchmarks.
    """

    def __init__(self, evaluator: AyaVisionEvaluator):
        """Initialize with an evaluator."""
        self.evaluator = evaluator

    def run_aya_vision_benchmark(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Run evaluation similar to AyaVisionBench.

        Args:
            dataset: Dataset for evaluation

        Returns:
            Benchmark results
        """
        results = {}

        # Captioning task
        captioning_result = self.evaluator.evaluate_captioning(dataset, num_samples=100)
        results["captioning"] = captioning_result

        # Cultural understanding
        cultural_result = self.evaluator.evaluate_cultural_understanding(dataset, num_samples=50)
        results["cultural"] = cultural_result

        # Speed benchmark
        speed_metrics = self.evaluator.benchmark_speed(dataset, num_samples=20)
        results["speed"] = speed_metrics

        # Overall score (weighted average)
        overall_score = (
            captioning_result.score * 0.5 +
            cultural_result.score * 0.3 +
            min(speed_metrics["samples_per_second"] / 2.0, 1.0) * 0.2  # Normalize speed
        )

        results["overall_score"] = overall_score

        return results


def load_model_for_evaluation(model_path: str,
                            base_model_id: str = "CohereLabs/aya-vision-8b",
                            device: str = "auto") -> Tuple[Any, Any]:
    """
    Load model and processor for evaluation.

    Args:
        model_path: Path to fine-tuned model (LoRA weights)
        base_model_id: Base model identifier
        device: Device to load on

    Returns:
        Tuple of (model, processor)
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Load base model
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    # Load fine-tuned weights if provided
    if model_path and os.path.exists(model_path):
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = base_model

    return model, processor


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results
        output_path: Path to save results
    """
    # Convert any non-serializable objects
    serializable_results = {}

    for key, value in results.items():
        if isinstance(value, EvaluationResult):
            serializable_results[key] = {
                "metric_name": value.metric_name,
                "score": value.score,
                "details": value.details,
                "sample_count": len(value.samples)
            }
        else:
            serializable_results[key] = value

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


# Example usage
def example_evaluation():
    """Example usage of evaluation utilities."""

    # Load model and processor
    """
    model, processor = load_model_for_evaluation(
        "path/to/finetuned/model",
        "CohereLabs/aya-vision-8b"
    )

    # Create evaluator
    evaluator = AyaVisionEvaluator(model, processor)

    # Load evaluation dataset
    eval_dataset = load_dataset("your-eval-dataset")

    # Run captioning evaluation
    captioning_result = evaluator.evaluate_captioning(eval_dataset, num_samples=100)
    print(f"Captioning score: {captioning_result.score}")

    # Run cultural understanding evaluation
    cultural_result = evaluator.evaluate_cultural_understanding(eval_dataset, num_samples=50)
    print(f"Cultural understanding score: {cultural_result.score}")

    # Run full benchmark
    runner = BenchmarkRunner(evaluator)
    benchmark_results = runner.run_aya_vision_benchmark(eval_dataset)

    # Save results
    save_evaluation_results(benchmark_results, "evaluation_results.json")
    """

    pass


if __name__ == "__main__":
    print("Evaluation Utilities for Aya Vision Fine-tuning")
    print("Import this module to use the evaluation functions.")
    example_evaluation()