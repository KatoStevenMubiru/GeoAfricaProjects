"""
Data Processing Utilities for Aya Vision Fine-tuning

This module provides utilities for processing datasets, formatting data for training,
and handling different data formats for the Aya Vision model.
"""

import os
import json
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from datasets import Dataset, load_dataset
from transformers import AutoProcessor


class AyaVisionDataProcessor:
    """
    Main data processor for Aya Vision fine-tuning.
    Handles dataset loading, formatting, and preprocessing.
    """

    def __init__(self, processor: AutoProcessor, config: Dict[str, Any]):
        """
        Initialize the data processor.

        Args:
            processor: The Aya Vision processor
            config: Configuration dictionary with data processing settings
        """
        self.processor = processor
        self.config = config
        self.prompts = config.get("prompts", {})
        self.field_mappings = config.get("field_mappings", {})

    def load_dataset(self, dataset_id: str, split: str = "train",
                    max_samples: Optional[int] = None) -> Dataset:
        """
        Load dataset from Hugging Face Hub.

        Args:
            dataset_id: Hugging Face dataset identifier
            split: Dataset split to load
            max_samples: Maximum number of samples to load (for testing)

        Returns:
            Loaded dataset
        """
        try:
            dataset = load_dataset(dataset_id, split=split)

            if max_samples is not None and len(dataset) > max_samples:
                dataset = dataset.shuffle(seed=self.config.get("seed", 42))
                dataset = dataset.select(range(max_samples))

            return dataset

        except Exception as e:
            raise ValueError(f"Failed to load dataset {dataset_id}: {e}")

    def format_for_chat_template(self, example: Dict[str, Any],
                                task_type: str = "captioning") -> Dict[str, Any]:
        """
        Format a single example for the Aya Vision chat template.

        Args:
            example: Single dataset example
            task_type: Type of task (captioning, vqa, cultural)

        Returns:
            Formatted example with messages structure
        """
        # Get image
        image = example.get(self.field_mappings.get("image_field", "image"))
        if image is None:
            raise ValueError("No image found in example")

        # Get text content based on task type
        if task_type == "captioning":
            response = self._get_caption_from_example(example)
            prompt = self._get_random_prompt("captioning")

        elif task_type == "vqa":
            response = self._get_answer_from_example(example)
            question = self._get_question_from_example(example)
            prompt = self._get_vqa_prompt(question)

        elif task_type == "cultural":
            response = self._get_caption_from_example(example)
            prompt = self._get_random_prompt("cultural")

        else:
            # Default to captioning
            response = self._get_caption_from_example(example)
            prompt = self._get_random_prompt("captioning")

        return {
            "image": image,
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>\\n{prompt}"
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        }

    def _get_caption_from_example(self, example: Dict[str, Any]) -> str:
        """Extract caption from example using field mappings."""
        caption_fields = self.field_mappings.get("caption_fields",
                                                 ["caption", "english_caption", "description", "text"])

        for field in caption_fields:
            if field in example and example[field]:
                return str(example[field]).strip()

        return "This is an image."  # Fallback

    def _get_question_from_example(self, example: Dict[str, Any]) -> str:
        """Extract question from example."""
        question_field = self.field_mappings.get("question_field", "question")
        return str(example.get(question_field, "What do you see?")).strip()

    def _get_answer_from_example(self, example: Dict[str, Any]) -> str:
        """Extract answer from example."""
        answer_field = self.field_mappings.get("answer_field", "answer")
        return str(example.get(answer_field, "I see an image.")).strip()

    def _get_random_prompt(self, prompt_type: str) -> str:
        """Get a random prompt of the specified type."""
        prompts = self.prompts.get(prompt_type, ["Describe this image."])
        return random.choice(prompts)

    def _get_vqa_prompt(self, question: str) -> str:
        """Format VQA prompt with question."""
        vqa_templates = self.prompts.get("vqa", ["Answer the following question about this image: {question}"])
        template = random.choice(vqa_templates)
        return template.format(question=question)

    def create_train_val_split(self, dataset: Dataset,
                              val_ratio: float = 0.1,
                              seed: int = 42) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into train and validation sets.

        Args:
            dataset: Input dataset
            val_ratio: Fraction of data to use for validation
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        dataset = dataset.shuffle(seed=seed)
        split_point = int(len(dataset) * (1 - val_ratio))

        train_dataset = dataset.select(range(split_point))
        val_dataset = dataset.select(range(split_point, len(dataset)))

        return train_dataset, val_dataset

    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Validate dataset structure and content.

        Args:
            dataset: Dataset to validate

        Returns:
            Validation report
        """
        report = {
            "total_samples": len(dataset),
            "columns": dataset.column_names,
            "sample_fields": {},
            "issues": []
        }

        # Check first few samples
        for i in range(min(5, len(dataset))):
            try:
                example = dataset[i]

                # Check for image
                image_field = self.field_mappings.get("image_field", "image")
                if image_field in example:
                    image = example[image_field]
                    if isinstance(image, Image.Image):
                        report["sample_fields"]["has_image"] = True
                    else:
                        report["issues"].append(f"Image field exists but not PIL Image in sample {i}")
                else:
                    report["issues"].append(f"No image field found in sample {i}")

                # Check for text content
                caption_fields = self.field_mappings.get("caption_fields",
                                                        ["caption", "english_caption", "description"])
                has_text = any(field in example and example[field] for field in caption_fields)
                if not has_text:
                    report["issues"].append(f"No text content found in sample {i}")

            except Exception as e:
                report["issues"].append(f"Error processing sample {i}: {e}")

        return report


class DatasetFormatter:
    """
    Utility class for formatting datasets from various sources.
    """

    @staticmethod
    def from_csv(csv_path: str, image_dir: str,
                image_column: str = "image_path",
                caption_column: str = "caption") -> Dataset:
        """
        Create dataset from CSV file with image paths.

        Args:
            csv_path: Path to CSV file
            image_dir: Directory containing images
            image_column: Column containing image filenames/paths
            caption_column: Column containing captions

        Returns:
            Dataset object
        """
        df = pd.read_csv(csv_path)

        def load_image(example):
            image_path = os.path.join(image_dir, example[image_column])
            example["image"] = Image.open(image_path).convert("RGB")
            example["caption"] = example[caption_column]
            return example

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(load_image)

        return dataset

    @staticmethod
    def from_json(json_path: str, image_dir: Optional[str] = None) -> Dataset:
        """
        Create dataset from JSON file.

        Args:
            json_path: Path to JSON file
            image_dir: Directory containing images (if paths are relative)

        Returns:
            Dataset object
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        def process_example(example):
            if "image_path" in example:
                image_path = example["image_path"]
                if image_dir and not os.path.isabs(image_path):
                    image_path = os.path.join(image_dir, image_path)
                example["image"] = Image.open(image_path).convert("RGB")
            return example

        dataset = Dataset.from_list(data)
        if "image_path" in dataset.column_names:
            dataset = dataset.map(process_example)

        return dataset

    @staticmethod
    def from_directory(image_dir: str, caption_file: Optional[str] = None) -> Dataset:
        """
        Create dataset from directory of images.

        Args:
            image_dir: Directory containing images
            caption_file: Optional JSON file with captions keyed by filename

        Returns:
            Dataset object
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # Find all image files
        image_files = []
        for file in os.listdir(image_dir):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file)

        # Load captions if provided
        captions = {}
        if caption_file and os.path.exists(caption_file):
            with open(caption_file, 'r') as f:
                captions = json.load(f)

        # Create dataset
        data = []
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            caption = captions.get(image_file, f"An image named {image_file}")

            data.append({
                "image": Image.open(image_path).convert("RGB"),
                "caption": caption,
                "filename": image_file
            })

        return Dataset.from_list(data)


class AfricanLanguageProcessor:
    """
    Specialized processor for African language datasets.
    """

    # Common African languages supported by Aya Vision
    AFRICAN_LANGUAGES = {
        "sw": "Swahili",
        "ig": "Igbo",
        "ha": "Hausa",
        "rw": "Kinyarwanda",
        "yo": "Yoruba",
        "zu": "Zulu",
        "lg": "Luganda",
        "ny": "Chichewa",
        "tw": "Twi",
        "so": "Somali",
        "ar": "Arabic"  # Including Arabic dialects
    }

    def __init__(self, target_language: str = "sw"):
        """
        Initialize processor for specific African language.

        Args:
            target_language: Target language code
        """
        self.target_language = target_language
        self.language_name = self.AFRICAN_LANGUAGES.get(target_language, target_language)

    def create_cultural_prompts(self) -> List[str]:
        """
        Create culturally relevant prompts for African contexts.

        Returns:
            List of cultural prompts
        """
        return [
            f"Describe this image in the context of {self.language_name} culture.",
            f"What cultural elements from {self.language_name} society can you identify?",
            f"How would someone from {self.language_name} culture describe this scene?",
            "Describe the traditional or cultural aspects visible in this image.",
            "What story does this image tell about African culture?",
            "Identify any traditional clothing, architecture, or customs in this image.",
            "Describe this image focusing on its cultural significance."
        ]

    def enhance_for_african_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance dataset example with African cultural context.

        Args:
            example: Original example

        Returns:
            Enhanced example with cultural prompts
        """
        cultural_prompts = self.create_cultural_prompts()

        # Add cultural variation
        if random.random() < 0.3:  # 30% chance of cultural prompt
            prompt = random.choice(cultural_prompts)
        else:
            standard_prompts = [
                "Describe this image in detail.",
                "What do you see in this image?",
                "Please provide a detailed caption for this image."
            ]
            prompt = random.choice(standard_prompts)

        # Get response text
        caption = example.get("caption", example.get("english_caption", "An image."))

        return {
            "image": example["image"],
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>\\n{prompt}"
                },
                {
                    "role": "assistant",
                    "content": caption
                }
            ],
            "language": self.target_language,
            "cultural_context": True
        }


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("PyYAML required for loading YAML configs. Install with: pip install pyyaml")
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def save_dataset_info(dataset: Dataset, output_path: str):
    """
    Save dataset information and statistics.

    Args:
        dataset: Dataset to analyze
        output_path: Path to save info JSON
    """
    info = {
        "total_samples": len(dataset),
        "columns": dataset.column_names,
        "sample_preview": []
    }

    # Add sample previews (first 3 samples, text only)
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        preview = {}
        for key, value in sample.items():
            if key == "image":
                preview[key] = "<PIL.Image>"
            elif isinstance(value, (str, int, float, bool)):
                preview[key] = value
            else:
                preview[key] = str(type(value))
        info["sample_preview"].append(preview)

    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)


# Example usage functions
def example_usage():
    """Example usage of the data processing utilities."""

    # Example 1: Load dataset from Hugging Face
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("CohereLabs/aya-vision-8b")
    config = {
        "prompts": {
            "captioning": ["Describe this image.", "What do you see?"],
            "cultural": ["Describe the cultural elements in this image."]
        },
        "field_mappings": {
            "image_field": "image",
            "caption_fields": ["caption", "english_caption"]
        }
    }

    data_processor = AyaVisionDataProcessor(processor, config)
    dataset = data_processor.load_dataset("your-dataset-id")

    # Format for training
    formatted_dataset = dataset.map(
        lambda x: data_processor.format_for_chat_template(x, "captioning")
    )
    """

    # Example 2: Create dataset from CSV
    """
    dataset = DatasetFormatter.from_csv(
        "captions.csv",
        "images/",
        image_column="filename",
        caption_column="description"
    )
    """

    # Example 3: African language processing
    """
    african_processor = AfricanLanguageProcessor("sw")  # Swahili
    enhanced_dataset = dataset.map(african_processor.enhance_for_african_context)
    """

    pass


if __name__ == "__main__":
    print("Data Processing Utilities for Aya Vision Fine-tuning")
    print("Import this module to use the data processing functions.")
    example_usage()