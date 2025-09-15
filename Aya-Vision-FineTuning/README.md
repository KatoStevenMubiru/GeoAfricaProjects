# Aya Vision 8B Fine-Tuning for African Languages

<div align="center">

![Aya Vision](https://img.shields.io/badge/Model-Aya%20Vision%208B-blue)
![License](https://img.shields.io/badge/License-CC--BY--NC--4.0-green)
![Platform](https://img.shields.io/badge/Platform-Kaggle-orange)
![Language](https://img.shields.io/badge/Language-Python-yellow)

*Fine-tuning the state-of-the-art multilingual vision-language model for enhanced African language understanding*

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)

## 🌍 Overview

This repository provides a comprehensive pipeline for fine-tuning the **Aya Vision 8B** model, specifically optimized for African language vision-language tasks. Building upon Cohere's state-of-the-art multilingual vision-language model, this project enhances performance on African cultural contexts and languages.

### Why Aya Vision 8B?

- **🌐 Multilingual**: Supports 23 languages including African languages
- **🎯 Efficient**: 8B parameters with excellent performance-to-size ratio
- **🔧 Optimized**: Advanced architecture with SigLIP2 vision encoder
- **📱 Practical**: Available on WhatsApp and other platforms
- **🏆 SOTA**: Outperforms larger models (up to 79% win rate vs competitors)

### Key Advantages of This Implementation

- **💾 Memory Efficient**: 4-bit quantization + LoRA for training on single GPU
- **⚡ Fast Training**: Optimized for Kaggle's free GPU environment
- **🔄 Resumable**: Automatic checkpoint saving and resumption
- **📊 Comprehensive**: Full evaluation and monitoring pipeline
- **🚀 Production Ready**: Direct deployment to Hugging Face Hub

## ✨ Features

### 🎯 Core Capabilities

- **Fine-tuning Pipeline**: Complete end-to-end training workflow
- **Memory Optimization**: 4-bit quantization with BitsAndBytes
- **LoRA Training**: Parameter-efficient fine-tuning
- **Automatic Deployment**: Direct upload to Hugging Face Hub
- **Progress Monitoring**: Real-time training metrics and logging
- **Model Testing**: Built-in inference validation

### 🛠️ Technical Features

- **Multi-GPU Support**: Automatic device mapping
- **Mixed Precision**: BFloat16 training for speed and stability
- **Gradient Checkpointing**: Memory-efficient backpropagation
- **Data Processing**: Flexible dataset formatting and preprocessing
- **Error Handling**: Comprehensive error recovery and reporting

### 📊 Monitoring & Evaluation

- **TensorBoard Integration**: Real-time training visualization
- **Checkpoint Management**: Automatic saving every 50 steps
- **Memory Tracking**: GPU memory usage monitoring
- **Performance Metrics**: Training loss and convergence tracking

## 🚀 Quick Start

### Prerequisites

- Kaggle account with GPU access
- Hugging Face account with write token
- **🚨 CRITICAL: Access to Aya Vision 8B gated model**
- Dataset uploaded to Hugging Face Hub

### 🔐 Gated Model Access Required

**Aya Vision 8B is a GATED model** - you must request access:

1. **Visit**: [CohereLabs/aya-vision-8b](https://huggingface.co/CohereLabs/aya-vision-8b)
2. **Click**: "Request Access" button
3. **Wait**: For approval (usually 24h)
4. **License**: CC-BY-NC-4.0 (Non-commercial only)

**Without access, the training will fail with authentication errors.**

### 30-Second Setup

1. **Create Kaggle Notebook**:
   ```bash
   # In Kaggle, create new notebook with GPU T4 x2
   ```

2. **Upload Files**:
   - Upload `aya_vision_8b_finetuning.ipynb` to Kaggle
   - Set your HF token in Kaggle Secrets as `HF_TOKEN`

3. **Configure Dataset**:
   ```python
   CONFIG["dataset_id"] = "your-username/your-dataset"
   ```

4. **Install Aya Vision Compatible Transformers**:
   ```bash
   pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision'
   ```

5. **Run Training**:
   - Execute all cells in sequence
   - Monitor progress in TensorBoard
   - Model auto-uploads to Hub when complete

## 🔧 Detailed Setup

### Step 1: Environment Preparation

#### Kaggle Setup
1. Navigate to [Kaggle](https://www.kaggle.com)
2. Create a new notebook
3. Select **GPU T4 x2** or **P100** accelerator
4. Enable internet access in settings

#### Secrets Configuration
1. In Kaggle notebook, go to **Add-ons → Secrets**
2. Add secret:
   - **Name**: `HF_TOKEN`
   - **Value**: Your Hugging Face write token

### Step 2: Dataset Preparation

Your dataset should be in Hugging Face format with the following structure:

```python
{
    "image": PIL.Image,           # Input image
    "caption": str,               # English caption
    "english_caption": str,       # Alternative caption field
    "culture": str,              # Cultural context (optional)
    "language": str,             # Target language (optional)
}
```

#### Supported Dataset Formats
- **Image-Caption Pairs**: Basic captioning datasets
- **VQA Format**: Visual Question Answering datasets
- **Conversational**: Multi-turn dialogue with images
- **Custom**: Any format (modify `format_for_aya_vision()` function)

### Step 3: Model Configuration

```python
CONFIG = {
    # Model settings
    "base_model_id": "CohereLabs/aya-vision-8b",
    "new_model_name": "aya-vision-8b-african-finetuned",

    # Training settings
    "num_epochs": 1,              # Start with 1, increase to 2-3 for better quality
    "batch_size": 2,              # Reduce to 1 if OOM errors
    "learning_rate": 2e-4,        # Optimal for LoRA fine-tuning

    # LoRA settings
    "lora_r": 16,                 # Rank - higher = more parameters
    "lora_alpha": 32,             # Alpha - affects learning rate scaling
    "lora_dropout": 0.05,         # Dropout for regularization
}
```

## 💻 Usage

### Training Your Model

1. **Load the notebook** in Kaggle
2. **Set your configuration** in the CONFIG dictionary
3. **Run all cells** sequentially
4. **Monitor progress** via output logs and TensorBoard
5. **Download or use** your trained model from Hugging Face Hub

### Using the Fine-tuned Model

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch

# Load base model and processor
base_model = AutoModelForImageTextToText.from_pretrained(
    "CohereLabs/aya-vision-8b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("CohereLabs/aya-vision-8b")

# Load your fine-tuned LoRA weights
model = PeftModel.from_pretrained(base_model, "your-username/your-model")

# Prepare input
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "url": "path/to/image.jpg"},
        {"type": "text", "text": "Describe this image in detail."}
    ]
}]

# Generate response
inputs = processor.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.3)

response = processor.tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)
print(response)
```

### Batch Inference

```python
# Process multiple images
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
captions = []

for image_path in images:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": "Describe this image"}
        ]
    }]

    inputs = processor.apply_chat_template(messages, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    caption = processor.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    captions.append(caption)
```

## ⚙️ Configuration

### Training Parameters

| Parameter | Default | Description | Recommendations |
|-----------|---------|-------------|-----------------|
| `num_epochs` | 1 | Training epochs | Start with 1, increase to 2-3 for better quality |
| `batch_size` | 2 | Batch size per device | Reduce to 1 if OOM errors occur |
| `gradient_accumulation_steps` | 8 | Gradient accumulation | Increase if reducing batch_size |
| `learning_rate` | 2e-4 | Learning rate | 1e-4 to 5e-4 for LoRA |
| `max_seq_length` | 1024 | Maximum sequence length | Reduce to 512 if OOM |

### LoRA Parameters

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `lora_r` | 16 | LoRA rank | Higher = more parameters, better quality |
| `lora_alpha` | 32 | LoRA alpha | Typically 2x the rank |
| `lora_dropout` | 0.05 | Dropout rate | 0.05-0.1 for regularization |
| `target_modules` | attention | Target layers | Attention layers work best |

### Memory Optimization

```python
# For 16GB GPU (Kaggle T4)
CONFIG.update({
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 1024,
    "lora_r": 16
})

# For 8GB GPU or memory issues
CONFIG.update({
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 512,
    "lora_r": 8
})
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce memory usage
CONFIG["batch_size"] = 1
CONFIG["gradient_accumulation_steps"] = 16
CONFIG["max_seq_length"] = 512
CONFIG["lora_r"] = 8

# Clear cache between operations
torch.cuda.empty_cache()
```

#### 2. Slow Training

**Symptoms**: Very slow training progress

**Solutions**:
```python
# Ensure proper GPU usage
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model device: {next(model.parameters()).device}")

# Optimize data loading
CONFIG["dataloader_num_workers"] = 2
```

#### 3. Model Upload Failures

**Symptoms**: Failed to push to Hugging Face Hub

**Solutions**:
```python
# Check token permissions
from huggingface_hub import HfApi
api = HfApi()
print(api.whoami())

# Manual upload
trainer.push_to_hub(commit_message="Manual upload after training")
```

#### 4. Dataset Loading Issues

**Symptoms**: Dataset fails to load or format incorrectly

**Solutions**:
```python
# Check dataset structure
dataset = load_dataset("your-dataset")
print(dataset[0].keys())

# Modify formatting function
def format_for_aya_vision(example):
    # Adjust field names based on your dataset
    return {
        "image": example["image"],
        "messages": [...]
    }
```

### Performance Optimization

#### Training Speed
- Use `gradient_checkpointing=True` (enabled by default)
- Set `dataloader_num_workers=2`
- Enable mixed precision with `bf16=True`

#### Memory Efficiency
- Use 4-bit quantization (enabled by default)
- Implement gradient accumulation
- Reduce sequence length for longer training

#### Quality Improvements
- Increase number of epochs (2-3 for better results)
- Use higher LoRA rank (32-64 for complex tasks)
- Validate on held-out data during training

## 📊 Results

### Expected Outcomes

After fine-tuning, you should expect:

1. **Improved African Language Understanding**: Better comprehension of cultural contexts
2. **Enhanced Visual Reasoning**: More accurate image descriptions
3. **Domain Adaptation**: Better performance on your specific use case
4. **Reduced Hallucination**: More factual and grounded responses

### Evaluation Metrics

The notebook includes automatic evaluation using:

- **Qualitative Assessment**: Visual inspection of generated captions
- **Comparison with Base Model**: Side-by-side evaluation
- **Cultural Relevance**: Assessment of African cultural understanding
- **Language Quality**: Fluency and naturalness of outputs

### Benchmarking

For comprehensive evaluation, consider using:

- **AyaVisionBench**: Multilingual vision-language benchmark
- **mWildVision**: Multilingual version of Wild Vision Bench
- **Custom Evaluation**: Domain-specific test sets

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Areas for Contribution

- **Dataset Integration**: Support for new dataset formats
- **Evaluation Metrics**: Additional evaluation methods
- **Performance Optimization**: Memory and speed improvements
- **Documentation**: Better guides and examples
- **Testing**: More comprehensive test coverage

## 📚 Resources

### Documentation

- **[Aya Vision Paper](https://arxiv.org/abs/2412.04261)**: Original research paper
- **[Hugging Face Model](https://huggingface.co/CohereLabs/aya-vision-8b)**: Base model documentation
- **[LoRA Paper](https://arxiv.org/abs/2106.09685)**: Low-Rank Adaptation methodology
- **[TRL Documentation](https://huggingface.co/docs/trl)**: Training framework

### Community

- **[Cohere Discord](https://discord.gg/co-mmunity)**: Community discussions
- **[Hugging Face Forums](https://discuss.huggingface.co/)**: Technical support
- **[GitHub Issues](https://github.com/your-repo/issues)**: Bug reports and feature requests

### Related Projects

- **[Aya Expanse](https://huggingface.co/CohereForAI/aya-expanse-32b)**: Multilingual language models
- **[AfriAya Dataset](https://huggingface.co/datasets/Afri-Aya)**: African language datasets
- **[Pangea](https://huggingface.co/Pangea7B)**: Multilingual multimodal models

## 📄 License

This project is licensed under the **CC-BY-NC 4.0 License** - see the [LICENSE](LICENSE) file for details.

### Important Notes

- **Non-Commercial Use Only**: This license restricts commercial usage
- **Attribution Required**: Please cite the original Aya Vision paper
- **Share-Alike**: Derivative works must use the same license

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{aya-vision-2025,
  title={Aya Vision: A Deepdive into Multilingual Multimodality},
  author={Dash, Saurabh and Nan, Yiyang and Ahmadian, Arash and others},
  journal={arXiv preprint arXiv:2412.04261},
  year={2025}
}
```

For this fine-tuning implementation:

```bibtex
@misc{aya-vision-finetuning-2025,
  title={Aya Vision 8B Fine-Tuning for African Languages},
  author={GeoAfrica Projects},
  year={2025},
  url={https://github.com/your-repo/aya-vision-finetuning}
}
```

## 🙏 Acknowledgments

Special thanks to:

- **Cohere For AI** for releasing Aya Vision models
- **Hugging Face** for the transformers library and model hosting
- **The African NLP Community** for datasets and cultural insights
- **Kaggle** for providing free GPU access for research

---

<div align="center">

**Made with ❤️ for the African AI Community**

[🌟 Star this repo](https://github.com/your-repo) • [🐛 Report Bug](https://github.com/your-repo/issues) • [💡 Request Feature](https://github.com/your-repo/issues)

</div>