# Aya Vision 8B Setup Instructions

## 🚨 Critical Requirements

### 1. Gated Model Access (REQUIRED)

Aya Vision 8B is a **GATED MODEL** - you MUST have access before starting:

1. **Request Access**: Visit [CohereLabs/aya-vision-8b](https://huggingface.co/CohereLabs/aya-vision-8b)
2. **Click**: "Request Access" button
3. **Fill Form**: Describe your research/educational use case
4. **Wait**: Approval usually takes 24 hours
5. **Verify**: Check you can view model files after approval

**⚠️ Without access, training will fail with 401/403 errors**

### 2. Special Transformers Version (REQUIRED)

Aya Vision requires a **specific transformers version** from source:

```bash
# DO NOT use regular transformers
# pip install transformers  # ❌ This won't work

# USE THIS INSTEAD:
pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision'
```

### 3. Hardware Requirements

**Minimum:**
- 16GB GPU memory (Kaggle T4)
- batch_size=1, max_seq_length=512

**Recommended:**
- 24GB+ GPU memory
- batch_size=2, max_seq_length=1024

**Model Size:** 8.63B parameters (large!)

## 📋 Step-by-Step Setup

### Step 1: Verify Gated Access

Test your access before training:

```python
from huggingface_hub import HfApi

api = HfApi()
try:
    # Test model access
    model_info = api.model_info("CohereLabs/aya-vision-8b", token=True)
    print("✅ Access granted!")
except Exception as e:
    print(f"❌ Access denied: {e}")
    print("Request access at: https://huggingface.co/CohereLabs/aya-vision-8b")
```

### Step 2: Install Dependencies

```bash
# Essential packages
pip install torch>=2.0.0 torchvision pillow

# Special transformers version for Aya Vision
pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision'

# Training packages
pip install datasets accelerate peft trl bitsandbytes
pip install huggingface-hub wandb
```

### Step 3: Configure Memory Settings

For **16GB GPU (Kaggle T4)**:
```python
CONFIG = {
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 512,
    "lora_r": 16,
}
```

For **24GB+ GPU**:
```python
CONFIG = {
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 1024,
    "lora_r": 32,
}
```

### Step 4: Test Model Loading

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# Test processor loading
processor = AutoProcessor.from_pretrained(
    "CohereLabs/aya-vision-8b",
    trust_remote_code=True,
    token=True
)

# Test model loading with quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
    "CohereLabs/aya-vision-8b",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    token=True
)

print("✅ Model loaded successfully!")
```

## 🔍 Troubleshooting

### Authentication Errors (401/403)

```
❌ Error: "401 Unauthorized" or "403 Forbidden"
```

**Solution:**
1. Request access at the model page
2. Wait for approval email
3. Check your HF token has read permissions
4. Verify token in Kaggle secrets: `HF_TOKEN`

### Transformers Version Error

```
❌ Error: "Model type aya_vision is not supported"
```

**Solution:**
```bash
# Uninstall regular transformers
pip uninstall transformers

# Install Aya Vision compatible version
pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision'
```

### Memory Errors (OOM)

```
❌ Error: "CUDA out of memory"
```

**Solution:**
1. Reduce batch_size to 1
2. Increase gradient_accumulation_steps to 16
3. Reduce max_seq_length to 256
4. Use smaller LoRA rank (r=8)
5. Restart kernel and clear GPU cache

### Model Loading Slow

```
ℹ️ Model taking 10+ minutes to load
```

**Expected:** Aya Vision 8B has 8.63B parameters - loading takes time!

**Tips:**
- First download: 15-20 minutes
- Subsequent loads: 5-10 minutes
- Kaggle caches models between sessions

## 📊 Model Specifications

- **Parameters:** 8.63 billion
- **Architecture:** Command R7B + SigLIP2-patch14-384
- **Context Length:** 16,384 tokens
- **Languages:** 23 languages
- **Image Processing:** Up to 12 tiles × 169 tokens/tile
- **License:** CC-BY-NC-4.0 (Non-commercial)

## ✅ Pre-Flight Checklist

Before starting training:

- [ ] ✅ Requested and received Aya Vision 8B access
- [ ] ✅ Installed special transformers version from git
- [ ] ✅ HF_TOKEN configured in Kaggle secrets
- [ ] ✅ GPU T4 or P100 selected in Kaggle
- [ ] ✅ Internet access enabled in notebook
- [ ] ✅ Dataset uploaded to Hugging Face Hub
- [ ] ✅ Memory settings configured for your GPU
- [ ] ✅ Test model loading completed successfully

## 🚀 Ready to Train!

Once all checkboxes are ✅, you're ready to run the full training pipeline!

The training will:
1. Load the 8.63B parameter model with 4-bit quantization
2. Apply LoRA adapters for efficient fine-tuning
3. Train on your African language dataset
4. Automatically save checkpoints every 50 steps
5. Upload the final model to Hugging Face Hub

**Expected Training Time:** 2-6 hours depending on dataset size and GPU.

Good luck! 🎯