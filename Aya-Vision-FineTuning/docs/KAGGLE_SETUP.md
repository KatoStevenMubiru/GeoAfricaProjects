# Kaggle Setup Guide for Aya Vision 8B Fine-tuning

This comprehensive guide walks you through setting up and running the Aya Vision 8B fine-tuning process on Kaggle's free GPU environment.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Kaggle Account Setup](#kaggle-account-setup)
- [Dataset Preparation](#dataset-preparation)
- [Notebook Setup](#notebook-setup)
- [Environment Configuration](#environment-configuration)
- [Running the Training](#running-the-training)
- [Monitoring Progress](#monitoring-progress)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## 🔧 Prerequisites

### Required Accounts

1. **Kaggle Account**: [Sign up at kaggle.com](https://www.kaggle.com/account/login)
2. **Hugging Face Account**: [Sign up at huggingface.co](https://huggingface.co/join)
3. **Dataset Access**: Your dataset should be uploaded to Hugging Face Hub

### Required Tokens

1. **Hugging Face Write Token**:
   - Go to [HF Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with **Write** permissions
   - Save this token securely

## 🏗️ Kaggle Account Setup

### Step 1: Verify Your Kaggle Account

1. Log into your Kaggle account
2. Navigate to **Settings** → **Account**
3. Verify your phone number (required for GPU access)
4. Complete account verification if prompted

### Step 2: Check GPU Quota

1. Go to **Settings** → **Account**
2. Check your **GPU Quota** status
3. New accounts get 30 hours of GPU time per week
4. Verified accounts get additional GPU access

### Step 3: Enable Required Features

1. Navigate to **Settings** → **Privacy**
2. Ensure these are enabled:
   - ✅ **Internet access** (required for model downloads)
   - ✅ **GPU acceleration** (required for training)

## 📊 Dataset Preparation

### Upload Your Dataset to Hugging Face

1. **Install Hugging Face CLI** (on your local machine):
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. **Create Dataset Repository**:
   ```bash
   # Create a new dataset repository
   huggingface-cli repo create your-dataset-name --type dataset
   ```

3. **Upload Dataset Files**:
   ```python
   from datasets import Dataset, load_dataset
   from PIL import Image
   import pandas as pd

   # Example: Upload from local files
   # Adjust this based on your data format

   # Load your data
   images = []  # List of PIL Images
   captions = []  # List of captions

   # Create dataset
   dataset = Dataset.from_dict({
       "image": images,
       "caption": captions
   })

   # Push to hub
   dataset.push_to_hub("your-username/your-dataset-name")
   ```

### Dataset Format Requirements

Your dataset should have these columns:
- **`image`**: PIL Image objects or image paths
- **`caption`** or **`english_caption`**: Text descriptions
- Optional: **`culture`**, **`language`**, **`question`**, **`answer`**

## 📝 Notebook Setup

### Step 1: Create New Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com)
2. Click **"Create"** → **"New Notebook"**
3. Choose **"Notebook"** (not "Script")

### Step 2: Configure Notebook Settings

1. **In the right panel**, configure:
   - **Accelerator**: Select **"GPU T4 x2"** or **"GPU P100"**
   - **Internet**: Turn **ON** (essential!)
   - **Environment**: Keep **"Latest Available"**

2. **Notebook Title**: Give it a descriptive name like:
   - "Aya Vision 8B Fine-tuning for African Languages"
   - "Custom Aya Vision Training Pipeline"

### Step 3: Upload the Notebook File

1. **Option A - Copy and Paste**:
   - Copy the content from `aya_vision_8b_finetuning.ipynb`
   - Paste into Kaggle notebook cells

2. **Option B - Upload File**:
   - Download the `.ipynb` file to your computer
   - In Kaggle, click **"File"** → **"Import Notebook"**
   - Upload the file

## 🔐 Environment Configuration

### Step 1: Set Up Secrets

1. In your Kaggle notebook, click **"Add-ons"** → **"Secrets"**
2. Add a new secret:
   - **Label**: `HF_TOKEN`
   - **Value**: Your Hugging Face write token
3. Click **"Add"**

### Step 2: Verify GPU Access

Add this cell at the beginning of your notebook:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ No GPU available! Check your accelerator settings.")
```

### Step 3: Configure Your Settings

Update the `CONFIG` dictionary in the notebook:

```python
CONFIG = {
    # YOUR SETTINGS
    "dataset_id": "your-username/your-dataset-name",  # ← Change this!
    "new_model_name": "aya-vision-8b-your-finetuned",  # ← Change this!

    # Training settings (adjust based on your needs)
    "num_epochs": 1,  # Start with 1, increase for better quality
    "batch_size": 2,  # Reduce to 1 if you get OOM errors
    "max_samples": None,  # Set to 100 for quick testing

    # Keep other settings as default initially
}
```

## 🚀 Running the Training

### Step 1: Initial Validation

Before starting full training:

1. **Test with small sample**:
   ```python
   CONFIG["max_samples"] = 10  # Test with 10 samples first
   ```

2. **Run first few cells** to verify:
   - Dependencies install correctly
   - Dataset loads successfully
   - Model downloads without errors
   - GPU is properly detected

### Step 2: Start Full Training

1. **Reset for full training**:
   ```python
   CONFIG["max_samples"] = None  # Use full dataset
   ```

2. **Run all cells sequentially**:
   - ✅ Execute each cell one by one
   - ✅ Wait for each cell to complete before proceeding
   - ✅ Check output for any errors

### Step 3: Monitor Session Time

Kaggle has session limits:
- **GPU sessions**: ~12 hours maximum
- **Save frequently**: The notebook auto-saves checkpoints every 50 steps
- **Plan accordingly**: For large datasets, you may need multiple sessions

## 📊 Monitoring Progress

### Training Logs

Monitor these key indicators:

```python
# Look for these in the output:
✅ "Model loaded successfully!"
✅ "Training dataset size: X samples"
✅ "🚀 Starting training..."
✅ "Step X/Y | Loss: 0.XX | Time: XX.Xs"
```

### TensorBoard (Optional)

To view TensorBoard logs:

```python
# In a new cell:
%load_ext tensorboard
%tensorboard --logdir ./logs
```

### GPU Memory Usage

Monitor memory usage throughout training:

```python
# Check memory periodically
if torch.cuda.is_available():
    memory_used = torch.cuda.memory_allocated() / 1e9
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {memory_used:.1f} GB / {memory_total:.1f} GB")
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce memory usage
CONFIG["batch_size"] = 1  # Reduce from 2 to 1
CONFIG["gradient_accumulation_steps"] = 16  # Increase to maintain effective batch size
CONFIG["max_seq_length"] = 512  # Reduce from 1024

# Clear cache
torch.cuda.empty_cache()
```

#### 2. Session Timeout

**Error**: Session disconnects during training

**Solutions**:
- Training automatically saves checkpoints every 50 steps
- Restart session and re-run from the last checkpoint
- Consider reducing dataset size for faster completion

#### 3. Model Download Fails

**Error**: Cannot download model files

**Solutions**:
```python
# Verify internet is enabled
import requests
try:
    response = requests.get("https://httpbin.org/get")
    print("✅ Internet connection working")
except:
    print("❌ No internet connection - check notebook settings")
```

#### 4. Dataset Loading Issues

**Error**: Dataset not found or fails to load

**Solutions**:
```python
# Test dataset access
from datasets import load_dataset
try:
    dataset = load_dataset("your-username/your-dataset", split="train")
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
except Exception as e:
    print(f"❌ Dataset error: {e}")
```

#### 5. Hugging Face Authentication

**Error**: Authentication failed

**Solutions**:
```python
# Verify token
from huggingface_hub import HfApi
try:
    api = HfApi()
    user = api.whoami()
    print(f"✅ Logged in as: {user['name']}")
except Exception as e:
    print(f"❌ Auth error: {e}")
    print("Check your HF_TOKEN in Kaggle secrets")
```

### Performance Optimization

#### For Faster Training

```python
CONFIG.update({
    "dataloader_num_workers": 2,  # Parallel data loading
    "gradient_checkpointing": True,  # Memory efficiency
    "bf16": True,  # Mixed precision
})
```

#### For Better Quality

```python
CONFIG.update({
    "num_epochs": 3,  # More training epochs
    "learning_rate": 1e-4,  # Lower learning rate
    "lora_r": 32,  # Higher LoRA rank
})
```

#### For Memory Efficiency

```python
CONFIG.update({
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 512,
    "lora_r": 8,
})
```

## 💡 Best Practices

### Before Starting Training

1. **Test with small dataset**: Use `max_samples=10` first
2. **Verify all configurations**: Double-check dataset ID and model name
3. **Check GPU quota**: Ensure you have enough time remaining
4. **Backup important data**: Download checkpoints periodically

### During Training

1. **Monitor progress regularly**: Check logs every 30 minutes
2. **Watch memory usage**: Ensure it's not growing unboundedly
3. **Don't interrupt**: Let training complete naturally
4. **Stay within session limits**: Plan for ~10-hour sessions

### After Training

1. **Test the model**: Run inference on sample images
2. **Save locally**: Download model files as backup
3. **Document results**: Keep notes on what worked
4. **Share with community**: Upload successful models to Hub

### Resource Management

#### GPU Quota Tips

- **New users**: 30 hours per week
- **Phone verified**: Additional hours
- **Competitions**: Extra quota for active participation
- **Check remaining time**: Monitor in account settings

#### Cost-Effective Training

- **Start small**: Test with minimal settings first
- **Use resumable training**: Leverage checkpoint saving
- **Optimize batch size**: Find the sweet spot for your GPU
- **Consider multiple short sessions**: Better than one long session

## 📋 Quick Checklist

Before clicking "Run All":

- [ ] ✅ GPU T4 or P100 selected
- [ ] ✅ Internet access enabled
- [ ] ✅ HF_TOKEN secret configured
- [ ] ✅ Dataset ID updated in CONFIG
- [ ] ✅ Model name updated in CONFIG
- [ ] ✅ Phone number verified on Kaggle
- [ ] ✅ Sufficient GPU quota remaining
- [ ] ✅ Dataset tested and accessible
- [ ] ✅ All dependencies will install correctly

## 🆘 Getting Help

### If Things Go Wrong

1. **Check the output logs**: Look for specific error messages
2. **Restart and retry**: Sometimes a fresh session helps
3. **Reduce complexity**: Start with minimal settings
4. **Ask for help**: Use Kaggle discussions or HF forums

### Community Resources

- **Kaggle Forums**: [kaggle.com/discussions](https://www.kaggle.com/discussions)
- **Hugging Face Discord**: [Discord community](https://discord.gg/hugging-face)
- **GitHub Issues**: Report bugs in the project repository

### Support Channels

- **Technical Issues**: Kaggle support or HF forums
- **Model Questions**: Cohere community discussions
- **Dataset Problems**: Check HF dataset documentation

---

## 🎯 Final Tips for Success

1. **Start Simple**: Begin with default settings and small datasets
2. **Iterate Gradually**: Increase complexity as you gain confidence
3. **Document Everything**: Keep notes on what works for your use case
4. **Be Patient**: Fine-tuning takes time, but results are worth it
5. **Share Results**: Contribute back to the community

Good luck with your Aya Vision fine-tuning journey! 🚀

---

*Last updated: January 2025*
*Compatible with: Aya Vision 8B, Kaggle GPU environments, Hugging Face Hub*