# CUDA Setup Guide

## Prerequisites

- NVIDIA GPU (GTX 10xx or newer recommended)
- Windows 10/11 or Linux
- 5GB+ VRAM for Mistral 7B with quantization

---

## Quick Start

```bash
# Check CUDA availability
python scripts/check_cuda.py

# If CUDA not available, run installer
python scripts/install_pytorch_cuda.py
```

---

## Manual Installation

### Step 1: Verify NVIDIA Driver

```bash
nvidia-smi
```

You should see GPU information. If not, install drivers from:
https://www.nvidia.com/Download/index.aspx

### Step 2: Install PyTorch with CUDA

**For CUDA 12.1 (recommended for modern GPUs):**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8 (better compatibility):**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Troubleshooting

### CUDA Not Detected

1. **Update NVIDIA drivers** to latest version
2. **Reinstall PyTorch** with correct CUDA version
3. **Restart** your terminal/IDE after installation

### Out of Memory (OOM)

Enable quantization in your config:
```python
USE_QUANTIZATION = True  # Reduces VRAM from 14GB to ~5GB
```

### BitsAndBytes Errors

On Windows, ensure you have Visual Studio Build Tools:
```bash
pip install bitsandbytes --upgrade
```

---

## VRAM Requirements

| Configuration | VRAM Required |
|--------------|---------------|
| GPT-2 (CPU) | 500MB RAM |
| GPT-2 (GPU) | 500MB VRAM |
| Mistral 7B (4-bit) | ~5GB VRAM |
| Mistral 7B (full) | ~14GB VRAM |
| Llama 8B (4-bit) | ~6GB VRAM |

---

## Environment Variables

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# Custom Hugging Face cache location
export HF_HOME=/path/to/cache
```

---

## Verification Script

Run the comprehensive CUDA diagnostic:

```bash
python scripts/check_cuda.py
```

This will:
1. Verify NVIDIA GPU detection
2. Check PyTorch CUDA support
3. Test GPU operations
4. Provide recommendations
