# Installation Guide

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 5GB+ VRAM for Mistral 7B (with quantization)

## Quick Installation

```bash
# Clone repository
git clone https://github.com/odavidohio/heimdall.git
cd heimdall

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/check_environment.py
```

## Dependencies

The main dependencies are:

```
torch>=2.0.0
transformers>=4.30.0
bitsandbytes>=0.41.0
gudhi>=3.8.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

## CUDA Setup

For GPU acceleration:

1. **Check CUDA availability:**
   ```bash
   python scripts/check_cuda.py
   ```

2. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify GPU detection:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

See [CUDA_SETUP.md](CUDA_SETUP.md) for detailed instructions.

## CPU-Only Installation

If you don't have an NVIDIA GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then use GPT-2 for testing:
```bash
python tests/test_gpt2.py
```

## Troubleshooting

### CUDA Not Found

```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### GUDHI Installation Error

```bash
pip uninstall gudhi -y
pip install gudhi --no-cache-dir
```

### Out of Memory

Reduce model size or use quantization:
```python
USE_QUANTIZATION = True  # in model_config.py
```
