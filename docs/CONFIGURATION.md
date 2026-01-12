# Configuration Guide

## Model Configurations

### Default: Mistral 7B with 4-bit Quantization

```python
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
USE_QUANTIZATION = True
TARGET_LAYER = 24
```

**Requirements:**
- NVIDIA GPU with CUDA
- ~5GB VRAM

---

### Option 1: GPT-2 (No GPU Required)

```python
MODEL_NAME = "gpt2"
USE_QUANTIZATION = False
TARGET_LAYER = -1
```

**Requirements:**
- CPU only
- ~500MB RAM

---

### Option 2: Mistral 7B Full Precision

```python
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
USE_QUANTIZATION = False
TARGET_LAYER = 24
```

**Requirements:**
- NVIDIA GPU with CUDA
- ~14GB VRAM

---

### Option 3: Llama 3.1 8B

```python
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
USE_QUANTIZATION = True
TARGET_LAYER = 24
```

**Requirements:**
- NVIDIA GPU with CUDA
- ~6GB VRAM
- Hugging Face token with Llama access

---

## Analysis Parameters

### Layer Selection

```python
TARGET_LAYER = 24  # Default for 7B models

# Guidelines:
# - Small models (≤24 layers): Use -1 (last layer)
# - Medium models (32 layers): Use 24
# - Large models (70B+): Experiment with 40-60
```

### Context Length

```python
MAX_LENGTH = 512  # Default

# Options:
MAX_LENGTH = 256   # Faster, less context
MAX_LENGTH = 1024  # More context, slower
```

### Multihead Analysis

```python
ANALYZE_MULTIHEAD = True   # Accurate (+15% precision)
ANALYZE_MULTIHEAD = False  # Fast (~2x faster)
```

---

## Performance Comparison

| Configuration | Speed | Precision | VRAM | Recommended For |
|--------------|-------|-----------|------|-----------------|
| GPT-2 CPU | ★★★★★ | ★★☆☆☆ | 500MB | Testing |
| GPT-2 GPU | ★★★★☆ | ★★☆☆☆ | 500MB | Baseline |
| Mistral 4-bit | ★★★☆☆ | ★★★★☆ | 5GB | **Production** |
| Mistral Full | ★★☆☆☆ | ★★★★★ | 14GB | Research |
| Llama 4-bit | ★★★☆☆ | ★★★★★ | 6GB | **Best Quality** |

---

## Environment Variables

```bash
# Hugging Face cache directory
export HF_HOME=/path/to/cache

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false
```

---

## Recommended Configurations

### For Research Papers

```python
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
USE_QUANTIZATION = True
TARGET_LAYER = 24
ANALYZE_MULTIHEAD = True
```

### For Production

```python
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
USE_QUANTIZATION = True
TARGET_LAYER = 24
ANALYZE_MULTIHEAD = False  # Speed priority
```

### For Quick Testing

```python
MODEL_NAME = "gpt2"
USE_QUANTIZATION = False
TARGET_LAYER = -1
ANALYZE_MULTIHEAD = False
```
