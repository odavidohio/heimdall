"""
Model Configurations for HEIMDALL
Facilitates switching between different LLM models
"""

# =============================================================================
# AVAILABLE MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    # Lightweight model for quick tests (CPU/GPU)
    "gpt2": {
        "name": "gpt2",
        "use_quantization": False,
        "target_layer": -1,  # Last layer (layer 11)
        "max_length": 512,
        "requires_hf_token": False,
        "vram_usage": "~500MB",
        "description": "Lightweight model for testing (12 layers)"
    },

    # GPT-2 Medium (more robust)
    "gpt2-medium": {
        "name": "gpt2-medium",
        "use_quantization": False,
        "target_layer": -1,  # Last layer (layer 23)
        "max_length": 512,
        "requires_hf_token": False,
        "vram_usage": "~1.5GB",
        "description": "GPT-2 Medium (24 layers)"
    },

    # Mistral 7B (recommended for production)
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.3",
        "use_quantization": True,
        "target_layer": 24,  # Layer where semantic vortices are strongest
        "max_length": 512,
        "requires_hf_token": False,
        "vram_usage": "~5GB (with 4-bit quant)",
        "description": "Mistral 7B with quantization (32 layers)"
    },

    # Llama-3.1 8B (requires HF approval)
    "llama-3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B",
        "use_quantization": True,
        "target_layer": 24,
        "max_length": 512,
        "requires_hf_token": True,
        "vram_usage": "~6GB (with 4-bit quant)",
        "description": "Llama-3.1 8B with quantization (32 layers)"
    },

    # Llama-3.1 70B (requires high-capacity GPU)
    "llama-3.1-70b": {
        "name": "meta-llama/Llama-3.1-70B",
        "use_quantization": True,
        "target_layer": 60,  # Deeper layers for larger models
        "max_length": 512,
        "requires_hf_token": True,
        "vram_usage": "~35GB (with 4-bit quant)",
        "description": "Llama-3.1 70B with quantization (80 layers)"
    },

    # Phi-2 (Microsoft - compact and efficient model)
    "phi-2": {
        "name": "microsoft/phi-2",
        "use_quantization": False,
        "target_layer": -1,
        "max_length": 512,
        "requires_hf_token": False,
        "vram_usage": "~3GB",
        "description": "Microsoft Phi-2 (32 layers, 2.7B params)"
    }
}

# =============================================================================
# ANALYSIS CONFIGURATIONS
# =============================================================================

ANALYSIS_CONFIGS = {
    "fast": {
        "analyze_multihead": False,  # Use head average
        "description": "Fast analysis (attention head average)"
    },

    "precise": {
        "analyze_multihead": True,  # Analyze all heads
        "description": "Precise analysis (all heads, takes maximum R)"
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_config(model_key):
    """
    Returns model configuration

    Args:
        model_key: Model key (e.g., 'gpt2', 'mistral-7b')

    Returns:
        dict: Model configuration
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_key}' not found. Available: {list(MODEL_CONFIGS.keys())}")

    return MODEL_CONFIGS[model_key]

def list_available_models():
    """List all available models with descriptions"""
    print("Available Models:")
    print("=" * 80)

    for key, config in MODEL_CONFIGS.items():
        print(f"\n{key}:")
        print(f"  Name: {config['name']}")
        print(f"  Quantization: {'Yes' if config['use_quantization'] else 'No'}")
        print(f"  VRAM: {config['vram_usage']}")
        print(f"  Requires HF Token: {'Yes' if config['requires_hf_token'] else 'No'}")
        print(f"  Description: {config['description']}")

def check_requirements(model_key):
    """
    Check requirements to run a model

    Args:
        model_key: Model key

    Returns:
        bool: True if requirements are met
    """
    config = get_model_config(model_key)

    import torch

    # Check CUDA if quantization is needed
    if config['use_quantization'] and not torch.cuda.is_available():
        print(f"[WARNING] Model '{model_key}' requires CUDA, but CUDA is not available.")
        print("  Suggestion: Use a smaller model without quantization (e.g., 'gpt2')")
        return False

    # Check HF token
    if config['requires_hf_token']:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token is None:
                print(f"[WARNING] Model '{model_key}' requires a Hugging Face token.")
                print("  Run: huggingface-cli login")
                return False
        except ImportError:
            print("[WARNING] huggingface-hub not installed.")
            print("  Run: pip install huggingface-hub")
            return False

    return True

# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    print("HEIMDALL - Model Configurations")
    print("=" * 80)

    list_available_models()

    print("\n" + "=" * 80)
    print("\nTo use in your script:")
    print(">>> from heimdall.model_config import get_model_config")
    print(">>> config = get_model_config('mistral-7b')")
    print(">>> MODEL_NAME = config['name']")
    print(">>> USE_QUANTIZATION = config['use_quantization']")
