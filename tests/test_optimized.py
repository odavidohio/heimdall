"""
HEIMDALL - Optimized LLM Hallucination Detector
Optimized version with 4-bit quantization for large models (7B+)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gudhi
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from scipy.stats import ttest_rel
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Available configurations:
# - "gpt2": Lightweight model for quick tests (~500MB, CPU/GPU)
# - "mistralai/Mistral-7B-v0.3": 7B model (~5GB VRAM with quantization)
# - "meta-llama/Llama-3.1-8B": Llama-3 8B (requires HF approval)

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
USE_QUANTIZATION = True  # 4-bit quantization enabled (requires CUDA)
TARGET_LAYER = 24  # Layer 24: semantic vortex layer

# =============================================================================
# 1. TEST DATASET (HaluEval-inspired)
# =============================================================================

print("="*70)
print("HEIMDALL - Hallucination Detection via Topology")
print("="*70)
print(f"\nModel: {MODEL_NAME}")
print(f"4-bit Quantization: {'Enabled' if USE_QUANTIZATION else 'Disabled'}")
print(f"Analysis layer: {TARGET_LAYER}")
print()

test_samples = [
    {"right": "The capital of France is Paris.", "wrong": "The capital of France is London."},
    {"right": "Water boils at 100 degrees Celsius at sea level.", "wrong": "Water boils at 500 degrees Celsius at sea level."},
    {"right": "Albert Einstein developed the theory of relativity.", "wrong": "Albert Einstein invented the internet in 1990."},
    {"right": "The moon orbits the Earth every 27.3 days.", "wrong": "The moon is made of green cheese and orbits Mars."},
    {"right": "Photosynthesis is the process plants use to convert sunlight into energy.", "wrong": "Photosynthesis is the process where dogs bark at the moon."},
    {"right": "The Pacific Ocean is the largest ocean on Earth.", "wrong": "The Pacific Ocean is a small lake located in Switzerland."},
    {"right": "Shakespeare wrote the play Hamlet.", "wrong": "Shakespeare was an astronaut who walked on the sun."},
    {"right": "Brazil is located in South America.", "wrong": "Brazil is a small island near the North Pole."},
    {"right": "DNA carries the genetic instructions for life.", "wrong": "DNA is a type of pasta used in Italian cooking."},
    {"right": "The Great Wall of China is a series of fortifications.", "wrong": "The Great Wall of China was built by aliens to stop laser beams."}
]

# =============================================================================
# 2. MODEL LOADING WITH QUANTIZATION (IF ENABLED)
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# Validation: Quantization requires CUDA
if USE_QUANTIZATION and device != "cuda":
    print("\n[WARNING] 4-bit quantization requires NVIDIA GPU with CUDA!")
    print("    Options:")
    print("    1. Use NVIDIA GPU")
    print("    2. Change in script: MODEL_NAME='gpt2', USE_QUANTIZATION=False")
    print("    3. Use free Google Colab: https://colab.research.google.com\n")
    raise RuntimeError("4-bit quantization not available without CUDA")

if USE_QUANTIZATION and device == "cuda":
    print("Configuring 4-bit quantization (BitsAndBytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 (best quality)
        bnb_4bit_use_double_quant=True,  # Double quantization for extra savings
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # Automatic distribution between CPU/GPU
        output_attentions=True,
        trust_remote_code=True,
        attn_implementation="eager"  # REQUIRED: eager implementation for full attention extraction
    )
    print("[OK] Model loaded with 4-bit quantization")
else:
    print("Loading model without quantization...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_attentions=True,
        trust_remote_code=True,
        attn_implementation="eager"  # REQUIRED: eager implementation for full attention extraction
    ).to(device)
    print("[OK] Model loaded in standard mode")

# Configure pad_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# 3. R-SCORE CALCULATION FUNCTION (Topological Coherence Ratio)
# =============================================================================

def get_R_score(text, analyze_multihead=True):
    """
    Calculate Topological Coherence Ratio (R) via H1 Persistent Homology

    Args:
        text: Text to analyze
        analyze_multihead: If True, analyzes all heads and returns max R

    Returns:
        R-Score: max(lifetime_h1) / count(h1)
    """
    if not text:
        return 1.0

    # Tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)  # Disable cache for compatibility

    # Select target layer
    attention_layer = outputs.attentions[TARGET_LAYER][0].cpu().numpy()

    if analyze_multihead:
        # Analyze all heads and return max R (dominant vortex)
        max_R = 0.0

        for head_idx in range(attention_layer.shape[0]):
            head_att = attention_layer[head_idx]
            R = _compute_persistence_R(head_att)
            max_R = max(max_R, R)

        return max_R
    else:
        # Use head average (original approach)
        mean_att = np.mean(attention_layer, axis=0)
        return _compute_persistence_R(mean_att)

def _compute_persistence_R(attention_matrix):
    """
    Calculate R-Score via Cubical Homology (O(n log n))
    """
    # Min-Max Normalization
    att_norm = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min() + 1e-10)

    # Heimdall Engine: Cubical Complex
    try:
        cubical = gudhi.CubicalComplex(
            dimensions=attention_matrix.shape,
            top_dimensional_cells=att_norm.flatten()
        )
        cubical.compute_persistence()
        persistence = cubical.persistence_intervals_in_dimension(1)  # H1: vortices

        if len(persistence) > 0:
            lifetimes = persistence[:, 1] - persistence[:, 0]
            # R = max(lifetime) / count(features) -> Coherence Ratio
            return np.max(lifetimes) / len(lifetimes)
    except Exception as e:
        print(f"Warning: Error in persistence calculation - {e}")
        return 1.0

    return 1.0

# =============================================================================
# 4. RUN EXPERIMENT
# =============================================================================

results = []
print("\n" + "="*70)
print("Starting topological analysis...")
print("="*70)

for item in tqdm(test_samples, desc="Processing samples"):
    r_fact = get_R_score(item['right'])
    r_hallu = get_R_score(item['wrong'])
    results.append({"factual": r_fact, "hallucinated": r_hallu})

df_res = pd.DataFrame(results)

# =============================================================================
# 5. SCIENTIFIC VISUALIZATION
# =============================================================================

print("\nGenerating visualizations...")

plt.style.use('seaborn-v0_8-paper')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5.1 KDE Plot (R-Score Distribution)
sns.kdeplot(data=df_res['factual'], label="Factual (Laminar)", fill=True, color="#2ecc71", alpha=0.6, ax=axes[0])
sns.kdeplot(data=df_res['hallucinated'], label="Hallucination (Turbulent)", fill=True, color="#e74c3c", alpha=0.6, ax=axes[0])
axes[0].set_title("Sanity Distribution (R-Score)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("R-Score (Coherence Ratio)", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 5.2 ROC Curve
y_true = np.concatenate([np.zeros(len(df_res)), np.ones(len(df_res))])
y_scores = np.concatenate([-df_res['factual'], -df_res['hallucinated']])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random')
axes[1].set_title("Detection Performance", fontsize=14, fontweight='bold')
axes[1].set_xlabel("False Positive Rate", fontsize=12)
axes[1].set_ylabel("True Positive Rate", fontsize=12)
axes[1].legend(loc="lower right", fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heimdall_optimized_results.png', dpi=300, bbox_inches='tight')
print("[OK] Plot saved: 'heimdall_optimized_results.png'")

# =============================================================================
# 6. STATISTICAL ANALYSIS
# =============================================================================

t_stat, p_value = ttest_rel(df_res['factual'], df_res['hallucinated'])

print("\n" + "="*70)
print("RESULTS - HEIMDALL ANALYSIS")
print("="*70)
print(f"\nDescriptive Statistics:")
print(f"  Mean R-Score (Factual):     {df_res['factual'].mean():.4f} +/- {df_res['factual'].std():.4f}")
print(f"  Mean R-Score (Hallucination):  {df_res['hallucinated'].mean():.4f} +/- {df_res['hallucinated'].std():.4f}")
print(f"\nHypothesis Test (Paired T-Test):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_value:.4e}")
print(f"\nDetection Performance:")
print(f"  AUC-ROC:     {roc_auc:.4f}")

# Calculate optimal threshold (Youden criterion)
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold = -thresholds[optimal_idx]
print(f"  Optimal threshold: {optimal_threshold:.4f} (Youden Index)")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

results_summary = {
    "model": MODEL_NAME,
    "quantization": USE_QUANTIZATION,
    "target_layer": TARGET_LAYER,
    "device": device,
    "factual_mean": float(df_res['factual'].mean()),
    "factual_std": float(df_res['factual'].std()),
    "hallucinated_mean": float(df_res['hallucinated'].mean()),
    "hallucinated_std": float(df_res['hallucinated'].std()),
    "t_statistic": float(t_stat),
    "p_value": float(p_value),
    "auc_roc": float(roc_auc),
    "optimal_threshold": float(optimal_threshold)
}

with open('heimdall_optimized_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n[OK] Results saved: 'heimdall_optimized_results.json'")
print("\n" + "="*70)
print("Analysis completed successfully!")
print("="*70)
