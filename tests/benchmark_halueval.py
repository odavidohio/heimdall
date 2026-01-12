"""
HEIMDALL - HaluEval Dataset Benchmark
Scientific evaluation using the official HaluEval dataset
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gudhi
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "mistralai/Mistral-7B-v0.3"
LAYER_IDX = 24  # Layer 24: semantic vortex layer
N_SAMPLES = 100  # Number of samples

print("="*70)
print("HEIMDALL - BENCHMARK WITH HALUEVAL DATASET")
print("="*70)
print(f"\nModel: {MODEL_ID}")
print(f"Analysis layer: {LAYER_IDX}")
print(f"Samples: {N_SAMPLES}")
print()

# =============================================================================
# R-SCORE CALCULATION FUNCTION
# =============================================================================

def calculate_heimdall_r(matrix):
    """
    Calculate cycle density (H1) via Cubical Complex

    Args:
        matrix: Attention matrix [seq_len, seq_len]

    Returns:
        float: R-Score (Max Persistence / Cycle Count)
    """
    # Min-max normalization to ensure topological relief
    m_min, m_max = matrix.min(), matrix.max()
    if m_max - m_min > 1e-10:
        matrix = (matrix - m_min) / (m_max - m_min)
    else:
        return 0.0

    # Create cubical complex
    cubical = gudhi.CubicalComplex(
        dimensions=matrix.shape,
        top_dimensional_cells=matrix.flatten()
    )
    cubical.compute_persistence()
    persistence = cubical.persistence_intervals_in_dimension(1)  # H1: Cycles/Vortices

    if len(persistence) == 0:
        return 0.0

    # R-Score: Max Persistence / Cycle Count
    lifetimes = [d - b for b, d in persistence if d != float('inf')]
    if not lifetimes:
        return 0.0

    return max(lifetimes) / len(lifetimes)

# =============================================================================
# MODEL LOADING WITH 4-BIT QUANTIZATION
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    print("[ERROR] This script requires NVIDIA GPU with CUDA")
    print("   Alternatives:")
    print("   1. Use Google Colab: https://colab.research.google.com")
    print("   2. Run: python tests/test_gpt2.py (CPU version)")
    exit(1)

print(f"Loading {MODEL_ID} at Layer {LAYER_IDX}...")
print("Configuring 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {MODEL_ID}...")
print("(First run: download ~5GB - may take a few minutes)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    output_attentions=True,
    attn_implementation="eager",  # Crucial for attention extraction
    trust_remote_code=True
)

print("[OK] Model loaded successfully!")

# =============================================================================
# LOADING HALUEVAL DATASET
# =============================================================================

print("\n" + "="*70)
print("LOADING HALUEVAL DATASET")
print("="*70)

try:
    print("Trying to load official HaluEval (pminervini/HaluEval)...")
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    samples = dataset.shuffle(seed=42).select(range(N_SAMPLES))

    print(f"[OK] {len(samples)} samples loaded successfully!")

    # Show example
    if len(samples) > 0:
        print("\nSample example:")
        print(f"  Question: {samples[0]['question'][:80]}...")
        print(f"  Correct Answer: {samples[0].get('answer', '')[:80]}...")
        print(f"  Hallucinated Answer: {samples[0].get('hallucination', '')[:80]}...")

except Exception as e:
    print(f"\n[WARNING] Failed to load remote dataset: {e}")
    print("Using local fallback dataset...")

    # Fallback: expanded local dataset
    base_samples = [
        {"question": "Which magazine was started first, Arthur's Magazine or First for Women?",
         "answer": "Arthur's Magazine was started first in 1844, while First for Women was started in 1989.",
         "hallucination": "First for Women was started first in 1844, while Arthur's Magazine was started in 1989."},
        {"question": "What is the capital of France?",
         "answer": "The capital of France is Paris, located in the north-central part of the country.",
         "hallucination": "The capital of France is London, which is actually the capital of England."},
        {"question": "Who wrote Hamlet?",
         "answer": "William Shakespeare wrote Hamlet around 1600.",
         "hallucination": "Charles Dickens wrote Hamlet in the Victorian era."},
        {"question": "What is the capital of Italy?",
         "answer": "Rome is the capital of Italy.",
         "hallucination": "Venice is the capital of Italy."},
        {"question": "Who is the author of 'The Old Man and the Sea'?",
         "answer": "Ernest Hemingway is the author of the book.",
         "hallucination": "F. Scott Fitzgerald is the author of the book."},
    ]
    # Expand to N_SAMPLES
    samples = (base_samples * (N_SAMPLES // len(base_samples) + 1))[:N_SAMPLES]
    print(f"[OK] Using {len(samples)} fallback samples")

# =============================================================================
# RUNNING ANALYSIS WITH TWO SEPARATE INFERENCES
# =============================================================================

print("\n" + "="*70)
print("RUNNING TOPOLOGICAL ANALYSIS")
print("="*70)
print("[IMPORTANT] Making TWO inferences per sample (factual + hallucination)")
print()

results = []

for i, entry in enumerate(tqdm(samples, desc="Heimdall Scanning")):
    try:
        # Extract sample data
        question = entry.get('question', entry.get('user_query', ""))

        # Robust capture: try new keys, fallback to old ones
        ans_factual = entry.get('answer') or entry.get('right_answer') or entry.get('factual')
        ans_hallu = entry.get('hallucination') or entry.get('hallucinated_answer')

        # SANITY CHECK: If still empty or equal, log warning
        if not ans_factual or not ans_hallu or ans_factual == ans_hallu:
            print(f"[WARNING] Dataset error at sample {i}! Keys found: {list(entry.keys())}")
            continue

        # Need TWO separate inferences
        scores = {}

        for tipo, resposta in [('factual', ans_factual), ('hallu', ans_hallu)]:
            # Combine Question + Answer to generate attention manifold
            full_text = f"Question: {question} Answer: {resposta}"

            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs,
                                output_attentions=True,
                                use_cache=False)  # Force full attention

                # Extract target layer (e.g., 24)
                # outputs.attentions is a tuple of tensors [batch, heads, seq, seq]
                attention_tensor = outputs.attentions[LAYER_IDX][0]  # [heads, seq, seq]

                # Calculate R-Score for EACH head and take maximum
                r_scores = []
                for head_idx in range(attention_tensor.shape[0]):
                    att_matrix = attention_tensor[head_idx].cpu().numpy()  # [seq, seq]
                    r = calculate_heimdall_r(att_matrix)
                    r_scores.append(r)

                # Store max R-Score (dominant vortex)
                scores[tipo] = max(r_scores) if r_scores else 0.0

        # Save result
        results.append({
            'sample_id': i,
            'question': question[:100],  # Truncate to save space
            'R_factual': scores['factual'],
            'R_hallucinated': scores['hallu'],
            'correct_detection': scores['factual'] > scores['hallu']  # True if detected correctly
        })

        # Clear GPU cache periodically
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\nError at sample {i}: {e}")
        continue

df_results = pd.DataFrame(results)

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

# Descriptive statistics
r_fact_mean = df_results['R_factual'].mean()
r_fact_std = df_results['R_factual'].std()
r_hallu_mean = df_results['R_hallucinated'].mean()
r_hallu_std = df_results['R_hallucinated'].std()

print(f"\nMean R-Score (Factual):     {r_fact_mean:.4f} +/- {r_fact_std:.4f}")
print(f"Mean R-Score (Hallucination):  {r_hallu_mean:.4f} +/- {r_hallu_std:.4f}")
print(f"Mean Difference:             {r_fact_mean - r_hallu_mean:.4f}")

# Hypothesis test (paired t-test)
t_stat, p_value = ttest_rel(df_results['R_factual'], df_results['R_hallucinated'])
print(f"\nPaired T-Test:")
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value:     {p_value:.4e}")

if p_value < 0.001:
    print("   [OK] HIGHLY significant difference (p < 0.001)")
elif p_value < 0.01:
    print("   [OK] Very significant difference (p < 0.01)")
elif p_value < 0.05:
    print("   [OK] Significant difference (p < 0.05)")
else:
    print("   [!] NOT significant difference (p >= 0.05)")

# Detection accuracy
accuracy = df_results['correct_detection'].mean()
print(f"\nDetection Accuracy: {accuracy*100:.2f}%")

# =============================================================================
# ROC CURVE
# =============================================================================

print("\nCalculating classification metrics...")

# Prepare data for ROC
y_true = np.concatenate([
    np.zeros(len(df_results)),  # 0 = Factual
    np.ones(len(df_results))    # 1 = Hallucination
])

y_scores = np.concatenate([
    -df_results['R_factual'].values,      # Negative because high R = coherent
    -df_results['R_hallucinated'].values
])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Optimal threshold (Youden Index)
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold = -thresholds[optimal_idx]

print(f"AUC-ROC: {roc_auc:.4f}")
print(f"Optimal Threshold: {optimal_threshold:.4f} (Youden Index)")

# Binary classification using optimal threshold
y_pred = (y_scores > thresholds[optimal_idx]).astype(int)
report = classification_report(y_true, y_pred, target_names=['Factual', 'Hallucination'])
print("\nClassification Report:")
print(report)

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

plt.style.use('seaborn-v0_8-paper')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. R-Score Distribution (KDE)
sns.kdeplot(
    data=df_results['R_factual'],
    label="Factual (Laminar)",
    fill=True,
    color="#2ecc71",
    alpha=0.6,
    ax=axes[0]
)
sns.kdeplot(
    data=df_results['R_hallucinated'],
    label="Hallucination (Turbulent)",
    fill=True,
    color="#e74c3c",
    alpha=0.6,
    ax=axes[0]
)
axes[0].axvline(optimal_threshold, color='black', linestyle='--', alpha=0.5, label=f'Threshold={optimal_threshold:.3f}')
axes[0].set_title("R-Score Distribution (HaluEval)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("R-Score (Coherence Ratio)", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 2. ROC Curve
axes[1].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random')
axes[1].scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100, label='Optimal Point', zorder=3)
axes[1].set_title("ROC Curve", fontsize=14, fontweight='bold')
axes[1].set_xlabel("False Positive Rate", fontsize=12)
axes[1].set_ylabel("True Positive Rate", fontsize=12)
axes[1].legend(loc="lower right", fontsize=10)
axes[1].grid(True, alpha=0.3)

# 3. Comparative Boxplot
data_for_box = pd.DataFrame({
    'R-Score': np.concatenate([df_results['R_factual'], df_results['R_hallucinated']]),
    'Type': ['Factual']*len(df_results) + ['Hallucination']*len(df_results)
})
sns.boxplot(data=data_for_box, x='Type', y='R-Score', palette=['#2ecc71', '#e74c3c'], ax=axes[2])
axes[2].set_title("R-Score Comparison", fontsize=14, fontweight='bold')
axes[2].set_ylabel("R-Score", fontsize=12)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('heimdall_halueval_benchmark.png', dpi=300, bbox_inches='tight')
print("[OK] Plot saved: heimdall_halueval_benchmark.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================

# CSV with detailed results
df_results.to_csv('heimdall_halueval_results.csv', index=False)
print("[OK] Results saved: heimdall_halueval_results.csv")

# JSON with aggregated metrics
summary = {
    "model": MODEL_ID,
    "layer": LAYER_IDX,
    "n_samples": len(df_results),
    "statistics": {
        "factual_mean": float(r_fact_mean),
        "factual_std": float(r_fact_std),
        "hallucinated_mean": float(r_hallu_mean),
        "hallucinated_std": float(r_hallu_std),
        "difference": float(r_fact_mean - r_hallu_mean)
    },
    "hypothesis_test": {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05)
    },
    "classification": {
        "auc_roc": float(roc_auc),
        "accuracy": float(accuracy),
        "optimal_threshold": float(optimal_threshold)
    }
}

with open('heimdall_halueval_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("[OK] Summary saved: heimdall_halueval_summary.json")

# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "="*70)
print("FINAL REPORT - HEIMDALL BENCHMARK")
print("="*70)

print(f"""
STATISTICS:
   Samples processed: {len(df_results)}
   R-Score Factual:      {r_fact_mean:.4f} +/- {r_fact_std:.4f}
   R-Score Hallucination:   {r_hallu_mean:.4f} +/- {r_hallu_std:.4f}
   Difference:            {r_fact_mean - r_hallu_mean:.4f}

STATISTICAL SIGNIFICANCE:
   T-test: t={t_stat:.4f}, p={p_value:.4e}
   Result: {'SIGNIFICANT [OK]' if p_value < 0.05 else 'NOT SIGNIFICANT [X]'}

DETECTION PERFORMANCE:
   AUC-ROC:      {roc_auc:.4f}
   Accuracy:     {accuracy*100:.2f}%
   Threshold:    {optimal_threshold:.4f}

INTERPRETATION:
   R > {optimal_threshold:.4f}  -> COHERENT (Factual)
   R < {optimal_threshold:.4f}  -> HALLUCINATION (Turbulent)

FILES GENERATED:
   heimdall_halueval_benchmark.png  (Visualizations)
   heimdall_halueval_results.csv    (Detailed results)
   heimdall_halueval_summary.json   (Aggregated metrics)
""")

print("="*70)
print("[OK] BENCHMARK COMPLETED SUCCESSFULLY!")
print("="*70)
print()
