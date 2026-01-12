"""
HEIMDALL - Basic Test with GPT-2 (CPU Compatible)
Lightweight test using GPT-2 model for quick validation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gudhi
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from scipy.stats import ttest_rel
import json

# 1. CREATE LOCAL TEST DATASET (HaluEval-style)
print("Creating test samples (HaluEval pattern)...")
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

# 2. MODEL SETUP
MODEL_NAME = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_attentions=True).to(device)

def get_R_score(text):
    """Calculate R-Score using persistent homology"""
    if not text: return 1.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Attention extraction (Final Layer)
    attention = outputs.attentions[-1][0].cpu().numpy()
    mean_att = np.mean(attention, axis=0)

    # Min-Max Normalization
    att_norm = (mean_att - mean_att.min()) / (mean_att.max() - mean_att.min() + 1e-10)

    # Heimdall Engine (O(n log n))
    cubical = gudhi.CubicalComplex(dimensions=mean_att.shape, top_dimensional_cells=att_norm.flatten())
    cubical.compute_persistence()
    persistence = cubical.persistence_intervals_in_dimension(1)

    if len(persistence) > 0:
        lifetimes = persistence[:, 1] - persistence[:, 0]
        return np.max(lifetimes) / len(lifetimes)
    return 1.0

# 3. RUN EXPERIMENT
results = []
print("\nStarting topological analysis...")

for item in tqdm(test_samples):
    r_fact = get_R_score(item['right'])
    r_hallu = get_R_score(item['wrong'])
    results.append({"factual": r_fact, "hallucinated": r_hallu})

df_res = pd.DataFrame(results)

# 4. SCIENTIFIC VISUALIZATION
plt.figure(figsize=(12, 5))

# KDE Plot
plt.subplot(1, 2, 1)
sns.kdeplot(df_res['factual'], label="Factual (Laminar)", fill=True, color="green")
sns.kdeplot(df_res['hallucinated'], label="Hallucination (Turbulent)", fill=True, color="red")
plt.title("Sanity Distribution (R-Score)")
plt.legend()

# ROC Curve
plt.subplot(1, 2, 2)
y_true = np.concatenate([np.zeros(len(df_res)), np.ones(len(df_res))])
y_scores = np.concatenate([-df_res['factual'], -df_res['hallucinated']])  # Inverted for detection

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("Detection Performance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('heimdall_gpt2_results.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'heimdall_gpt2_results.png'")

# 5. STATISTICAL RESULTS
t_stat, p_value = ttest_rel(df_res['factual'], df_res['hallucinated'])

print("\n" + "="*50)
print("HEIMDALL ANALYSIS RESULTS")
print("="*50)
print(f"\nMean R-Score (Factual): {df_res['factual'].mean():.4f} +/- {df_res['factual'].std():.4f}")
print(f"Mean R-Score (Hallucination): {df_res['hallucinated'].mean():.4f} +/- {df_res['hallucinated'].std():.4f}")
print(f"\nT-test: t={t_stat:.4f}, p={p_value:.4e}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Save results
results_summary = {
    "factual_mean": float(df_res['factual'].mean()),
    "factual_std": float(df_res['factual'].std()),
    "hallucinated_mean": float(df_res['hallucinated'].mean()),
    "hallucinated_std": float(df_res['hallucinated'].std()),
    "t_statistic": float(t_stat),
    "p_value": float(p_value),
    "auc_roc": float(roc_auc)
}

with open('heimdall_gpt2_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\nResults saved in 'heimdall_gpt2_summary.json'")
print("\nAnalysis completed!")
