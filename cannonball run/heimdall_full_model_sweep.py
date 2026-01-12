"""
Heimdall Full Model & Layer Sweep
Complete sweep: 3 models x 32 layers each
Objective: Find the best model+layer combination
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
import warnings
import gc

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# 3 Models to test
MODELS_CONFIG = {
    "phi3": {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "name": "Phi-3 Mini 3.8B",
        "layers": 32
    },
    "mistral": {
        "id": "mistralai/Mistral-7B-v0.3",
        "name": "Mistral 7B",
        "layers": 32
    },
    "llama": {
        "id": "meta-llama/Meta-Llama-3.1-8B",
        "name": "Llama-3.1 8B",
        "layers": 32
    }
}

# Layers to test (all from 0 to 31)
LAYERS_TO_TEST = list(range(32))
N_SAMPLES = 100  # Samples per model+layer combination

print("="*70)
print("HEIMDALL - FULL MODEL AND LAYER SWEEP")
print("="*70)
print(f"\nModels to test: {len(MODELS_CONFIG)}")
for key, config in MODELS_CONFIG.items():
    print(f"  - {config['name']} ({config['id']})")
print(f"\nLayers per model: {len(LAYERS_TO_TEST)} (0-31)")
print(f"Samples per layer: {N_SAMPLES}")
print(f"\nTotal inferences: {len(MODELS_CONFIG) * len(LAYERS_TO_TEST) * N_SAMPLES * 2:,}")
print("Estimated total time: ~4-6 hours")
print()

# =============================================================================
# R-SCORE CALCULATION FUNCTION
# =============================================================================

def calculate_heimdall_r(matrix):
    # 1. CONTRAST AMPLIFICATION (Essential for Llama 3.1)
    # Llama is very 'smooth'. We raise to power to highlight attention peaks.
    matrix = np.power(matrix, 0.5)

    # Normalization
    m_min, m_max = matrix.min(), matrix.max()
    if m_max - m_min < 1e-12: return 0.0
    matrix = (matrix - m_min) / (m_max - m_min)

    cubical = gudhi.CubicalComplex(dimensions=matrix.shape, top_dimensional_cells=matrix.flatten())
    cubical.compute_persistence()
    persistence = cubical.persistence_intervals_in_dimension(1)

    if len(persistence) == 0: return 0.0

    lifetimes = [d - b for b, d in persistence if d != float('inf')]
    # We use Log1p to stabilize the sensor scale
    return np.log1p(max(lifetimes) / len(lifetimes))

# =============================================================================
# DATASET LOADING
# =============================================================================

print("Loading HaluEval (calibration sample)...")

try:
    print("Trying pufanyi/HaluEval...")
    dataset = load_dataset("pufanyi/HaluEval", "qa", split="data")
    samples = dataset.shuffle(seed=42).select(range(N_SAMPLES))
    print(f"[OK] {len(samples)} samples loaded!")
except Exception as e:
    print(f"[WARNING] Error loading dataset: {e}")
    print("Using expanded local fallback...")
    base_samples = [
        {"question": "Which magazine was started first, Arthur's Magazine or First for Women?",
         "right_answer": "Arthur's Magazine was started first in 1844.",
         "hallucinated_answer": "First for Women was started first in 1844."},
        {"question": "What is the capital of France?",
         "right_answer": "Paris is the capital of France.",
         "hallucinated_answer": "London is the capital of France."},
        {"question": "Who wrote Hamlet?",
         "right_answer": "William Shakespeare wrote Hamlet around 1600.",
         "hallucinated_answer": "Charles Dickens wrote Hamlet in the Victorian era."},
        {"question": "What is the speed of light?",
         "right_answer": "The speed of light is approximately 299,792 km/s.",
         "hallucinated_answer": "The speed of light is 50 meters per second."},
        {"question": "Who discovered penicillin?",
         "right_answer": "Alexander Fleming discovered penicillin in 1928.",
         "hallucinated_answer": "Alexander Fleming discovered penicillin on Mars in 3000."},
    ]
    samples = (base_samples * (N_SAMPLES // len(base_samples) + 1))[:N_SAMPLES]
    print(f"[OK] Using {len(samples)} fallback samples")

# =============================================================================
# QUANTIZATION CONFIGURATION
# =============================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# =============================================================================
# FULL SWEEP - MODEL LOOP
# =============================================================================

all_results = []

for model_key, model_config in MODELS_CONFIG.items():
    print("\n" + "="*70)
    print(f"STARTING SWEEP: {model_config['name']}")
    print("="*70)
    print(f"Model ID: {model_config['id']}")
    print(f"Layers to test: {len(LAYERS_TO_TEST)}")
    print()

    # --- MODEL LOADING ---
    print(f"Loading {model_config['name']}...")

    tokenizer = AutoTokenizer.from_pretrained(model_config['id'])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config['id'],
        quantization_config=bnb_config,
        device_map="auto",
        output_attentions=True,
        attn_implementation="eager",
        trust_remote_code=True
    )

    print(f"[OK] {model_config['name']} loaded successfully!")

    # --- LAYER SWEEP ---
    model_sweep_results = []

    for layer in tqdm(LAYERS_TO_TEST, desc=f"{model_key.upper()} Layers", position=0):
        layer_scores = []

        # Nested progress bar for samples
        pbar_samples = tqdm(samples, desc=f"  Layer {layer:2d} Samples", position=1, leave=False)

        for entry in pbar_samples:
            q = entry.get('question', '')

            # Robust capture: try new keys, fallback to old ones
            ans_fact = entry.get('answer') or entry.get('right_answer')
            ans_hallu = entry.get('hallucination') or entry.get('hallucinated_answer')

            # SANITY CHECK: If still empty or equal, skip
            if not ans_fact or not ans_hallu or ans_fact == ans_hallu:
                continue

            for tipo, resp in [('fact', ans_fact), ('hallu', ans_hallu)]:
                # Combine question + answer
                text = f"Question: {q} Answer: {resp}"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")

                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True, use_cache=False)

                    # Average all heads to simplify
                    att = outputs.attentions[layer][0].mean(dim=0).cpu().numpy()
                    r_score = calculate_heimdall_r(att)

                    layer_scores.append({
                        'type': tipo,
                        'r': r_score
                    })

            # Update description with real-time statistics
            if len(layer_scores) > 0:
                df_temp = pd.DataFrame(layer_scores)
                r_fact_temp = df_temp[df_temp['type'] == 'fact']['r']
                r_hallu_temp = df_temp[df_temp['type'] == 'hallu']['r']
                if len(r_fact_temp) > 0 and len(r_hallu_temp) > 0:
                    diff_temp = r_fact_temp.mean() - r_hallu_temp.mean()
                    pbar_samples.set_postfix({'D': f'{diff_temp:+.4f}'})

        pbar_samples.close()

        # Layer Statistics
        df_layer = pd.DataFrame(layer_scores)
        r_fact = df_layer[df_layer['type'] == 'fact']['r']
        r_hallu = df_layer[df_layer['type'] == 'hallu']['r']

        # Paired T-test
        if len(r_fact) > 1 and len(r_hallu) > 1:
            t_stat, p_val = ttest_rel(r_fact, r_hallu)
        else:
            t_stat, p_val = 0, 1.0

        # Mean difference (higher = better separation)
        diff = r_fact.mean() - r_hallu.mean()

        model_sweep_results.append({
            'model': model_key,
            'model_name': model_config['name'],
            'layer': layer,
            'r_fact_mean': r_fact.mean(),
            'r_hallu_mean': r_hallu.mean(),
            'diff': diff,
            'abs_diff': abs(diff),
            'p_value': p_val,
            't_statistic': t_stat,
            'significant': p_val < 0.05
        })

        # Clear GPU cache periodically
        if (layer + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # Add to global result
    all_results.extend(model_sweep_results)

    # --- INDIVIDUAL MODEL ANALYSIS ---
    df_model_results = pd.DataFrame(model_sweep_results)

    # Save individual CSV
    df_model_results.to_csv(f'heimdall_sweep_{model_key}.csv', index=False)
    print(f"\n[OK] Results saved: heimdall_sweep_{model_key}.csv")

    # --- INDIVIDUAL MODEL REPORT ---
    print("\n" + "-"*70)
    print(f"REPORT: {model_config['name']}")
    print("-"*70)

    # Find best layer
    best_layer_row = df_model_results.sort_values('abs_diff', ascending=False).iloc[0]
    print(f"\nBEST LAYER: {int(best_layer_row['layer'])}")
    print(f"   - R-Score Difference: {best_layer_row['diff']:.4f}")
    print(f"   - R-Score Factual: {best_layer_row['r_fact_mean']:.4f}")
    print(f"   - R-Score Hallu: {best_layer_row['r_hallu_mean']:.4f}")
    print(f"   - p-value: {best_layer_row['p_value']:.4e}")
    print(f"   - Significant: {'YES' if best_layer_row['significant'] else 'NO'}")

    # Top 5 layers
    print(f"\nTOP 5 LAYERS:")
    top5 = df_model_results.nlargest(5, 'abs_diff')
    for idx, row in top5.iterrows():
        print(f"   {int(row['layer']):2d}. Layer {int(row['layer']):2d} | "
              f"D={row['diff']:+.4f} | p={row['p_value']:.2e}")

    # Descriptive statistics
    print(f"\nGENERAL STATISTICS:")
    print(f"   - Best difference: {df_model_results['diff'].max():.4f}")
    print(f"   - Worst difference: {df_model_results['diff'].min():.4f}")
    print(f"   - Mean difference: {df_model_results['diff'].mean():.4f}")
    print(f"   - Significant layers (p<0.05): {df_model_results['significant'].sum()}/{len(df_model_results)}")

    # --- INDIVIDUAL MODEL PLOT ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Detailed Analysis: {model_config["name"]}', fontsize=16, fontweight='bold')

    # Plot 1: Difference per layer
    axes[0, 0].plot(df_model_results['layer'], df_model_results['diff'],
                    marker='o', color='#3498db', linewidth=2)
    axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.3)
    axes[0, 0].scatter([best_layer_row['layer']], [best_layer_row['diff']],
                       color='red', s=200, zorder=5, label='Best layer')
    axes[0, 0].set_xlabel('Layer', fontsize=11)
    axes[0, 0].set_ylabel('R-Score Difference (Factual - Hallu)', fontsize=11)
    axes[0, 0].set_title('Separation by Layer', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: p-values
    axes[0, 1].plot(df_model_results['layer'], -np.log10(df_model_results['p_value'] + 1e-100),
                    marker='o', color='#e74c3c', linewidth=2)
    axes[0, 1].axhline(-np.log10(0.05), color='green', linestyle='--', alpha=0.5, label='p=0.05')
    axes[0, 1].axhline(-np.log10(0.01), color='orange', linestyle='--', alpha=0.5, label='p=0.01')
    axes[0, 1].set_xlabel('Layer', fontsize=11)
    axes[0, 1].set_ylabel('-log10(p-value)', fontsize=11)
    axes[0, 1].set_title('Statistical Significance', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: R-Scores Factual vs Hallu
    axes[1, 0].plot(df_model_results['layer'], df_model_results['r_fact_mean'],
                    marker='o', color='#2ecc71', linewidth=2, label='Factual')
    axes[1, 0].plot(df_model_results['layer'], df_model_results['r_hallu_mean'],
                    marker='s', color='#e67e22', linewidth=2, label='Hallucination')
    axes[1, 0].set_xlabel('Layer', fontsize=11)
    axes[1, 0].set_ylabel('Mean R-Score', fontsize=11)
    axes[1, 0].set_title('R-Scores by Response Type', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Distribution of differences
    axes[1, 1].hist(df_model_results['diff'], bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(best_layer_row['diff'], color='red', linestyle='--', linewidth=2,
                       label=f'Best: {best_layer_row["diff"]:.4f}')
    axes[1, 1].axvline(df_model_results['diff'].mean(), color='green', linestyle='--', linewidth=2,
                       label=f'Mean: {df_model_results["diff"].mean():.4f}')
    axes[1, 1].set_xlabel('R-Score Difference', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Separation Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'heimdall_sweep_{model_key}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Individual plot saved: heimdall_sweep_{model_key}_analysis.png")

    # --- SAVE REPORT TO TXT ---
    with open(f'heimdall_sweep_{model_key}_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"SWEEP REPORT: {model_config['name']}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model ID: {model_config['id']}\n")
        f.write(f"Layers tested: {len(LAYERS_TO_TEST)}\n")
        f.write(f"Samples per layer: {N_SAMPLES}\n\n")

        f.write("BEST LAYER\n")
        f.write("-"*70 + "\n")
        f.write(f"Layer: {int(best_layer_row['layer'])}\n")
        f.write(f"R-Score Difference: {best_layer_row['diff']:.4f}\n")
        f.write(f"p-value: {best_layer_row['p_value']:.4e}\n\n")

        f.write("COMPLETE LAYER RANKING\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Rank':<6}{'Layer':<10}{'Difference':<15}{'p-value':<15}{'Sig.':<10}\n")
        f.write("-"*70 + "\n")

        ranked = df_model_results.sort_values('abs_diff', ascending=False)
        for rank, (idx, row) in enumerate(ranked.iterrows(), 1):
            sig = "Y" if row['significant'] else "N"
            f.write(f"{rank:<6}{int(row['layer']):<10}{row['diff']:<15.4f}"
                   f"{row['p_value']:<15.2e}{sig:<10}\n")

    print(f"[OK] TXT report saved: heimdall_sweep_{model_key}_report.txt")
    print("-"*70 + "\n")

    # --- CLEAR MEMORY BEFORE NEXT MODEL ---
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[OK] Memory cleared for next model\n")

# =============================================================================
# GLOBAL ANALYSIS AND VISUALIZATIONS
# =============================================================================

print("\n" + "="*70)
print("GLOBAL ANALYSIS - ALL MODELS AND LAYERS")
print("="*70)

df_all = pd.DataFrame(all_results)

# Save complete results
df_all.to_csv('heimdall_full_model_sweep_results.csv', index=False)
print("\n[OK] Complete results saved: heimdall_full_model_sweep_results.csv")

# --- FIND BEST MODEL+LAYER COMBINATION ---
best_overall = df_all.sort_values('abs_diff', ascending=False).iloc[0]

print("\n" + "="*70)
print("BEST MODEL + LAYER COMBINATION")
print("="*70)
print(f"\nModel: {best_overall['model_name']}")
print(f"Layer: {int(best_overall['layer'])}")
print(f"R-Score Difference: {best_overall['diff']:.4f}")
print(f"p-value: {best_overall['p_value']:.4e}")
print(f"Significance: {'YES' if best_overall['significant'] else 'NO'}")

# --- BEST LAYERS PER MODEL ---
print("\n" + "="*70)
print("BEST LAYER PER MODEL")
print("="*70)

for model_key in MODELS_CONFIG.keys():
    df_model = df_all[df_all['model'] == model_key]
    best_layer = df_model.sort_values('abs_diff', ascending=False).iloc[0]
    print(f"\n{MODELS_CONFIG[model_key]['name']}:")
    print(f"  - Best layer: {int(best_layer['layer'])}")
    print(f"  - Difference: {best_layer['diff']:.4f}")
    print(f"  - p-value: {best_layer['p_value']:.4e}")

# --- VISUALIZATIONS ---
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot 1: Difference per layer for each model
for model_key in MODELS_CONFIG.keys():
    df_model = df_all[df_all['model'] == model_key]
    axes[0].plot(df_model['layer'], df_model['diff'],
                 marker='o', label=MODELS_CONFIG[model_key]['name'], linewidth=2)

axes[0].set_xlabel('Layer', fontsize=12)
axes[0].set_ylabel('R-Score Difference (Factual - Hallu)', fontsize=12)
axes[0].set_title('Separation by Layer in Each Model', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: p-value per layer
for model_key in MODELS_CONFIG.keys():
    df_model = df_all[df_all['model'] == model_key]
    axes[1].plot(df_model['layer'], -np.log10(df_model['p_value'] + 1e-100),
                 marker='o', label=MODELS_CONFIG[model_key]['name'], linewidth=2)

axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
axes[1].set_xlabel('Layer', fontsize=12)
axes[1].set_ylabel('-log10(p-value)', fontsize=12)
axes[1].set_title('Significance by Layer', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Heatmap of differences
pivot_data = df_all.pivot(index='layer', columns='model_name', values='diff')
sns.heatmap(pivot_data, cmap='RdYlGn', center=0, annot=False,
            cbar_kws={'label': 'R-Score Difference'}, ax=axes[2])
axes[2].set_title('Heatmap: Separation by Model and Layer', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Model', fontsize=12)
axes[2].set_ylabel('Layer', fontsize=12)

plt.tight_layout()
plt.savefig('heimdall_full_model_sweep.png', dpi=300, bbox_inches='tight')
print("[OK] Plot saved: heimdall_full_model_sweep.png")

# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "="*70)
print("BENCHMARK RECOMMENDATIONS")
print("="*70)

print(f"\nBEST OVERALL OPTION:")
print(f"   MODEL_ID = \"{MODELS_CONFIG[best_overall['model']]['id']}\"")
print(f"   LAYER_IDX = {int(best_overall['layer'])}")
print(f"   # Difference: {best_overall['diff']:.4f}, p={best_overall['p_value']:.4e}")

print(f"\nOPTIONS BY MODEL:")
for model_key in MODELS_CONFIG.keys():
    df_model = df_all[df_all['model'] == model_key]
    best = df_model.sort_values('abs_diff', ascending=False).iloc[0]
    print(f"\n{MODELS_CONFIG[model_key]['name']}:")
    print(f"   LAYER_IDX = {int(best['layer'])} (diff={best['diff']:.4f})")

print("\n" + "="*70)
print("[OK] FULL SWEEP COMPLETED!")
print("="*70)
print()
