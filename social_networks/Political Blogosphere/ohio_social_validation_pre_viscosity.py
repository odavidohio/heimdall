#!/usr/bin/env python3
"""
ohio_social_validation_pre_viscosity.py
=========================================
Law of Ohio validation WITHOUT Information Viscosity.

This script demonstrates the R-score ceiling problem (R = ln(2) = 0.6931)
that occurs when using discrete shortest-path distances.

COMPARE WITH: ohio_social_validation.py (post-viscosity version)

Together, these scripts demonstrate the Prandtl Effect:
- Pre-viscosity: Discrete hops → uniform cycle lifetimes → R ceiling
- Post-viscosity: Viscosity weighting → varied lifetimes → meaningful R-scores
"""

import networkx as nx
import numpy as np
import gudhi
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
import re
import os


# =============================================================================
# GML PARSER (handles duplicate edges)
# =============================================================================

def _parse_gml_with_duplicates(gml_path):
    """Manual GML parser that handles duplicate edges."""
    G = nx.DiGraph()
    with open(gml_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    node_pattern = r'node\s*\[\s*id\s+(\d+).*?value\s+(\d+).*?\]'
    nodes = re.findall(node_pattern, content, re.DOTALL)
    for node_id, value in nodes:
        G.add_node(int(node_id), value=int(value))

    edge_pattern = r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)\s*\]'
    edges = re.findall(edge_pattern, content)
    for source, target in edges:
        src, tgt = int(source), int(target)
        if not G.has_edge(src, tgt):
            G.add_edge(src, tgt)

    print(f"  [OK] Parsed: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G


def load_political_blogosphere():
    """Load Political Blogosphere dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gml_path = os.path.join(script_dir, 'polblogs.gml')

    print(f"  Path: {gml_path}")

    if not os.path.exists(gml_path):
        raise FileNotFoundError(f"File not found: {gml_path}")

    try:
        G = nx.read_gml(gml_path, label='id')
    except Exception:
        print("  Using manual parser (duplicate edges)...")
        G = _parse_gml_with_duplicates(gml_path)

    labels = nx.get_node_attributes(G, 'value')
    return G, labels


# =============================================================================
# PRE-VISCOSITY DISTANCE MATRIX (causes R = 0.6931 ceiling)
# =============================================================================

def compute_distance_matrix_discrete(G):
    """
    PRE-VISCOSITY VERSION: Pure discrete shortest-path distances.

    This causes the R-score ceiling problem because:
    - All distances are integers (1, 2, 3, ...)
    - All cycle lifetimes become uniform (death - birth = 1)
    - R = log1p(max/mean) = log1p(1/1) = log(2) = 0.6931

    This is the CONTROL condition for demonstrating the Prandtl Effect.
    """
    # Get largest connected component
    if G.is_directed():
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)

    G_sub = G.subgraph(largest_cc).copy().to_undirected()
    nodes = list(G_sub.nodes())
    n = len(nodes)

    # DISCRETE shortest paths (the problem!)
    dist_dict = dict(nx.all_pairs_shortest_path_length(G_sub))

    dist_matrix = np.zeros((n, n))
    max_dist = 0

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j and node_i in dist_dict and node_j in dist_dict[node_i]:
                dist_matrix[i, j] = dist_dict[node_i][node_j]  # INTEGER!
                max_dist = max(max_dist, dist_matrix[i, j])

    dist_matrix[dist_matrix == np.inf] = max_dist + 1

    # Normalize to [0, 1]
    if max_dist > 0:
        dist_matrix = dist_matrix / dist_matrix.max()

    return dist_matrix, nodes


# =============================================================================
# H1 PERSISTENCE
# =============================================================================

def compute_h1_persistence(dist_matrix):
    """Compute H1 persistence using Rips complex."""
    rips = gudhi.RipsComplex(
        distance_matrix=dist_matrix.astype(np.float64),
        max_edge_length=1.1
    )
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()

    persistence = simplex_tree.persistence_intervals_in_dimension(1)

    if len(persistence) == 0:
        return 0.0, [], []

    lifetimes = [death - birth for birth, death in persistence if death != np.inf]

    if not lifetimes:
        return 0.0, [], []

    # R-score (consistent with HEIMDALL)
    r_score = np.log1p(max(lifetimes) / np.mean(lifetimes))

    return r_score, lifetimes, persistence


# =============================================================================
# COMMUNITY EXTRACTION
# =============================================================================

def extract_subgraph_by_label(G, labels, target_label):
    """Extract subgraph of a specific community."""
    nodes = [n for n, label in labels.items() if label == target_label]
    return G.subgraph(nodes).copy()


def extract_intra_vs_inter_community(G, labels):
    """Extract intra-community (echo) vs inter-community (bridge) edges."""
    G_intra = nx.DiGraph()
    G_inter = nx.DiGraph()

    for node in G.nodes():
        if node in labels:
            G_intra.add_node(node, value=labels[node])
            G_inter.add_node(node, value=labels[node])

    intra_count = 0
    inter_count = 0

    for source, target in G.edges():
        if source in labels and target in labels:
            if labels[source] == labels[target]:
                G_intra.add_edge(source, target)
                intra_count += 1
            else:
                G_inter.add_edge(source, target)
                inter_count += 1

    print(f"  Intra-community edges (echo): {intra_count}")
    print(f"  Inter-community edges (bridge): {inter_count}")

    return G_intra, G_inter


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_community(G, name):
    """Analyze a community's topological properties."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    if G.number_of_nodes() < 3:
        print("[SKIP] Too few nodes")
        return {'name': name, 'r_score': 0, 'lifetimes': []}

    # Compute distance matrix (DISCRETE - no viscosity)
    print("\nComputing DISCRETE distance matrix (no viscosity)...")
    dist_matrix, nodes = compute_distance_matrix_discrete(G)

    # H1 persistence
    r_score, lifetimes, persistence = compute_h1_persistence(dist_matrix)

    print(f"H1 cycles: {len(persistence)}")
    print(f"Lifetimes - Mean: {np.mean(lifetimes) if lifetimes else 0:.4f}, "
          f"Max: {max(lifetimes) if lifetimes else 0:.4f}")
    print(f"R-score: {r_score:.4f}")

    if abs(r_score - 0.6931) < 0.01:
        print("[!] R-score = ln(2) ceiling detected (discrete distance problem)")

    return {
        'name': name,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'h1_cycles': len(persistence),
        'r_score': r_score,
        'lifetimes': lifetimes
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("LAW OF OHIO - PRE-VISCOSITY VERSION")
    print("(Demonstrating the R = ln(2) = 0.6931 ceiling problem)")
    print("="*70)

    print("\nLoading Political Blogosphere dataset...")
    try:
        G, labels = load_political_blogosphere()
        print(f"[OK] Loaded {G.number_of_nodes()} blogs, {G.number_of_edges()} links")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Extract intra vs inter
    print("\n" + "="*70)
    print("INTRA vs INTER COMMUNITY (Echo vs Bridge)")
    print("="*70)

    print("\nExtracting edge types...")
    G_intra, G_inter = extract_intra_vs_inter_community(G, labels)

    # Analyze both
    results_intra = analyze_community(G_intra, "INTRA-Community (Echo Chamber)")
    results_inter = analyze_community(G_inter, "INTER-Community (Bridging)")

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON (PRE-VISCOSITY)")
    print("="*70)

    diff_r = results_intra['r_score'] - results_inter['r_score']
    print(f"\nIntra R-Score: {results_intra['r_score']:.4f}")
    print(f"Inter R-Score: {results_inter['r_score']:.4f}")
    print(f"Difference: {diff_r:+.4f}")

    # Statistical test
    if len(results_intra['lifetimes']) > 1 and len(results_inter['lifetimes']) > 1:
        t_stat, p_value = ttest_ind(
            results_intra['lifetimes'],
            results_inter['lifetimes']
        )
        print(f"\nt-statistic: {t_stat:.2f}")
        print(f"p-value: {p_value:.2e}")
    else:
        p_value = 1.0

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PRE-VISCOSITY: R-score Ceiling Problem (R = ln(2) = 0.6931)",
                 fontsize=14, fontweight='bold')

    # Plot 1: R-scores
    labels_plot = ['INTRA\n(Echo)', 'INTER\n(Bridge)']
    r_scores = [results_intra['r_score'], results_inter['r_score']]

    bars = axes[0].bar(labels_plot, r_scores, color=['darkorange', 'forestgreen'], alpha=0.8)
    axes[0].axhline(y=0.6931, color='red', linestyle='--', linewidth=2, label='ln(2) ceiling')
    axes[0].set_ylabel('R-Score', fontsize=12)
    axes[0].set_title('R-Score Comparison\n(Both hit the ln(2) ceiling)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Lifetime distribution
    if results_intra['lifetimes']:
        axes[1].hist(results_intra['lifetimes'], bins=20, alpha=0.6,
                    label='Intra (Echo)', color='darkorange')
    if results_inter['lifetimes']:
        axes[1].hist(results_inter['lifetimes'], bins=20, alpha=0.6,
                    label='Inter (Bridge)', color='forestgreen')
    axes[1].set_xlabel('Cycle Lifetime', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('H1 Lifetime Distribution\n(All lifetimes are nearly identical)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ohio_pre_viscosity.png', dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved: ohio_pre_viscosity.png")
    plt.close()

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION (PRE-VISCOSITY)")
    print("="*70)
    print("""
WITHOUT Information Viscosity, the R-score is stuck at ln(2) = 0.6931.

This happens because:
1. Shortest-path distances are discrete integers (1, 2, 3, ...)
2. All H1 cycles have nearly identical lifetimes
3. R = log1p(max/mean) ≈ log1p(1) = ln(2)

The Prandtl Effect shows that we need VISCOSITY in the distance
metric to capture the true topological differences between
echo chambers and bridging structures.

Run ohio_social_validation.py (POST-viscosity) to see the fix.
    """)


if __name__ == "__main__":
    main()
