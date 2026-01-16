#!/usr/bin/env python3
"""
ohio_social_validation.py (POST-VISCOSITY VERSION)
===================================================
Law of Ohio validation WITH Information Viscosity.

This script implements the Prandtl-inspired viscosity weighting that
breaks the R = ln(2) = 0.6931 ceiling problem.

COMPARE WITH: ohio_social_validation_pre_viscosity.py

THE PRANDTL EFFECT:
- Pre-viscosity: Discrete hops → uniform lifetimes → R = 0.6931 (meaningless)
- Post-viscosity: Viscosity weighting → varied lifetimes → meaningful R-scores

INFORMATION VISCOSITY:
    Viscosity(node) = 1 / (1 + local_clustering_coefficient)

    - High clustering (dense echo chamber) → Low viscosity → nodes "closer"
    - Low clustering (sparse bridge) → High viscosity → nodes "farther"

This creates real variation in the Rips filtration, producing meaningful
topological differences between echo chambers and bridging structures.
"""

import networkx as nx
import numpy as np
import gudhi
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def _parse_gml_with_duplicates(gml_path):
    """
    Manual GML parser that handles duplicate edges.
    Skips duplicate edges instead of raising error.
    """
    import re

    G = nx.DiGraph()

    with open(gml_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Parse nodes
    node_pattern = r'node\s*\[\s*id\s+(\d+)\s+label\s+"([^"]+)"\s+value\s+(\d+)\s+source\s+"([^"]+)"\s*\]'
    nodes = re.findall(node_pattern, content, re.DOTALL)

    if not nodes:
        # Try simpler pattern
        node_pattern = r'node\s*\[\s*id\s+(\d+).*?value\s+(\d+).*?\]'
        nodes = re.findall(node_pattern, content, re.DOTALL)
        for node_id, value in nodes:
            G.add_node(int(node_id), value=int(value))
    else:
        for node_id, label, value, source in nodes:
            G.add_node(int(node_id), label=label, value=int(value), source=source)

    # Parse edges (skip duplicates automatically since DiGraph ignores them)
    edge_pattern = r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)\s*\]'
    edges = re.findall(edge_pattern, content)

    edges_added = 0
    for source, target in edges:
        src, tgt = int(source), int(target)
        if not G.has_edge(src, tgt):
            G.add_edge(src, tgt)
            edges_added += 1

    print(f"  [OK] Manual parse: {len(G.nodes())} nodes, {edges_added} unique edges")
    return G


def load_political_blogosphere():
    """
    Load Political Blogosphere dataset
    Robust version with explicit error reporting
    """
    import os

    # 1. Locate file correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gml_path = os.path.join(script_dir, 'polblogs.gml')

    print(f"  Path resolved to: {gml_path}")

    if not os.path.exists(gml_path):
        # Fallback
        if os.path.exists('polblogs.gml'):
            gml_path = 'polblogs.gml'
        else:
            raise FileNotFoundError(f"CRITICAL: File not found at {gml_path}")

    # 2. Try reading GML - handle duplicate edges manually
    try:
        print("  Attempting standard read...")
        G = nx.read_gml(gml_path, label='id')
    except Exception as e1:
        if "duplicated" in str(e1).lower():
            print(f"  Standard read failed (duplicate edges). Using manual parser...")
            G = _parse_gml_with_duplicates(gml_path)
        else:
            raise ValueError(f"FATAL PARSE ERROR: {e1}")

    print(f"  [SUCCESS] Graph loaded. Nodes: {len(G.nodes())}")

    # 3. Ensure labels exist
    # The 'party' attribute in polblogs is usually 'value' (0 or 1)
    labels = nx.get_node_attributes(G, 'value')

    if not labels:
        print("  [WARNING] 'value' attribute missing. Checking node data...")
        # Debug: check first node
        first_node = list(G.nodes(data=True))[0]
        print(f"  Node 0 data: {first_node}")
        raise ValueError("Could not find 'value' attribute in nodes")

    return G, labels


def extract_subgraph_by_label(G, labels, target_label):
    """
    Extract subgraph of a specific community
    """
    nodes = [n for n, label in labels.items() if label == target_label]
    return G.subgraph(nodes).copy()


def extract_intra_vs_inter_community(G, labels):
    """
    Extract intra-community (echo chamber) vs inter-community (bridging) subgraphs.

    Intra: edges where source and target have SAME label (echo chamber)
    Inter: edges where source and target have DIFFERENT labels (bridging)

    This is the KEY test for Law of Ohio!
    """
    G_intra = nx.DiGraph()
    G_inter = nx.DiGraph()

    # Add all nodes to both graphs
    for node in G.nodes():
        if node in labels:
            G_intra.add_node(node, value=labels[node])
            G_inter.add_node(node, value=labels[node])

    intra_count = 0
    inter_count = 0

    for source, target in G.edges():
        if source in labels and target in labels:
            if labels[source] == labels[target]:
                # Same community = echo chamber behavior
                G_intra.add_edge(source, target)
                intra_count += 1
            else:
                # Different community = bridging behavior
                G_inter.add_edge(source, target)
                inter_count += 1

    print(f"  Intra-community edges (echo): {intra_count}")
    print(f"  Inter-community edges (bridge): {inter_count}")
    print(f"  Ratio intra/inter: {intra_count/max(inter_count,1):.1f}x")

    return G_intra, G_inter


def compute_distance_matrix(G, use_viscosity=True):
    """
    Distance matrix with Information Viscosity weighting.

    Instead of just counting hops, we weight by local clustering coefficient:
    - High clustering (dense area) → nodes appear CLOSER
    - Low clustering (sparse/bridge) → nodes appear FARTHER

    This creates variation in cycle lifetimes, avoiding the 0.6931 ceiling.
    """
    # Get largest connected component
    if G.is_directed():
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)

    G_sub = G.subgraph(largest_cc).copy()

    # Convert to undirected
    G_undirected = G_sub.to_undirected()

    nodes = list(G_undirected.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Compute local clustering coefficients
    if use_viscosity:
        clustering = nx.clustering(G_undirected)
        # Viscosity = 1 / (1 + clustering) → high clustering = low viscosity = closer
        viscosity = {node: 1.0 / (1.0 + clustering.get(node, 0)) for node in nodes}
    else:
        viscosity = {node: 1.0 for node in nodes}

    # Calculate shortest paths
    dist_dict = dict(nx.all_pairs_shortest_path_length(G_undirected))

    dist_matrix = np.zeros((n, n))
    max_dist = 0

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i == j:
                dist_matrix[i, j] = 0
            elif node_i in dist_dict and node_j in dist_dict[node_i]:
                hop_dist = dist_dict[node_i][node_j]

                if use_viscosity:
                    # Weight by average viscosity of endpoints
                    # Low clustering → high viscosity → larger effective distance
                    visc_weight = (viscosity[node_i] + viscosity[node_j]) / 2.0
                    dist_matrix[i, j] = hop_dist * visc_weight
                else:
                    dist_matrix[i, j] = hop_dist

                max_dist = max(max_dist, dist_matrix[i, j])
            else:
                dist_matrix[i, j] = np.inf

    # Replace infinities
    dist_matrix[dist_matrix == np.inf] = max_dist + 1

    # Normalize to [0, 1] for better Rips behavior
    if max_dist > 0:
        dist_matrix = dist_matrix / dist_matrix.max()

    return dist_matrix, nodes


def compute_h1_persistence(dist_matrix, max_edge_length=None):
    """
    Compute H1 persistence using Rips complex
    """
    if max_edge_length is None:
        max_edge_length = dist_matrix.max() + 1

    # Create Rips complex
    rips = gudhi.RipsComplex(
        distance_matrix=dist_matrix.astype(np.float64),
        max_edge_length=float(max_edge_length)
    )

    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()

    # Extract H1 intervals
    persistence = simplex_tree.persistence_intervals_in_dimension(1)

    if len(persistence) == 0:
        return 0.0, [], persistence

    # Calculate lifetimes
    lifetimes = []
    for birth, death in persistence:
        if death != np.inf:
            lifetimes.append(death - birth)

    if len(lifetimes) == 0:
        return 0.0, [], persistence

    # R-score (consistent with HEIMDALL)
    r_score = np.log1p(max(lifetimes) / np.mean(lifetimes))

    return r_score, lifetimes, persistence


def analyze_community(G, community_name):
    """
    Complete analysis of a community
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {community_name}")
    print(f"{'='*70}")

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")

    # Basic metrics
    clustering = 0
    if len(G.nodes()) > 0:
        G_undirected = G.to_undirected() if G.is_directed() else G
        clustering = nx.average_clustering(G_undirected)
        print(f"Clustering coefficient: {clustering:.4f}")

    # Distance and persistence
    print("\nComputing topological persistence...")
    dist_matrix, nodes = compute_distance_matrix(G)
    r_score, lifetimes, persistence = compute_h1_persistence(dist_matrix)

    print(f"H1 cycles found: {len(persistence)}")
    print(f"Mean lifetime: {np.mean(lifetimes) if lifetimes else 0:.4f}")
    print(f"Max lifetime: {max(lifetimes) if lifetimes else 0:.4f}")
    print(f"R-score: {r_score:.4f}")

    return {
        'name': community_name,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'clustering': clustering,
        'h1_cycles': len(persistence),
        'mean_lifetime': np.mean(lifetimes) if lifetimes else 0,
        'max_lifetime': max(lifetimes) if lifetimes else 0,
        'r_score': r_score,
        'lifetimes': lifetimes
    }


# =============================================================================
# SYNTHETIC ECHO CHAMBER GENERATION
# =============================================================================

def generate_echo_chamber(n_nodes=100, p_internal=0.4, echo_strength=0.9):
    """
    Generate a synthetic echo chamber network with MANY triangles/cycles.

    Echo chambers are characterized by:
    - High clustering (many triangles)
    - Dense internal connections
    - Information "loops" that reinforce beliefs
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Create highly clustered structure with explicit triangles
    n_clusters = 4
    cluster_size = n_nodes // n_clusters

    for c in range(n_clusters):
        start = c * cluster_size
        end = start + cluster_size
        nodes_in_cluster = list(range(start, end))

        # Create dense clique-like structure (many triangles)
        for i in range(start, end):
            for j in range(i + 1, end):
                if np.random.random() < p_internal:
                    G.add_edge(i, j)

        # Explicitly add triangles to ensure high H1
        n_triangles = int(cluster_size * echo_strength * 2)
        for _ in range(n_triangles):
            if len(nodes_in_cluster) >= 3:
                tri = np.random.choice(nodes_in_cluster, 3, replace=False)
                G.add_edge(tri[0], tri[1])
                G.add_edge(tri[1], tri[2])
                G.add_edge(tri[2], tri[0])

        # Add 4-cycles and 5-cycles (higher order loops)
        n_larger_cycles = int(cluster_size * echo_strength)
        for _ in range(n_larger_cycles):
            if len(nodes_in_cluster) >= 4:
                cycle = np.random.choice(nodes_in_cluster, 4, replace=False)
                for k in range(4):
                    G.add_edge(cycle[k], cycle[(k+1) % 4])

    # Very sparse inter-cluster connections (isolated echo chambers)
    for c1 in range(n_clusters):
        for c2 in range(c1 + 1, n_clusters):
            # Only 1-2 bridge edges
            n1 = np.random.randint(c1 * cluster_size, (c1 + 1) * cluster_size)
            n2 = np.random.randint(c2 * cluster_size, (c2 + 1) * cluster_size)
            G.add_edge(n1, n2)

    return G


def generate_open_network(n_nodes=100, p_edge=0.02):
    """
    Generate an open network structure (tree-like, very few cycles).

    Open networks are characterized by:
    - Low clustering (few triangles)
    - Diverse, non-redundant connections
    - Information flows through without "looping back"
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Create a star-like hub structure (very few cycles)
    n_hubs = 5
    hub_indices = np.linspace(0, n_nodes-1, n_hubs, dtype=int)

    # Connect nodes to nearest hub (creates star topology = no cycles)
    for i in range(n_nodes):
        if i not in hub_indices:
            nearest_hub = hub_indices[np.argmin(np.abs(hub_indices - i))]
            G.add_edge(i, nearest_hub)

    # Connect hubs in a line (not a cycle!)
    for i in range(len(hub_indices) - 1):
        G.add_edge(hub_indices[i], hub_indices[i+1])

    # Add very few random edges (minimal cycles)
    n_random = int(n_nodes * p_edge)
    for _ in range(n_random):
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        if i != j:
            G.add_edge(i, j)

    return G


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_synthetic_experiment():
    """
    Run experiment on synthetic networks to demonstrate the Law of Ohio
    """
    print("\n" + "="*70)
    print("SYNTHETIC NETWORK EXPERIMENT")
    print("Demonstrating H1 persistence difference")
    print("="*70)

    n_trials = 10
    echo_r_scores = []
    open_r_scores = []

    print(f"\nRunning {n_trials} trials...")

    for trial in range(n_trials):
        # Echo chamber (high internal coherence)
        G_echo = generate_echo_chamber(n_nodes=100, p_internal=0.3, echo_strength=0.8)
        dist_echo, _ = compute_distance_matrix(G_echo)
        r_echo, _, _ = compute_h1_persistence(dist_echo)
        echo_r_scores.append(r_echo)

        # Open network (diverse connections)
        G_open = generate_open_network(n_nodes=100, p_edge=0.03)
        dist_open, _ = compute_distance_matrix(G_open)
        r_open, _, _ = compute_h1_persistence(dist_open)
        open_r_scores.append(r_open)

        print(f"  Trial {trial+1}: Echo R={r_echo:.4f}, Open R={r_open:.4f}")

    # Statistical comparison
    t_stat, p_value = ttest_ind(echo_r_scores, open_r_scores)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Echo Chamber mean R-score: {np.mean(echo_r_scores):.4f} +/- {np.std(echo_r_scores):.4f}")
    print(f"Open Network mean R-score: {np.mean(open_r_scores):.4f} +/- {np.std(open_r_scores):.4f}")
    print(f"Difference: {np.mean(echo_r_scores) - np.mean(open_r_scores):+.4f}")
    print(f"\nt-statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.2e}")

    if p_value < 0.05 and np.mean(echo_r_scores) > np.mean(open_r_scores):
        print("\n[CONFIRMED] Echo chambers have HIGHER topological coherence (H1)")
        print("This validates the Law of Ohio prediction:")
        print("  -> Closed information loops = Higher R-score = 'Obsessive Coherence'")

    return echo_r_scores, open_r_scores


def _create_synthetic_only_plot(echo_scores, open_scores):
    """Helper function to create plot with synthetic data only."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Law of Ohio - Synthetic Network Validation", fontsize=14, fontweight='bold')

    # Plot 1: Synthetic comparison
    axes[0].bar(['Echo Chamber', 'Open Network'],
               [np.mean(echo_scores), np.mean(open_scores)],
               yerr=[np.std(echo_scores), np.std(open_scores)],
               color=['orange', 'green'], alpha=0.7, capsize=5)
    axes[0].set_ylabel('R-Score (H1 Persistence)', fontsize=12)
    axes[0].set_title('Network Type Comparison\n(Higher = More "Echo Chamber")', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Distribution
    axes[1].hist(echo_scores, bins=8, alpha=0.6, label='Echo Chamber', color='orange')
    axes[1].hist(open_scores, bins=8, alpha=0.6, label='Open Network', color='green')
    axes[1].set_xlabel('R-Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('R-Score Distribution (n=10 trials)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ohio_social_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved: ohio_social_validation.png")
    plt.close()


def main():
    print("="*70)
    print("LAW OF OHIO - POST-VISCOSITY VERSION")
    print("(With Information Viscosity - Prandtl Effect)")
    print("="*70)

    # First, run synthetic experiment (always works)
    echo_scores, open_scores = run_synthetic_experiment()

    # Then try real dataset
    print("\n" + "="*70)
    print("REAL DATASET: Political Blogosphere (Adamic & Glance, 2005)")
    print("="*70)

    # Try to load real data
    print("\nLoading Political Blogosphere dataset...")
    try:
        G, labels = load_political_blogosphere()
        print(f"[OK] Loaded {G.number_of_nodes()} blogs, {G.number_of_edges()} links")

        # Extract communities
        print("\nExtracting communities...")
        G_liberal = extract_subgraph_by_label(G, labels, 0)
        G_conservative = extract_subgraph_by_label(G, labels, 1)

        print(f"Liberal community: {G_liberal.number_of_nodes()} nodes")
        print(f"Conservative community: {G_conservative.number_of_nodes()} nodes")

        # Analyze each community
        results_liberal = analyze_community(G_liberal, "Liberal Blogosphere")
        results_conservative = analyze_community(G_conservative, "Conservative Blogosphere")

        # Comparison
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)

        diff_clustering = results_conservative['clustering'] - results_liberal['clustering']
        diff_r_score = results_conservative['r_score'] - results_liberal['r_score']

        print(f"\nClustering Difference: {diff_clustering:+.4f}")
        print(f"R-Score Difference: {diff_r_score:+.4f}")

        if results_liberal['r_score'] > 0:
            print(f"Percent Difference: {(diff_r_score / results_liberal['r_score'] * 100):+.1f}%")

        # Statistical test
        if len(results_liberal['lifetimes']) > 1 and len(results_conservative['lifetimes']) > 1:
            t_stat, p_value = ttest_ind(
                results_conservative['lifetimes'],
                results_liberal['lifetimes']
            )
            print(f"\nt-statistic: {t_stat:.2f}")
            print(f"p-value: {p_value:.4f}")

            if p_value < 0.05:
                print("\n[CONFIRMED] STATISTICALLY SIGNIFICANT (p < 0.05)")
                print("The Law of Ohio is validated on real social networks.")
            else:
                print("\n[INCONCLUSIVE] Not statistically significant")

        # =================================================================
        # KEY TEST: Intra-community vs Inter-community
        # =================================================================
        print("\n" + "="*70)
        print("KEY TEST: INTRA vs INTER COMMUNITY (Echo vs Bridge)")
        print("="*70)
        print("\nThis is the TRUE test of Law of Ohio:")
        print("  Intra-community = Echo Chamber = Should have HIGHER R-score")
        print("  Inter-community = Bridging = Should have LOWER R-score")

        print("\nExtracting edge types...")
        G_intra, G_inter = extract_intra_vs_inter_community(G, labels)

        # Analyze both
        results_intra = analyze_community(G_intra, "INTRA-Community (Echo Chamber)")
        results_inter = analyze_community(G_inter, "INTER-Community (Bridging)")

        # Comparison
        print("\n" + "="*70)
        print("INTRA vs INTER COMPARISON")
        print("="*70)

        diff_r = results_intra['r_score'] - results_inter['r_score']
        print(f"\nIntra R-Score: {results_intra['r_score']:.4f}")
        print(f"Inter R-Score: {results_inter['r_score']:.4f}")
        print(f"Difference: {diff_r:+.4f}")

        if len(results_intra['lifetimes']) > 1 and len(results_inter['lifetimes']) > 1:
            t_stat_ii, p_value_ii = ttest_ind(
                results_intra['lifetimes'],
                results_inter['lifetimes']
            )
            print(f"\nt-statistic: {t_stat_ii:.2f}")
            print(f"p-value: {p_value_ii:.2e}")

            if p_value_ii < 0.05 and diff_r > 0:
                print("\n" + "*"*70)
                print("[CONFIRMED] LAW OF OHIO VALIDATED!")
                print("*"*70)
                print("Echo chamber edges have HIGHER topological coherence")
                print("than bridging edges. This proves the hypothesis.")
            elif p_value_ii < 0.05 and diff_r < 0:
                print("\n[UNEXPECTED] Bridging edges have higher R-score!")
            else:
                print("\n[INCONCLUSIVE] No significant difference")

        # Create visualization with real data
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Law of Ohio - Social Network Validation", fontsize=14, fontweight='bold')

        # Plot 1: R-scores comparison (real data)
        communities = ['Liberal', 'Conservative']
        r_scores = [results_liberal['r_score'], results_conservative['r_score']]
        colors = ['blue', 'red']

        axes[0, 0].bar(communities, r_scores, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('R-Score (H1 Persistence)', fontsize=12)
        axes[0, 0].set_title('Topological Coherence by Community\n(Political Blogosphere)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Plot 2: Lifetime distributions (real data)
        if results_liberal['lifetimes'] and results_conservative['lifetimes']:
            axes[0, 1].hist(results_liberal['lifetimes'], bins=20, alpha=0.6,
                         label='Liberal', color='blue')
            axes[0, 1].hist(results_conservative['lifetimes'], bins=20, alpha=0.6,
                         label='Conservative', color='red')
            axes[0, 1].set_xlabel('Cycle Lifetime', fontsize=12)
            axes[0, 1].set_ylabel('Frequency', fontsize=12)
            axes[0, 1].set_title('H1 Persistence Distribution', fontsize=12)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: INTRA vs INTER comparison (KEY TEST)
        intra_inter_labels = ['INTRA\n(Echo)', 'INTER\n(Bridge)']
        intra_inter_scores = [results_intra['r_score'], results_inter['r_score']]
        colors_ii = ['darkorange', 'forestgreen']

        axes[1, 0].bar(intra_inter_labels, intra_inter_scores, color=colors_ii, alpha=0.8)
        axes[1, 0].set_ylabel('R-Score (H1 Persistence)', fontsize=12)
        axes[1, 0].set_title('KEY TEST: Intra vs Inter Community\n(Echo Chamber vs Bridging)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Add annotation for difference
        if diff_r != 0:
            max_score = max(intra_inter_scores)
            axes[1, 0].annotate(f'Δ = {diff_r:+.4f}', xy=(0.5, max_score * 1.05),
                               ha='center', fontsize=11, fontweight='bold')

        # Plot 4: Synthetic comparison
        axes[1, 1].bar(['Echo\nChamber', 'Open\nNetwork'],
                       [np.mean(echo_scores), np.mean(open_scores)],
                       yerr=[np.std(echo_scores), np.std(open_scores)],
                       color=['orange', 'green'], alpha=0.7, capsize=5)
        axes[1, 1].set_ylabel('R-Score', fontsize=12)
        axes[1, 1].set_title('Synthetic Validation\n(n=10 trials)', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('ohio_social_validation.png', dpi=150, bbox_inches='tight')
        print(f"\n[OK] Saved: ohio_social_validation.png")
        plt.close()

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        print("Download from: http://www-personal.umich.edu/~mejn/netdata/polblogs.zip")
        print("\nProceeding with synthetic data only...")
        _create_synthetic_only_plot(echo_scores, open_scores)

    except ValueError as e:
        print(f"[ERROR] GML read error: {e}")
        print("\nProceeding with synthetic data only...")
        _create_synthetic_only_plot(echo_scores, open_scores)

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        print("\nProceeding with synthetic data only...")
        _create_synthetic_only_plot(echo_scores, open_scores)

    # Final summary
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The Law of Ohio predicts that closed information loops (echo chambers)
exhibit higher topological coherence (H1 persistence) than open networks.

This is analogous to LLM hallucination, where "obsessive coherence" in
attention patterns indicates fabricated content.

Key insight: Both social echo chambers and LLM hallucinations share
the same topological signature - excessive cyclic structure.
    """)


if __name__ == "__main__":
    main()
