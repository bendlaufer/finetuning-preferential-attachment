import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from datetime import datetime
from collections import defaultdict
import bisect
from multiprocessing import Pool, cpu_count
from functools import partial

def parse_date(date_str):
    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))

def parse_licenses(license_str):
    licenses = []
    license_str = license_str.strip('[]')
    for license_name in license_str.split(','):
        license_name = license_name.strip() 
        license_name = license_name.strip("'")
        if len(license_name) > 0:
            licenses.append(license_name) 
    return licenses

def precompute_graph_data(G):
    """
    Precompute adjacency with neighbor timestamps for faster degree calculation / lookups
    """
    node_dates = {}
    for node, attrs in G.nodes.data():
        if 'createdAt' in attrs:
            try:
                node_dates[node] = parse_date(attrs['createdAt'])
            except:
                continue
    
    adjacency_with_dates = {}
    for node in G.nodes():
        neighbors = set(G.predecessors(node)) | set(G.successors(node))
        adjacency_with_dates[node] = [
            (neighbor, node_dates.get(neighbor)) 
            for neighbor in neighbors 
            if neighbor in node_dates
        ]
    
    return node_dates, adjacency_with_dates

def compute_degree_at_time_fast(neighbor_date_pairs, attachment_time):
    return sum(1 for _, date in neighbor_date_pairs if date < attachment_time)

def process_edge_chunk(edge_chunk, node_array, date_array, adjacency_with_dates, negative_sample_ratio):
    """
    Process a chunk of edges in parallel.
    Returns: 
        - local degree opportunities 
        - degree chosen counts
    """
    local_opportunities = defaultdict(int)
    local_chosen = defaultdict(int)
    
    for attachment_time, newer_node, target_node in edge_chunk:
        # Binary search to find available nodes
        idx = bisect.bisect_left(date_array, attachment_time)
        mask = node_array[:idx] != newer_node
        available_nodes = node_array[:idx][mask]
        
        if len(available_nodes) < 2:
            continue
        
        # Compute target degree (positive sample)
        if target_node in adjacency_with_dates:
            target_degree = compute_degree_at_time_fast(
                adjacency_with_dates[target_node], attachment_time
            )
            local_chosen[target_degree] += 1
            local_opportunities[target_degree] += 1
        
        # Sample negatives
        non_target_mask = available_nodes != target_node
        non_target_nodes = available_nodes[non_target_mask]
        
        if len(non_target_nodes) == 0:
            continue
            
        num_negatives = min(negative_sample_ratio, len(non_target_nodes))
        negative_indices = np.random.choice(len(non_target_nodes), num_negatives, replace=False)
        
        for idx in negative_indices:
            neg_node = non_target_nodes[idx]
            if neg_node in adjacency_with_dates:
                neg_degree = compute_degree_at_time_fast(
                    adjacency_with_dates[neg_node], attachment_time
                )
                local_opportunities[neg_degree] += 1
    
    return dict(local_opportunities), dict(local_chosen)

def analyze_empirical_attachment_probs(G, sample_size=None, negative_sample_ratio=5, n_jobs=None):
    """
    Parallel version using multiprocessing for very large graphs.
    
    Args:
        G: NetworkX graph
        sample_size: Number of edges to sample (None = use all)
        negative_sample_ratio: Number of negative samples per positive
        n_jobs: Number of parallel workers (None = use all CPU cores)
    
    Returns:
        empirical_probs: Dictionary of degree -> attachment probability
        degree_opportunities: Dictionary of degree -> number of times available
        degree_chosen: Dictionary of degree -> number of times chosen
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # print(f"Using {n_jobs} parallel workers...")
    

    node_dates, adjacency_with_dates = precompute_graph_data(G)

    edges_with_dates = [
        (v_date, v, u) if u_date < v_date else (u_date, u, v)
        for u, v in G.edges()
        if (u_date := node_dates.get(u)) and (v_date := node_dates.get(v)) and u_date != v_date
    ]
    
    edges_with_dates.sort(key=lambda x: x[0])
    
    if sample_size and len(edges_with_dates) > sample_size:
        sampled_indices = np.random.choice(len(edges_with_dates), sample_size, replace=False)
        edges_with_dates = [edges_with_dates[i] for i in sorted(sampled_indices)]
    
    # print(f"Processing {len(edges_with_dates)} edges...")
    
    sorted_nodes = sorted(node_dates.items(), key=lambda x: x[1])
    node_array = np.array([n[0] for n in sorted_nodes], dtype=object)
    date_array = np.array([n[1] for n in sorted_nodes])
    

    chunk_size = max(1, len(edges_with_dates) // n_jobs)
    chunks = [edges_with_dates[i:i + chunk_size] for i in range(0, len(edges_with_dates), chunk_size)]
    
    print(f"Split into {len(chunks)} chunks of ~{chunk_size} edges each")
    
    process_func = partial(
        process_edge_chunk,
        node_array=node_array,
        date_array=date_array,
        adjacency_with_dates=adjacency_with_dates,
        negative_sample_ratio=negative_sample_ratio
    )
    
    with Pool(n_jobs) as pool:
        results = pool.map(process_func, chunks)
    
    # Merge results from all workers
    degree_opportunities = defaultdict(int)
    degree_chosen = defaultdict(int)
    
    for local_opp, local_chosen in results:
        for degree, count in local_opp.items():
            degree_opportunities[degree] += count
        for degree, count in local_chosen.items():
            degree_chosen[degree] += count
    
    empirical_probs = {
        degree: degree_chosen[degree] / degree_opportunities[degree]
        for degree in degree_opportunities.keys()
        if degree_opportunities[degree] > 0
    }
    
    print(f"Analyzed {len(degree_opportunities)} unique degree values")
    
    return empirical_probs, degree_opportunities, degree_chosen

def plot_empirical_attachment_analysis(empirical_probs, degree_opportunities, degree_chosen, min_opportunities=5):
    
    filtered_degrees = []
    filtered_probs = []
    filtered_opportunities = []
    filtered_chosen = []
    
    for degree in sorted(empirical_probs.keys()):
        if degree_opportunities[degree] >= min_opportunities:
            filtered_degrees.append(degree)
            filtered_probs.append(empirical_probs[degree])
            filtered_opportunities.append(degree_opportunities[degree])
            filtered_chosen.append(degree_chosen[degree])
  
    conf_intervals = []
    for i, degree in enumerate(filtered_degrees):
        n = filtered_opportunities[i]
        p = filtered_probs[i]
        
        if n > 0:
            std_err = np.sqrt(p * (1 - p) / n)
            margin = 1.96 * std_err  # 95% CI
            conf_intervals.append(margin)
        else:
            conf_intervals.append(0)
    

    fig, axes = plt.subplots(2, 1, figsize=(15, 6))
    
  
    axes[0].errorbar(filtered_degrees, filtered_probs, yerr=conf_intervals, 
                       fmt='o-', capsize=5, alpha=0.8, linewidth=2, markersize=6)
    axes[0].set_xlabel('Degree at Time of Attachment')
    axes[0].set_ylabel('Empirical Attachment Probability')
    axes[0].set_title('Empirical Attachment Probability vs Degree')
    axes[0].grid(True, alpha=0.3)
    
    if filtered_degrees:
        # Linear: P ∝ (degree + 1)
        theoretical_linear = np.array(filtered_degrees) + 1
        theoretical_linear = theoretical_linear / np.sum(theoretical_linear) * len(filtered_degrees) * np.mean(filtered_probs)
        axes[0].plot(filtered_degrees, theoretical_linear, 'r--', 
                      label='Linear PA (∝ degree + 1)', alpha=0.7, linewidth=2)
        
        # Uniform: P = constant
        uniform_prob = np.mean(filtered_probs)
        axes[0].axhline(y=uniform_prob, color='g', linestyle='--', 
                         label=f'Uniform (P = {uniform_prob:.4f})', alpha=0.7, linewidth=2)
        
        axes[0].legend()
    

    log_degrees = []
    log_probs = []
    
    for deg, prob in zip(filtered_degrees, filtered_probs):
        if deg > 0 and prob > 0:
            log_degrees.append(np.log(deg + 1))  # +1 to handle degree 0
            log_probs.append(np.log(prob))
    
    if len(log_degrees) > 1:
        axes[1].scatter(log_degrees, log_probs, s=50, alpha=0.8)
        
        # Fit line
        z = np.polyfit(log_degrees, log_probs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(log_degrees), max(log_degrees), 100)
        axes[1].plot(x_line, p(x_line), "r-", linewidth=2, 
                      label=f'Slope = {z[0]:.3f}')
        axes[1].legend()
        
        axes[1].set_xlabel('Log(Degree + 1)')
        axes[1].set_ylabel('Log(Empirical Attachment Probability)')
        axes[1].set_title(f'Log-Log Plot (Empirical PA Exponent ≈ {z[0]:.3f})')
        axes[1].grid(True, alpha=0.3)
        
        print(f"\nPower law analysis:")
        print(f"Log-log slope of empirical probabilities: {z[0]:.3f}")
        if abs(z[0] - 1.0) < 0.2:
            print("Strong evidence for linear preferential attachment")
        elif z[0] > 0.5:
            print("Some evidence for preferential attachment")

    
    plt.tight_layout()
    plt.savefig('empirical_attachment_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return filtered_degrees, filtered_probs, filtered_opportunities

# def plot_license_data(G):
#     license_children = defaultdict(list)
#     for node_name in G.nodes:
#         node = G.nodes[node_name]
#         licenses = parse_licenses(node['licenses'])
#         for license in licenses:
#             license_children[license].append(len(set(G.successors(node_name))))
    

#     avgs = defaultdict(int)
#     for license in license_children:
#         avg = sum(license_children[license]) / len(license_children[license])
#         if avg > 0.01:
#             avgs[license] = avg

#     fig, ax = plt.subplots(1,1,figsize=(25, 12))

#     licenses = avgs.keys() 
#     counts = avgs.values()
#     ax.bar(licenses, counts, label=licenses)
#     print(licenses)
#     ax.set_ylabel('Average # of Children')
#     ax.set_title('Avg. Counts by License')
#     # ax.legend(title='License Name')
#     ax.tick_params(axis='x', labelrotation=35)

#     plt.savefig('license_counts.png')



if __name__ == "__main__":
    """
    Example node: 

    moonshotai/Kimi-K2-Instruct: {'likes': 479, 'downloads': 13356, 'pipeline_tag': 'text-generation', 
        'library_name': 'transformers', 'createdAt': '2025-07-11T00:55:12.000Z', 'licenses': "['other']", 
        'datasets': '[]', 'languages': '[]'}

        
    Example edge:

    (moonshotai/Kimi-K2-Instruct, yujiepan/kimi-k2-tiny-random): {'edge_types': ['finetune'], 
        'edge_type': 'finetune', 'change_in_likes': -478, 'percentage_change_in_likes': -0.9979123173277662, 
        'change_in_downloads': -13356, 'percentage_change_in_downloads': -1.0, 'change_in_createdAt_days': 1}
    """
    
    G = pickle.load(open('data/ai_ecosystem_graph_finetune.pkl', 'rb'))
    
    # plot_license_data(G)
   
 
    # attachment_events, empirical_probs, degree_opportunities, degree_chosen = analyze_empirical_attachment_probs(G, sample_size=sample_size)
    # empirical_probs, degree_opportunities, degree_chosen = analyze_with_balanced_negative_sampling(G, sample_size=1000, negatives_per_degree=2)
    empirical_probs, degree_opportunities, degree_chosen = analyze_empirical_attachment_probs(G, sample_size=5000, negative_sample_ratio=20)
    plot_empirical_attachment_analysis(empirical_probs, degree_opportunities, degree_chosen, min_opportunities=11)
   