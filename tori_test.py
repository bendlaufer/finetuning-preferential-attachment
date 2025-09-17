import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import random
from datetime import datetime
from collections import defaultdict, Counter
import json

def parse_date(date_str):
    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))

def analyze_empirical_attachment_probs(G, sample_size=None):
    """
    P(attach to degree k) = (times degree k was chosen) / (times degree k was available)
    """
        
    node_dates = {}
    for node, attrs in G.nodes.data():
        if 'createdAt' in attrs:
            try:
                node_dates[node] = parse_date(attrs['createdAt'])
            except:
                continue
        
    edges_with_dates = []
    for u, v in G.edges():
        if u in node_dates and v in node_dates:
            u_date = node_dates[u]
            v_date = node_dates[v]
            
            if u_date != v_date: 
                # (attachment_time, newer_node, older_node)
                if u_date < v_date:
                    edges_with_dates.append((v_date, v, u))  # v attached to u
                else:
                    edges_with_dates.append((u_date, u, v))  # u attached to v
    
    edges_with_dates.sort(key=lambda x: x[0])
    
    if sample_size and len(edges_with_dates) > sample_size:
        sampled_indices = sorted(random.sample(range(len(edges_with_dates)), sample_size))
        edges_with_dates = [edges_with_dates[i] for i in sampled_indices]
    
    degree_opportunities = defaultdict(int)  # Times each degree was available
    degree_chosen = defaultdict(int)         # Times each degree was actually chosen
    
    attachment_events = []
    
    for i, (attachment_time, newer_node, target_node) in enumerate(edges_with_dates):
        # Find all nodes that existed before this attachment time
        available_nodes = []
        for node, creation_date in node_dates.items():
            if creation_date < attachment_time and node != newer_node:
                available_nodes.append(node)
        
        if len(available_nodes) < 2:  
            continue
            
        # Calculate degree of each available node at attachment time
        node_degrees_at_time = {}
        
        for node in available_nodes:
            degree_at_time = 0
            all_neighbors = set(G.predecessors(node)) | set(G.successors(node))
            
            for neighbor in all_neighbors:
                if neighbor in node_dates and node_dates[neighbor] < attachment_time:
                    degree_at_time += 1
            
            node_degrees_at_time[node] = degree_at_time
        
        for node, degree in node_degrees_at_time.items():
            degree_opportunities[degree] += 1
        
        if target_node in node_degrees_at_time:
            target_degree = node_degrees_at_time[target_node]
            degree_chosen[target_degree] += 1
            
            attachment_events.append({
                'target_node': target_node,
                'newer_node': newer_node,
                'target_degree': target_degree,
                'attachment_time': attachment_time,
                'num_choices_available': len(available_nodes),
                'available_degrees': list(node_degrees_at_time.values()),
                'mean_available_degree': np.mean(list(node_degrees_at_time.values())),
                'max_available_degree': max(node_degrees_at_time.values()) if node_degrees_at_time else 0
            })
    
    empirical_probs = {}
    degree_counts = {}
    
    for degree in degree_opportunities.keys():
        opportunities = degree_opportunities[degree]
        chosen = degree_chosen[degree]
        
        if opportunities > 0:
            empirical_probs[degree] = chosen / opportunities
            degree_counts[degree] = chosen
        else:
            empirical_probs[degree] = 0
            degree_counts[degree] = 0
    
    return attachment_events, empirical_probs, degree_opportunities, degree_chosen

def plot_empirical_attachment_analysis(attachment_events, empirical_probs, degree_opportunities, degree_chosen, min_opportunities=5):
    
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
    
    # 1: Empirical attachment probability vs degree
    axes[0].errorbar(filtered_degrees, filtered_probs, yerr=conf_intervals, 
                       fmt='o-', capsize=5, alpha=0.8, linewidth=2, markersize=6)
    axes[0].set_xlabel('Degree at Time of Attachment')
    axes[0].set_ylabel('Empirical Attachment Probability')
    axes[0].set_title('Empirical Attachment Probability vs Degree')
    axes[0].grid(True, alpha=0.3)
    
    if filtered_degrees:
        # Linear preferential attachment: P ∝ (degree + 1)
        theoretical_linear = np.array(filtered_degrees) + 1
        theoretical_linear = theoretical_linear / np.sum(theoretical_linear) * len(filtered_degrees) * np.mean(filtered_probs)
        axes[0].plot(filtered_degrees, theoretical_linear, 'r--', 
                      label='Linear PA (∝ degree + 1)', alpha=0.7, linewidth=2)
        
        # Uniform attachment: P = constant
        uniform_prob = np.mean(filtered_probs)
        axes[0].axhline(y=uniform_prob, color='g', linestyle='--', 
                         label=f'Uniform (P = {uniform_prob:.4f})', alpha=0.7, linewidth=2)
        
        axes[0].legend()
    
    # 2: Log-log plot
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

if __name__ == "__main__":
    G = pickle.load(open('data/ai_ecosystem_graph_finetune.pkl', 'rb'))
    
    sample_size = 1000  
    attachment_events, empirical_probs, degree_opportunities, degree_chosen = analyze_empirical_attachment_probs(G, sample_size=sample_size)
    
    if attachment_events:
        degrees, probs, opportunities = plot_empirical_attachment_analysis(
            attachment_events, empirical_probs, degree_opportunities, degree_chosen, min_opportunities=5)
        
        # results = {
        #     'attachment_events': len(attachment_events),
        #     'empirical_probabilities': empirical_probs,
        #     'degree_opportunities': dict(degree_opportunities),
        #     'degree_chosen': dict(degree_chosen)
        # }
        
        # with open("empirical_attachment_analysis.json", "w") as f:
        #     json.dump(results, f, indent=2)
        
    else:
        print("No attachment events found.")