import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def visualize_weight_distribution(
    weights: torch.Tensor,
    step_dir: str,
    step_number: int,
    k_observe: Optional[int] = None
) -> None:
    """
    Visualize weight distributions for latent embeddings at a given step.
    
    Args:
        weights: Tensor of shape (n_particles,) containing weights for each particle
        step_dir: Directory path where visualizations should be saved
        step_number: Current step number for labeling
        k_observe: Number of particles selected (if None, not shown)
    """
    os.makedirs(step_dir, exist_ok=True)
    
    weights_np = weights.detach().cpu().numpy() if isinstance(weights, torch.Tensor) else weights
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Weight Distribution Analysis - Step {step_number}', fontsize=14, fontweight='bold')
    
    # 1. Histogram
    ax1 = axes[0, 0]
    ax1.hist(weights_np, bins=min(30, len(weights_np)), edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Weight Histogram')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(weights_np), color='red', linestyle='--', label=f'Mean: {np.mean(weights_np):.4f}')
    ax1.axvline(np.median(weights_np), color='green', linestyle='--', label=f'Median: {np.median(weights_np):.4f}')
    ax1.legend()
    
    # 2. Bar plot (sorted)
    ax2 = axes[0, 1]
    sorted_indices = np.argsort(weights_np)
    sorted_weights = weights_np[sorted_indices]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_weights)))
    ax2.bar(range(len(sorted_weights)), sorted_weights, color=colors)
    if k_observe is not None:
        ax2.axvline(len(sorted_weights) - k_observe, color='red', linestyle='--', 
                   label=f'Top {k_observe} selected', linewidth=2)
    ax2.set_xlabel('Particle Index (Sorted)')
    ax2.set_ylabel('Weight Value')
    ax2.set_title('Sorted Weight Values')
    ax2.grid(True, alpha=0.3)
    if k_observe is not None:
        ax2.legend()
    
    # 3. Box plot
    ax3 = axes[1, 0]
    bp = ax3.boxplot(weights_np, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax3.set_ylabel('Weight Value')
    ax3.set_title('Weight Box Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    Weight Statistics:
    
    Count: {len(weights_np)}
    Mean: {np.mean(weights_np):.6f}
    Median: {np.median(weights_np):.6f}
    Std: {np.std(weights_np):.6f}
    Min: {np.min(weights_np):.6f}
    Max: {np.max(weights_np):.6f}
    Range: {np.max(weights_np) - np.min(weights_np):.6f}
    
    Percentiles:
    25th: {np.percentile(weights_np, 25):.6f}
    75th: {np.percentile(weights_np, 75):.6f}
    90th: {np.percentile(weights_np, 90):.6f}
    95th: {np.percentile(weights_np, 95):.6f}
    """
    if k_observe is not None:
        top_k_threshold = np.partition(weights_np, -k_observe)[-k_observe]
        stats_text += f"\n    Top {k_observe} threshold: {top_k_threshold:.6f}"
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(step_dir, 'weight_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save raw weights as numpy array for later analysis
    weights_path = os.path.join(step_dir, 'weights.npy')
    np.save(weights_path, weights_np)

