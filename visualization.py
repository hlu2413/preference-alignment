"""
Visualization functions for FKC diffusion algorithm.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from base import baseline_score_function

def visualize_base_model_distribution():
    """
    Visualize the probability distribution of the base diffusion model.
    Shows where the base model assigns high vs low probability.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    
    # Create grid
    x_grid = jnp.linspace(0, 1, 200)
    y_grid = jnp.linspace(0, 1, 200)
    X, Y = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    
    # === LEFT PLOT: Base Model Distribution ===
    mean = jnp.array([0.5, 0.5])
    sigma = 0.15
    
    # Compute probability density
    distances_sq = jnp.sum((grid_points - mean[None, :])**2, axis=1)
    log_prob = -distances_sq / (2 * sigma**2)
    prob = jnp.exp(log_prob)
    prob = prob.reshape(X.shape)
    
    # Plot base model with better colormap
    im1 = ax1.contourf(X, Y, prob, levels=50, cmap='viridis', alpha=0.9)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Base Model Probability', shrink=0.8)
    cbar1.ax.tick_params(labelsize=10)
    
    # Mark the base model peak with a clean star
    ax1.scatter([0.5], [0.5], c='white', s=300, marker='*', 
               edgecolors='red', linewidths=3, 
               label='Base Model Peak', zorder=10)
    
    # Add subtle grid
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_title('Base Model Distribution\n(Gaussian at Center)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    
    # === RIGHT PLOT: Reward Function ===
    # Compute reward function
    reward_peaks = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    reward_values = jnp.zeros_like(X)
    
    for peak_x, peak_y in reward_peaks:
        dist = jnp.sqrt((X - peak_x)**2 + (Y - peak_y)**2)
        reward_values += jnp.exp(-dist**2 / 0.05)
    
    reward_values = jnp.clip(reward_values, 0, 1)
    
    # Plot reward function with better colormap
    im2 = ax2.contourf(X, Y, reward_values, levels=50, cmap='plasma', alpha=0.9)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Reward Function', shrink=0.8)
    cbar2.ax.tick_params(labelsize=10)
    
    # Mark reward peaks with clean circles
    for i, (px, py) in enumerate(reward_peaks):
        ax2.scatter([px], [py], c='white', s=200, marker='o', 
                   edgecolors='darkblue', linewidths=3, 
                   label=f'Reward Peak {i+1}' if i == 0 else "", zorder=10)
    
    # Add subtle grid
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_title('Reward Function\n(Four Peaks at Corners)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    
    # === THIRD PLOT: Three-Mode Reward Function ===
    # Compute three-mode reward function
    three_mode_peaks = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    three_mode_values = jnp.zeros_like(X)
    
    for peak_x, peak_y in three_mode_peaks:
        dist = jnp.sqrt((X - peak_x)**2 + (Y - peak_y)**2)
        three_mode_values += jnp.exp(-dist**2 / 0.03)
    
    three_mode_values = jnp.clip(three_mode_values, 0, 1)
    
    # Plot three-mode reward function
    im3 = ax3.contourf(X, Y, three_mode_values, levels=50, cmap='plasma', alpha=0.9)
    cbar3 = plt.colorbar(im3, ax=ax3, label='Three-Mode Reward', shrink=0.8)
    cbar3.ax.tick_params(labelsize=10)
    
    # Mark three-mode peaks with clean circles
    peak_labels = ['Lower Left', 'Center', 'Upper Right']
    colors = ['red', 'blue', 'green']
    for i, ((px, py), label, color) in enumerate(zip(three_mode_peaks, peak_labels, colors)):
        ax3.scatter([px], [py], c=color, s=200, marker='o', 
                   edgecolors='white', linewidths=3, 
                   label=f'{label} ({px}, {py})', zorder=10)
    
    # Add subtle grid
    ax3.grid(True, alpha=0.2, linestyle='--')
    ax3.set_title('Three-Mode Reward Function\n(Lower Left, Center, Upper Right)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    
    # Add main title
    fig.suptitle('Base Model vs Reward Distributions: Multiple Scenarios', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Print concise summary
    print("="*60)
    print("BASE MODEL vs REWARD DISTRIBUTIONS")
    print("="*60)
    print("Base Model: Gaussian at (0.5,0.5), σ=0.15")
    print("Four-Quadrant Reward: 4 peaks at corners (0.25,0.25), (0.75,0.25), (0.25,0.75), (0.75,0.75)")
    print("Three-Mode Reward: 3 peaks at (0.25,0.25), (0.5,0.5), (0.75,0.75)")
    print("="*60)


def visualize_step(step: int, particles: jnp.ndarray, weights: jnp.ndarray, 
                  observed_particles: jnp.ndarray, observed_rewards: jnp.ndarray,
                  success_rates: list, network: Dict, network_params: Dict,
                  reward_fn):
    """
    Visualize results for current step.
    
    Plot 1: CNN learned reward landscape + all generated particles (colored by weight) + selected particles (circled)
    Plot 2: True reward landscape + observed particles (colored by actual reward - continuous)
    Plot 3: Success rate evolution
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create grid for landscape visualization
    x_grid = jnp.linspace(0, 1, 100)
    y_grid = jnp.linspace(0, 1, 100)
    X, Y = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    
    # === PLOT 1: CNN Learned Reward + All Particles (colored by weight) + Selected Particles (circled) ===
    # Get CNN learned rewards for grid
    learned_rewards = network['forward'](network_params, grid_points)
    learned_rewards = learned_rewards.reshape(X.shape)
    
    # Plot learned reward landscape
    im1 = axes[0].contourf(X, Y, learned_rewards, levels=20, cmap='viridis', alpha=0.4)
    plt.colorbar(im1, ax=axes[0], label='Learned Reward')
    
    # Plot all generated particles colored by their weights
    valid_particles = ~jnp.isnan(particles).any(axis=1)
    valid_weights = weights[valid_particles]
    valid_particles_coords = particles[valid_particles]
    
    if len(valid_particles_coords) > 0:
        # Use a colormap to show weight distribution
        scatter_all = axes[0].scatter(
            valid_particles_coords[:, 0], 
            valid_particles_coords[:, 1],
            c=valid_weights,  # Color by weight
            cmap='plasma',  # Different colormap from landscape
            s=30, 
            alpha=0.6,
            vmin=jnp.min(valid_weights),
            vmax=jnp.max(valid_weights),
            edgecolors='none',
            label=f'All particles (n={len(valid_particles_coords)})'
        )
        # Add colorbar for particle weights
        cbar_weights = plt.colorbar(scatter_all, ax=axes[0], label='Particle Weight', 
                                    orientation='horizontal', pad=0.05, aspect=30)
    
    # Circle the selected/observed particles with thick black edges
    if len(observed_particles) > 0:
        axes[0].scatter(
            observed_particles[:, 0], 
            observed_particles[:, 1],
            s=150,  # Larger size
            facecolors='none',  # Transparent fill
            edgecolors='red',  # Red edge
            linewidths=2.5,  # Thick edge
            marker='o',
            label=f'Selected (n={len(observed_particles)})'
        )
    
    axes[0].set_title(f'Step {step}: CNN Learned Reward\n+ All Particles (colored by weight) + Selected (red circles)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # === PLOT 2: True Reward + Observed Particles (color by actual reward - continuous) ===
    # Get true rewards for grid
    true_rewards = reward_fn(grid_points)
    true_rewards = true_rewards.reshape(X.shape)
    
    # Plot true reward landscape
    im2 = axes[1].contourf(X, Y, true_rewards, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(im2, ax=axes[1], label='True Reward')
    
    # Plot observed particles colored by their ACTUAL REWARD VALUE (continuous)
    if len(observed_particles) > 0:
        scatter = axes[1].scatter(
            observed_particles[:, 0], 
            observed_particles[:, 1],
            c=observed_rewards,  # Color by actual reward value
            cmap='RdYlGn',  # Red (low) to Yellow to Green (high)
            s=100, 
            alpha=0.9, 
            marker='o', 
            edgecolors='black', 
            linewidths=1.5,
            vmin=0, 
            vmax=1,  # Reward range [0,1]
            label=f'Observed (n={len(observed_particles)})'
        )
        # Add colorbar for observed rewards
        cbar_scatter = plt.colorbar(scatter, ax=axes[1], label='Observed Reward Value', 
                                   orientation='horizontal', pad=0.1, aspect=30)
    
    axes[1].set_title(f'Step {step}: True Reward Landscape\n+ Observed Particles (Colored by Actual Reward)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    # === PLOT 3: Success Rate Evolution ===
    axes[2].plot(range(1, len(success_rates) + 1), success_rates, 'b-o', linewidth=2, markersize=6)
    axes[2].set_title('Success Rate Evolution')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Success Rate')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    axes[2].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (0.7)')
    axes[2].legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print key statistics
    if len(observed_particles) > 0:
        n_success = jnp.sum(observed_rewards > 0.7)
        avg_reward = jnp.mean(observed_rewards)
        print(f"Step {step}: {n_success}/{len(observed_particles)} success ({success_rates[-1]:.1%}) | Avg: {avg_reward:.3f}")


def visualize_gamma_and_diversity(gamma_history, all_weights, k_observe):
    """
    Visualize gamma schedule and weight distribution evolution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    steps = range(1, len(gamma_history) + 1)
    
    # Plot 1: Gamma schedule over time
    axes[0, 0].plot(steps, gamma_history, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Gamma Schedule (Diversity Coefficient)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Step', fontsize=12)
    axes[0, 0].set_ylabel('Gamma', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, max(gamma_history) * 1.1 if max(gamma_history) > 0 else 0.1)
    
    # Plot 2: Weight distribution evolution (box plot)
    weight_data = [w for w in all_weights[1:]]  # Skip cold start
    axes[0, 1].boxplot(weight_data, positions=steps)
    axes[0, 1].set_title('Weight Distribution Evolution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Step', fontsize=12)
    axes[0, 1].set_ylabel('Weight Value', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Mean and std of weights over time
    mean_weights = [jnp.mean(w) for w in all_weights[1:]]
    std_weights = [jnp.std(w) for w in all_weights[1:]]
    
    axes[1, 0].plot(steps, mean_weights, 'o-', color='blue', linewidth=2, markersize=6, label='Mean')
    axes[1, 0].fill_between(steps, 
                            np.array(mean_weights) - np.array(std_weights),
                            np.array(mean_weights) + np.array(std_weights),
                            alpha=0.3, color='blue', label='±1 Std')
    axes[1, 0].set_title('Weight Statistics', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Step', fontsize=12)
    axes[1, 0].set_ylabel('Weight Value', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: REMOVED (was selection threshold)
    # NEW: Weight range evolution (min/max)
    min_weights = [jnp.min(w) for w in all_weights[1:]]
    max_weights = [jnp.max(w) for w in all_weights[1:]]
    
    axes[1, 1].plot(steps, max_weights, 'o-', color='green', linewidth=2, markersize=6, label='Max Weight')
    axes[1, 1].plot(steps, min_weights, 'o-', color='red', linewidth=2, markersize=6, label='Min Weight')
    axes[1, 1].fill_between(steps, min_weights, max_weights, alpha=0.2, color='gray')
    axes[1, 1].set_title('Weight Range Evolution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Step', fontsize=12)
    axes[1, 1].set_ylabel('Weight Value', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print key summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Steps: {len(gamma_history)} | Gamma: {min(gamma_history):.3f}→{max(gamma_history):.3f}")
    print(f"Selection: top-{k_observe} particles | Final gamma: {gamma_history[-1]:.3f}")
    print("="*50)
