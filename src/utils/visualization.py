"""
Visualization Utilities
=======================

Plotting functions for ZoMBI-Hop optimization results,
specialized for simplex-constrained materials research.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.tri as tri
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_optimization_progress(
    history: List[Dict],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ZoMBI-Hop optimization progress over iterations.

    Parameters
    ----------
    history : List[Dict]
        Optimization history from ZoMBIHop checkpoints.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
    """
    _check_matplotlib()

    if len(history) == 0:
        print("No history to plot")
        return

    iterations = list(range(len(history)))
    best_values = [h.get('best_value', 0) for h in history]
    n_needles = [h.get('num_needles', 0) for h in history]
    n_points = [h.get('num_points_total', 0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Best value over time
    axes[0].plot(iterations, best_values, 'b-o', markersize=4, linewidth=1.5)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best Value')
    axes[0].set_title('Optimization Progress')
    axes[0].grid(True, alpha=0.3)

    # Number of needles found
    axes[1].plot(iterations, n_needles, 'g-o', markersize=4, linewidth=1.5)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Needles Found')
    axes[1].set_title('Local Optima Discovery')
    axes[1].grid(True, alpha=0.3)

    # Total points sampled
    axes[2].plot(iterations, n_points, 'r-o', markersize=4, linewidth=1.5)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Total Points')
    axes[2].set_title('Sample Efficiency')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_simplex_2d(
    X: torch.Tensor,
    Y: torch.Tensor,
    needles: Optional[torch.Tensor] = None,
    needle_vals: Optional[torch.Tensor] = None,
    component_names: Optional[List[str]] = None,
    title: str = "Simplex Optimization Results",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot optimization results on a 2D ternary diagram (for 3 components).

    Parameters
    ----------
    X : torch.Tensor
        Observed points, shape (n, 3).
    Y : torch.Tensor
        Observed values, shape (n,) or (n, 1).
    needles : torch.Tensor, optional
        Identified local optima, shape (m, 3).
    needle_vals : torch.Tensor, optional
        Values at needles, shape (m,) or (m, 1).
    component_names : List[str], optional
        Names for the 3 components.
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
    """
    _check_matplotlib()

    X = X.cpu().numpy() if torch.is_tensor(X) else X
    Y = Y.cpu().numpy().flatten() if torch.is_tensor(Y) else Y.flatten()

    if X.shape[1] != 3:
        raise ValueError("plot_simplex_2d requires 3 components (ternary diagram)")

    if component_names is None:
        component_names = ['Component A', 'Component B', 'Component C']

    # Convert to ternary coordinates
    def to_ternary(x):
        """Convert simplex coords to 2D ternary plot coords."""
        a, b, c = x[:, 0], x[:, 1], x[:, 2]
        x_coord = 0.5 * (2 * b + c) / (a + b + c)
        y_coord = (np.sqrt(3) / 2) * c / (a + b + c)
        return x_coord, y_coord

    fig, ax = plt.subplots(figsize=figsize)

    # Draw triangle outline
    triangle = plt.Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(triangle)

    # Plot observed points
    x_coords, y_coords = to_ternary(X)
    scatter = ax.scatter(x_coords, y_coords, c=Y, cmap='viridis',
                        s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label='Objective Value')

    # Plot needles if provided
    if needles is not None:
        needles_np = needles.cpu().numpy() if torch.is_tensor(needles) else needles
        needle_x, needle_y = to_ternary(needles_np)
        ax.scatter(needle_x, needle_y, c='red', s=200, marker='*',
                  edgecolors='black', linewidths=1, label='Local Optima', zorder=5)

    # Add component labels
    ax.text(0, -0.05, component_names[0], ha='center', fontsize=12)
    ax.text(1, -0.05, component_names[1], ha='center', fontsize=12)
    ax.text(0.5, np.sqrt(3)/2 + 0.05, component_names[2], ha='center', fontsize=12)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')

    if needles is not None:
        ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_simplex_3d(
    X: torch.Tensor,
    Y: torch.Tensor,
    needles: Optional[torch.Tensor] = None,
    component_names: Optional[List[str]] = None,
    title: str = "4-Component Simplex",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot optimization results on a 3D tetrahedron (for 4 components).

    Parameters
    ----------
    X : torch.Tensor
        Observed points, shape (n, 4).
    Y : torch.Tensor
        Observed values, shape (n,) or (n, 1).
    needles : torch.Tensor, optional
        Identified local optima, shape (m, 4).
    component_names : List[str], optional
        Names for the 4 components.
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
    """
    _check_matplotlib()

    X = X.cpu().numpy() if torch.is_tensor(X) else X
    Y = Y.cpu().numpy().flatten() if torch.is_tensor(Y) else Y.flatten()

    if X.shape[1] != 4:
        raise ValueError("plot_simplex_3d requires 4 components (tetrahedron)")

    if component_names is None:
        component_names = ['A', 'B', 'C', 'D']

    # Tetrahedron vertices in 3D
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
    ])

    # Convert simplex coords to 3D
    def to_3d(x):
        return x @ vertices

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot tetrahedron edges
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for i, j in edges:
        ax.plot3D(*zip(vertices[i], vertices[j]), 'k-', alpha=0.3)

    # Plot observed points
    points_3d = to_3d(X)
    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                        c=Y, cmap='viridis', s=40, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Objective Value', shrink=0.6)

    # Plot needles
    if needles is not None:
        needles_np = needles.cpu().numpy() if torch.is_tensor(needles) else needles
        needles_3d = to_3d(needles_np)
        ax.scatter(needles_3d[:, 0], needles_3d[:, 1], needles_3d[:, 2],
                  c='red', s=200, marker='*', label='Local Optima')

    # Add vertex labels
    for i, name in enumerate(component_names):
        ax.text(*vertices[i], name, fontsize=12)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if needles is not None:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_needles_summary(
    needles_results: List[Dict],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot summary of discovered local optima (needles).

    Parameters
    ----------
    needles_results : List[Dict]
        List of needle results from ZoMBIHop.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
    """
    _check_matplotlib()

    if len(needles_results) == 0:
        print("No needles to plot")
        return

    values = [n['value'] for n in needles_results]
    activations = [n['activation'] for n in needles_results]
    zooms = [n['zoom'] for n in needles_results]
    iterations = [n['iteration'] for n in needles_results]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Needle values
    indices = list(range(1, len(values) + 1))
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    bars = axes[0].bar(indices, values, color=colors, edgecolor='black')
    axes[0].set_xlabel('Needle Index')
    axes[0].set_ylabel('Objective Value')
    axes[0].set_title('Discovered Local Optima Values')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Discovery timeline
    discovery_order = list(range(len(needles_results)))
    axes[1].scatter(discovery_order, values, c=activations, cmap='tab10',
                   s=100, edgecolors='black', linewidths=1)
    axes[1].plot(discovery_order, values, 'k--', alpha=0.3)
    axes[1].set_xlabel('Discovery Order')
    axes[1].set_ylabel('Objective Value')
    axes[1].set_title('Discovery Timeline (colored by activation)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    # Print summary table
    print("\n" + "="*60)
    print("NEEDLES SUMMARY")
    print("="*60)
    print(f"{'#':<4} {'Value':<12} {'Activation':<12} {'Zoom':<8} {'Iteration':<10}")
    print("-"*60)
    for i, n in enumerate(needles_results):
        print(f"{i+1:<4} {n['value']:<12.4f} {n['activation']:<12} {n['zoom']:<8} {n['iteration']:<10}")
    print("="*60)
    print(f"Total needles: {len(needles_results)}")
    print(f"Best value: {max(values):.4f}")
    print(f"Mean value: {np.mean(values):.4f}")
    print("="*60)