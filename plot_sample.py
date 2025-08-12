#!/usr/bin/env python3
"""
Plot a single sample from the Rayleigh-Benard dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from rayleigh_benard_dataset import RayleighBenardDataset


def plot_rayleigh_benard_sample(sample, save_path="sample_plot.png"):
    """
    Plot a 4-panel view of buoyancy evolution.

    Args:
        sample: Dataset sample containing buoyancy trajectories
        save_path: Path to save the plot
    """
    # Extract metadata
    Nx = sample["grid_shape_x"]
    Nz = sample["grid_shape_z"]
    Lx = sample["domain_size_x"]
    Lz = sample["domain_size_z"]
    perturbation_scale = sample["perturbation_scale"]

    # Extract buoyancy data
    b_initial = sample["buoyancy_initial"].reshape(Nx, Nz)
    b_trajectory = sample["buoyancy_trajectory"]  # (n_times, Nx*Nz)
    time_coordinates = sample["time_coordinates"]

    # Select 4 time points: initial, two middle, and final
    n_times = len(time_coordinates)
    time_indices = [0, n_times // 3, 2 * n_times // 3, n_times - 1]
    time_labels = ["Initial", "Early", "Late", "Final"]

    # Create figure with 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Get global ranges for consistent color scaling
    b_min = min(b_initial.min(), b_trajectory.min())
    b_max = max(b_initial.max(), b_trajectory.max())

    for i, (time_idx, label) in enumerate(zip(time_indices, time_labels)):
        ax = axes[i]

        if time_idx == 0:
            # Use initial condition for first plot
            b_data = b_initial
        else:
            # Reshape trajectory data for other plots
            b_data = b_trajectory[time_idx].reshape(Nx, Nz)

        im = ax.imshow(
            b_data.T,  # Transpose to get correct orientation
            extent=[0, Lx, 0, Lz],
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=b_min,
            vmax=b_max,
        )
        ax.set_title(f"{label} Buoyancy\n(t={time_coordinates[time_idx]:.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    # Add main title
    fig.suptitle(
        f"Rayleigh-Benard Convection Evolution\n"
        f"Perturbation Scale={perturbation_scale:.1e}",
        fontsize=16,
        fontweight="bold",
    )

    # Fix layout before adding colorbar
    plt.tight_layout()

    # Add colorbar with proper positioning after tight_layout
    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Buoyancy")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Sample visualization saved to {save_path}")
    print(f"Perturbation scale: {perturbation_scale:.1e}")
    print(f"Time steps saved: {len(time_coordinates)}")
    print(f"Grid size: {Nx}×{Nz}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create dataset with smaller parameters for faster testing
    dataset = RayleighBenardDataset(
        Nx=256,
        Nz=64,
        stop_sim_time=50,
        save_interval=1.0,
    )

    # Generate a single sample
    print("Generating Rayleigh-Benard sample...")
    sample = next(iter(dataset))

    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: {value}")

    # Plot the sample
    plot_rayleigh_benard_sample(sample)
