#!/usr/bin/env python3
"""
Create animated GIF from Rayleigh-Benard dataset sample.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image
from rayleigh_benard_dataset import RayleighBenardDataset


def create_buoyancy_gif(sample, save_path="buoyancy_evolution.gif", fps=4):
    """
    Create an animated GIF showing buoyancy evolution over time.

    Args:
        sample: Dataset sample containing buoyancy trajectory
        save_path: Path to save the GIF
        fps: Frames per second for animation
    """
    # Extract metadata
    Nx = sample["grid_shape_x"]
    Nz = sample["grid_shape_z"]
    Lx = sample["domain_size_x"]
    Lz = sample["domain_size_z"]
    perturbation_scale = sample["perturbation_scale"]

    # Extract buoyancy data
    b_initial = sample["buoyancy_initial"].reshape(Nx, Nz)
    b_trajectory = sample["buoyancy_trajectory"]
    time_coordinates = sample["time_coordinates"]

    # Get global range for consistent color scaling
    b_min = min(b_initial.min(), b_trajectory.min())
    b_max = max(b_initial.max(), b_trajectory.max())

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_paths = []

        # Generate frames
        for i in range(len(time_coordinates)):
            fig, ax = plt.subplots(figsize=(10, 6))

            if i == 0:
                b_data = b_initial
            else:
                b_data = b_trajectory[i].reshape(Nx, Nz)

            im = ax.imshow(
                b_data.T,
                extent=[0, Lx, 0, Lz],
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                vmin=b_min,
                vmax=b_max,
            )

            ax.set_title(
                f"Buoyancy at t={time_coordinates[i]:.2f} (Scale={perturbation_scale:.1e})",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("x")
            ax.set_ylabel("z")

            plt.colorbar(im, ax=ax, label="Buoyancy")

            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches="tight", facecolor="white")
            frame_paths.append(frame_path)
            plt.close()

        # Create GIF from frames
        images = []
        for frame_path in frame_paths:
            img = Image.open(frame_path)
            images.append(img)

        # Calculate duration per frame (in milliseconds)
        frame_duration = int(1000 / fps)

        # Save as GIF
        images[0].save(
            save_path,
            save_all=True,
            append_images=images[1:],
            duration=frame_duration,
            loop=0,  # Loop forever
        )

    print(f"Animated GIF saved to {save_path}")
    print(
        f"Animation duration: {len(time_coordinates) * frame_duration / 1000:.1f} seconds"
    )
    print(f"Frames per second: {fps}")

    return save_path


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create dataset with smaller parameters for faster testing
    dataset = RayleighBenardDataset(
        Nx=128,
        Nz=128,
        stop_sim_time=50,
        save_interval=0.5,  # Less frequent saves for animation
    )

    # Generate a single sample
    print("Generating Rayleigh-Benard sample for animation...")
    sample = next(iter(dataset))

    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: {value}")

    # Create animated GIF
    print("\nCreating animated GIF...")
    create_buoyancy_gif(sample, "buoyancy_evolution.gif", fps=25)

    print("\nAnimation creation complete!")
