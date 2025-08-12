#!/usr/bin/env python3
"""
Generate Rayleigh-Benard dataset and save to parquet files in chunks.
"""

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rayleigh_benard_dataset import RayleighBenardDataset


def generate_dataset_split(
    split_name="train", num_samples=1000, chunk_size=100, output_dir="data"
):
    """Generate a dataset split and save as chunked parquet files."""

    os.makedirs(output_dir, exist_ok=True)

    dataset = RayleighBenardDataset(
        Nx=256,
        Nz=64,
        stop_sim_time=30,
        save_interval=1.0,
    )
    num_chunks = (num_samples + chunk_size - 1) // chunk_size  # Ceiling division

    print(f"Generating {num_samples} {split_name} samples in {num_chunks} chunks...")

    dataset_iter = iter(dataset)
    chunk_data = None

    for i in range(num_samples):
        sample = next(dataset_iter)

        if chunk_data is None:
            # Initialize chunk data on first sample
            chunk_data = {key: [] for key in sample.keys()}

        # Add sample to current chunk
        for key, value in sample.items():
            chunk_data[key].append(value)

        # Save chunk when full or at end
        if (i + 1) % chunk_size == 0 or i == num_samples - 1:
            chunk_idx = i // chunk_size

            # Convert numpy arrays and scalars to PyArrow compatible format
            table_data = {}
            for key, values in chunk_data.items():
                # Handle both arrays and scalar values
                converted_values = []
                for val in values:
                    if isinstance(val, np.ndarray):
                        # Numpy array - convert to list
                        converted_values.append(val.tolist())
                    else:
                        # Scalar value - use directly
                        converted_values.append(val)
                table_data[key] = converted_values

            # Convert to PyArrow table
            table = pa.table(table_data)

            # Save chunk
            filename = f"{split_name}-{chunk_idx:05d}-of-{num_chunks:05d}.parquet"
            filepath = os.path.join(output_dir, filename)
            pq.write_table(table, filepath)

            print(f"Saved chunk {chunk_idx + 1}/{num_chunks}: {filepath}")

            # Reset for next chunk
            chunk_data = {key: [] for key in sample.keys()}

    print(f"Generated {num_samples} {split_name} samples")
    return num_samples


if __name__ == "__main__":
    np.random.seed(42)

    # Generate train split (smaller samples for 2D system)
    generate_dataset_split("train", num_samples=1000, chunk_size=5)

    # Generate test split
    generate_dataset_split("test", num_samples=200, chunk_size=5)
