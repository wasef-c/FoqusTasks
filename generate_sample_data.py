import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import random
import time

# Import all functions from your existing generation code
from generate_dataset import *

def create_dataset(output_folder, num_samples=100, volume_shape=(64, 64, 64), 
                  slices_per_sample=3, undersample_factor=4, 
                  central_lines_zero_percent=0.05):
    """
    Generate and save a dataset of fully sampled and undersampled MRI slices.
    
    Parameters:
    - output_folder: Where to save the generated dataset
    - num_samples: Number of 3D structures to generate
    - volume_shape: Shape of the 3D volumes
    - slices_per_sample: Number of slices to extract per sample
    - undersample_factor: Keep 1 out of every N lines in k-space
    - central_lines_zero_percent: Percentage of central lines to set to zero
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating dataset with {num_samples} 3D samples ({num_samples * slices_per_sample} slices)...")
    start_time = time.time()
    
    # Generate the dataset using your existing function
    fully_sampled_data, undersampled_data = generate_dataset(
        num_samples=num_samples, 
        volume_shape=volume_shape,
        slices_per_sample=slices_per_sample,
        undersample_factor=undersample_factor,
        central_lines_zero_percent=central_lines_zero_percent
    )
    
    # Save the dataset
    fully_sampled_path = os.path.join(output_folder, "fully_sampled_data.npy")
    undersampled_path = os.path.join(output_folder, "undersampled_data.npy")
    
    np.save(fully_sampled_path, fully_sampled_data)
    np.save(undersampled_path, undersampled_data)
    
    # Save generation parameters
    with open(os.path.join(output_folder, "dataset_info.txt"), 'w') as f:
        f.write(f"Dataset Generation Parameters:\n")
        f.write(f"Number of 3D samples: {num_samples}\n")
        f.write(f"Volume shape: {volume_shape}\n")
        f.write(f"Slices per sample: {slices_per_sample}\n")
        f.write(f"Total slices: {len(fully_sampled_data)}\n")
        f.write(f"Undersample factor: {undersample_factor}\n")
        f.write(f"Central lines zero percent: {central_lines_zero_percent}\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create a few example visualizations
    create_example_visualizations(fully_sampled_data, undersampled_data, output_folder)
    
    elapsed_time = time.time() - start_time
    print(f"Dataset generation complete in {elapsed_time:.2f} seconds.")
    print(f"Generated {len(fully_sampled_data)} slices.")
    print(f"Dataset saved to {output_folder}")

def create_example_visualizations(fully_sampled_data, undersampled_data, output_folder):
    """Create and save visualizations of a few example slices."""
    vis_folder = os.path.join(output_folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)
    
    # Number of examples to visualize
    num_examples = min(5, len(fully_sampled_data))
    
    # Random indices for visualization
    indices = np.random.choice(len(fully_sampled_data), num_examples, replace=False)
    
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Fully sampled image
        im0 = axes[0].imshow(fully_sampled_data[idx], cmap='gray')
        axes[0].set_title('Fully Sampled (Target)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Undersampled image
        im1 = axes[1].imshow(undersampled_data[idx], cmap='gray')
        axes[1].set_title('Undersampled (Input)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_folder, f"example_{i+1}.png"), dpi=300)
        plt.close(fig)
    
    print(f"Example visualizations saved to {vis_folder}")

def main():
    parser = argparse.ArgumentParser(description='Generate MRI Dataset')
    parser.add_argument('--output_folder', type=str, required=True,
                      help='Folder to save the generated dataset')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of 3D brain structures to generate')
    parser.add_argument('--slices_per_sample', type=int, default=3,
                      help='Number of 2D slices to extract from each 3D sample')
    parser.add_argument('--undersample_factor', type=int, default=4,
                      help='Keep 1 out of every N lines in k-space')
    parser.add_argument('--central_lines_zero_percent', type=float, default=0.05,
                      help='Percentage of central lines to set to zero')
    args = parser.parse_args()
    
    create_dataset(
        output_folder=args.output_folder,
        num_samples=args.num_samples,
        slices_per_sample=args.slices_per_sample,
        undersample_factor=args.undersample_factor,
        central_lines_zero_percent=args.central_lines_zero_percent
    )

if __name__ == "__main__":
    main()