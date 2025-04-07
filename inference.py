import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pathlib import Path

# Import models from your model_functions.py
from model_functions import *


def load_model(model_path, model_type, device):
    """
    Load a trained model from a checkpoint file.

    Parameters:
    - model_path: Path to the model checkpoint (.pt file)
    - model_type: Type of model to load ('spatial', 'frequency', 'dual', 'sequential')
    - device: Device to load the model on (CPU or CUDA)

    Returns:
    - Loaded model
    """

    # Create the appropriate model based on type
    if model_type == 'spatial':
        model = SpatialDomainUNet(in_channels=1, out_channels=1)
    elif model_type == 'frequency':
        model = FrequencyDomainUNet(in_channels=1, out_channels=1)
    elif model_type == 'dual':
        model = DualDomainUNet(in_channels=1, out_channels=1)
    elif model_type == 'SequentialHybrid':
        model = SequentialHybridModel2(
            in_channels=1, out_channels=1, base_filters=64, depth=4, use_fusion_skips=False)
    elif model_type == 'SequentialHybrid_Skips':
        model = SequentialHybridModel2(
            in_channels=1, out_channels=1, base_filters=64, depth=4, use_fusion_skips=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load state dictionary from checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded {model_type} model from {model_path}")
    return model


def run_inference(model, input_data, device, batch_size=16):
    """
    Run inference on the input data using the loaded model.

    Parameters:
    - model: Trained model
    - input_data: Undersampled MRI data
    - device: Device to run inference on
    - batch_size: Batch size for inference

    Returns:
    - Reconstructed images
    """
    # Convert input data to torch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Create a TensorDataset for batching
    dataset = torch.utils.data.TensorDataset(input_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    # Container for reconstructed images
    reconstructed = []

    # Run inference in batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            inputs = batch[0].to(device)

            # Handle channels dimension if needed
            if inputs.ndim == 3:  # [batch, height, width]
                inputs = inputs.unsqueeze(1)  # Add channel dimension

            # Forward pass and get output
            outputs_dict = model(inputs)
            outputs = outputs_dict['output']

            # Collect reconstructed images
            reconstructed.append(outputs.cpu().numpy())

    # Concatenate results and ensure proper shape
    reconstructed = np.concatenate(reconstructed, axis=0)

    # Remove singleton dimensions if needed (e.g., [N, 1, H, W] -> [N, H, W])
    if reconstructed.shape[1] == 1:
        reconstructed = reconstructed.squeeze(1)

    return reconstructed


def calculate_metrics(reconstructed, target):
    """
    Calculate image quality metrics between reconstructed and target images.

    Parameters:
    - reconstructed: Reconstructed images
    - target: Ground truth target images

    Returns:
    - Dictionary of metrics
    """
    # Mean Squared Error
    mse = np.mean((reconstructed - target) ** 2, axis=(1, 2))

    # Mean Absolute Error
    mae = np.mean(np.abs(reconstructed - target), axis=(1, 2))

    # Peak Signal-to-Noise Ratio
    max_pixel = 1.0  # Assuming normalized data
    psnr = 10 * np.log10((max_pixel ** 2) / (mse + 1e-10))

    # Calculate average metrics
    avg_mse = np.mean(mse)
    avg_mae = np.mean(mae)
    avg_psnr = np.mean(psnr)

    return {
        'MSE': avg_mse,
        'MAE': avg_mae,
        'PSNR': avg_psnr
    }


def visualize_results(undersampled, reconstructed, target, output_dir, indices=None, num_samples=None):
    """
    Create and save visualizations of reconstruction results.

    Parameters:
    - undersampled: Undersampled input images
    - reconstructed: Reconstructed images by the model
    - target: Ground truth target images
    - output_dir: Directory to save visualizations
    - indices: Specific indices to visualize (if None, random samples are chosen)
    - num_samples: Number of samples to visualize (if indices is None)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine indices to visualize
    if indices is None:
        # If no specific indices, select random ones
        if num_samples is None:
            num_samples = min(10, len(reconstructed))
        indices = np.random.choice(
            len(reconstructed), num_samples, replace=False)

    # Calculate difference images for comparison
    difference = np.abs(target - reconstructed)

    # Create visualizations for each selected index
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Reconstructed output
        im0 = axes[0].imshow(reconstructed[idx], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Reconstructed (Output)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Target (ground truth)
        im1 = axes[1].imshow(target[idx], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Target (Ground Truth)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Difference map
        max_diff = np.max(difference[idx]) if np.max(
            difference[idx]) > 0 else 1.0
        im2 = axes[2].imshow(difference[idx], cmap='hot',
                             vmin=0, vmax=max_diff)
        axes[2].set_title('Difference |Target - Reconstructed|')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{idx}.png"), dpi=300)
        plt.close(fig)


def save_metrics_summary(metrics, output_dir):
    """
    Save reconstruction metrics to a text file.

    Parameters:
    - metrics: Dictionary of metrics
    - output_dir: Directory to save the metrics file
    """
    metrics_path = os.path.join(output_dir, "metrics_summary.txt")

    with open(metrics_path, 'w') as f:
        f.write("Reconstruction Metrics Summary\n")
        f.write("============================\n\n")

        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with a trained MRI reconstruction model')

    parser.add_argument('--input_dir', type=str, default='../SampleData',
                        help='Directory containing undersampled and fully sampled data (default: ./SampleData)')
    parser.add_argument('--output_dir', type=str, default='./InferenceResults',
                        help='Directory to save reconstruction results (default: ./InferenceResults)')
    parser.add_argument('--model_path', type=str, default='../best_mri_SequentialHybrid_model.pt',
                        help='Path to the trained model checkpoint (.pt file) (default: ./best_mri_SequentialHybrid_model.pt)')
    parser.add_argument('--model_type', type=str, default='SequentialHybrid',
                        choices=['spatial', 'frequency', 'dual',
                                 'SequentialHybrid', 'SequentialHybrid_Skips'],
                        help='Type of model architecture (default: SequentialHybrid)')
    parser.add_argument('--num_visualizations', type=int, default=10,
                        help='Number of sample visualizations to create (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cuda or cpu)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load input data
    undersampled_path = os.path.join(args.input_dir, "undersampled_data.npy")
    fully_sampled_path = os.path.join(args.input_dir, "fully_sampled_data.npy")

    # Handle the case where only undersampled data is available (no ground truth)
    only_undersampled = False
    if not os.path.exists(undersampled_path):
        raise FileNotFoundError(
            f"Undersampled data file not found in {args.input_dir}")

    if not os.path.exists(fully_sampled_path):
        print(
            f"Warning: Fully sampled data file not found in {args.input_dir}")
        print("Running in inference-only mode (no metrics will be calculated)")
        only_undersampled = True

    print(f"Loading data from {args.input_dir}...")
    undersampled_data = np.load(undersampled_path)
    fully_sampled_data = np.load(fully_sampled_path)

    print(f"Data loaded: {len(undersampled_data)} samples")
    print(f"Data shape: {undersampled_data.shape}")

    # Load model
    model = load_model(args.model_path, args.model_type, device)

    # Run inference
    print("Running inference...")
    start_time = time.time()
    reconstructed_data = run_inference(
        model, undersampled_data, device, args.batch_size)
    elapsed_time = time.time() - start_time

    print(f"Inference complete in {elapsed_time:.2f} seconds")
    print(f"Processed {len(reconstructed_data)} samples")

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(reconstructed_data, fully_sampled_data)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

    # Save metrics
    save_metrics_summary(metrics, args.output_dir)

    # Create visualizations
    print(f"Creating {args.num_visualizations} visualizations...")
    vis_indices = np.random.choice(
        len(reconstructed_data), args.num_visualizations, replace=False)
    vis_output_dir = os.path.join(args.output_dir, "visualizations")

    visualize_results(
        undersampled_data,
        reconstructed_data,
        fully_sampled_data,
        vis_output_dir,
        indices=vis_indices
    )

    # Save reconstructed data
    reconstructed_path = os.path.join(
        args.output_dir, "reconstructed_data.npy")
    np.save(reconstructed_path, reconstructed_data)

    print(f"Results saved to {args.output_dir}")
    print(f"Visualizations saved to {vis_output_dir}")


if __name__ == "__main__":
    main()
