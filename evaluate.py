import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import random
from tqdm import tqdm

from model_functions import SequentialHybridModel2

# Load your model components from paste-2.txt
# For demonstration, assuming these are defined elsewhere
# from model_components import FFTModule, IFFTModule, UNetConvBlock, UNetUpBlock, SequentialHybridModel2

# Reusing your data generation code with added noise and undersampling options


def create_sphere(shape, center, radius, value=1.0):
    """Create a 3D sphere in a volume."""
    coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance = np.sqrt(
        (coords[0] - center[0])**2 +
        (coords[1] - center[1])**2 +
        (coords[2] - center[2])**2
    )
    mask = distance <= radius
    return mask.astype(float) * value


def generate_brain_structure(shape=(64, 64, 64), num_small_spheres=None):
    """Generate a 3D brain-like structure with a large sphere and smaller spheres inside."""
    if num_small_spheres is None:
        num_small_spheres = np.random.randint(3, 6)  # 3-5 smaller spheres

    volume = np.zeros(shape)

    # Center of the volume
    center = np.array([shape[0]//2, shape[1]//2, shape[2]//2])

    # Radius of the large sphere (brain)
    main_radius = min(shape) // 2 - 2

    # Create the main sphere with gray level 0.5
    volume = create_sphere(shape, center, main_radius, value=0.5)

    # Create smaller spheres with gray level 1.0
    for _ in range(num_small_spheres):
        # Random position within the main sphere
        # Limit to 80% of main radius distance
        r = np.random.random() * 0.8 * main_radius
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi

        # Convert spherical to Cartesian coordinates
        dx = r * np.sin(phi) * np.cos(theta)
        dy = r * np.sin(phi) * np.sin(theta)
        dz = r * np.cos(phi)

        small_center = center + np.array([dx, dy, dz])
        small_center = small_center.astype(int)

        # Random radius for the small sphere (between 10% and 25% of main radius)
        small_radius = np.random.uniform(0.1, 0.25) * main_radius

        # Create the small sphere with gray level 1.0
        small_sphere = create_sphere(
            shape, small_center, small_radius, value=1.0)

        # Add the small sphere to the volume (overwrite any existing values)
        volume = np.maximum(volume, small_sphere)

    return volume


def get_2d_slices(volume, min_small_spheres=1):
    """Extract 2D slices from the 3D volume, keeping only those with small spheres."""
    slices = []
    for z in range(volume.shape[2]):
        slice_2d = volume[:, :, z]

        # Check if the slice contains any smaller spheres (pixels with value 1.0)
        if np.sum(slice_2d > 0.9) >= min_small_spheres:
            slices.append(slice_2d)

    return slices

# Enhanced k-space undersampling with additional features for evaluation


def undersample_kspace(slice_2d, undersample_factor=4, central_lines_zero_percent=0.05,
                       add_noise=False, noise_level=0.05, missing_points_percent=0):
    """
    Apply FFT, undersample, add noise, introduce missing points, and apply IFFT 
    to simulate various MRI data challenges.

    Parameters:
    - slice_2d: 2D array representing a fully sampled slice
    - undersample_factor: Keep 1 out of every N lines (2 for 2x acceleration)
    - central_lines_zero_percent: Percentage of central lines to set to zero
    - add_noise: Whether to add AWGN noise to k-space
    - noise_level: Standard deviation of the Gaussian noise
    - missing_points_percent: Percentage of random k-space points to set to zero

    Returns:
    - undersampled_slice: 2D array after undersampling and reconstruction
    - kspace_modified: The modified k-space for analysis
    """
    # Apply FFT
    kspace = np.fft.fft2(slice_2d)
    kspace_shifted = np.fft.fftshift(kspace)  # Shift to center low frequencies

    # Create undersampling mask
    mask = np.zeros_like(kspace_shifted)
    rows, cols = mask.shape

    # Keep 1 out of every undersample_factor rows
    for i in range(0, rows, undersample_factor):
        mask[i, :] = 1

    # Calculate number of central lines to zero out
    central_lines = int(rows * central_lines_zero_percent)
    if central_lines > 0:
        center_row = rows // 2
        start_row = center_row - central_lines // 2
        end_row = start_row + central_lines

        # Zero out the central lines
        mask[start_row:end_row, :] = 0

    # Apply the mask
    undersampled_kspace = kspace_shifted * mask

    # Add Gaussian noise if specified
    if add_noise:
        # Generate complex noise (real and imaginary parts)
        real_noise = np.random.normal(
            0, noise_level, undersampled_kspace.shape)
        imag_noise = np.random.normal(
            0, noise_level, undersampled_kspace.shape)
        complex_noise = real_noise + 1j * imag_noise

        # Add noise to k-space
        undersampled_kspace = undersampled_kspace + complex_noise

    # Set random points to zero if specified
    if missing_points_percent > 0:
        # Calculate how many points to set to zero
        total_points = rows * cols
        num_missing_points = int(total_points * missing_points_percent / 100)

        # Generate random indices for points to set to zero
        flat_indices = np.random.choice(
            total_points, num_missing_points, replace=False)
        row_indices = flat_indices // cols
        col_indices = flat_indices % cols

        # Set the selected points to zero
        for i, j in zip(row_indices, col_indices):
            undersampled_kspace[i, j] = 0

    # Apply IFFT to reconstruct
    undersampled_slice = np.fft.ifft2(np.fft.ifftshift(undersampled_kspace))

    # Take the absolute value to get a real image
    undersampled_slice = np.abs(undersampled_slice)

    return undersampled_slice, undersampled_kspace

# Modified dataset generation for evaluation


def generate_evaluation_dataset(num_samples=100, volume_shape=(64, 64, 64), slices_per_sample=3):
    """
    Generate a clean dataset for evaluation.

    Returns:
    - fully_sampled_data: List of fully sampled slices (ground truth)
    """
    fully_sampled_data = []

    for _ in tqdm(range(num_samples), desc="Generating evaluation dataset"):
        # Generate a new brain structure
        volume = generate_brain_structure(shape=volume_shape)

        # Get 2D slices containing small spheres
        slices = get_2d_slices(volume)

        # If we have enough slices, use them
        if len(slices) >= slices_per_sample:
            # Randomly select slices_per_sample slices
            selected_indices = np.random.choice(
                len(slices), slices_per_sample, replace=False)
            selected_slices = [slices[i] for i in selected_indices]

            for slice_2d in selected_slices:
                # Add the fully sampled slice (ground truth)
                fully_sampled_data.append(slice_2d)

    return np.array(fully_sampled_data)

# Class to apply different degradation conditions


class DegradationConditions:
    def __init__(self):
        self.conditions = {
            'clean': {
                'undersample_factor': 4,
                'central_lines_zero_percent': 0.05,
                'add_noise': False,
                'noise_level': 0,
                'missing_points_percent': 0
            },
            'noisy': {
                'undersample_factor': 4,
                'central_lines_zero_percent': 0.05,
                'add_noise': True,
                'noise_level': 0.05,
                'missing_points_percent': 0
            },
            '2x_acceleration': {
                'undersample_factor': 2,  # 2x acceleration
                'central_lines_zero_percent': 0.05,
                'add_noise': False,
                'noise_level': 0,
                'missing_points_percent': 0
            },
            'missing_kspace': {
                'undersample_factor': 4,
                'central_lines_zero_percent': 0.05,
                'add_noise': False,
                'noise_level': 0,
                'missing_points_percent': 5  # 5% of k-space points missing
            },
            'combined': {
                'undersample_factor': 2,  # 2x acceleration
                'central_lines_zero_percent': 0.05,
                'add_noise': True,
                'noise_level': 0.05,
                'missing_points_percent': 5  # 5% of k-space points missing
            }
        }

    def apply_condition(self, slice_2d, condition_name):
        """Apply a specific degradation condition to a slice."""
        if condition_name not in self.conditions:
            raise ValueError(f"Unknown condition: {condition_name}")

        params = self.conditions[condition_name]
        return undersample_kspace(
            slice_2d,
            undersample_factor=params['undersample_factor'],
            central_lines_zero_percent=params['central_lines_zero_percent'],
            add_noise=params['add_noise'],
            noise_level=params['noise_level'],
            missing_points_percent=params['missing_points_percent']
        )

# Metrics for evaluation


def calculate_metrics(ground_truth, reconstructed):
    """Calculate PSNR and SSIM metrics between ground truth and reconstructed images."""
    psnr_value = psnr(ground_truth, reconstructed,
                      data_range=ground_truth.max() - ground_truth.min())
    ssim_value = ssim(ground_truth, reconstructed,
                      data_range=ground_truth.max() - ground_truth.min())

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((ground_truth - reconstructed) ** 2)

    # Calculate Normalized Root Mean Square Error (NRMSE)
    rmse = np.sqrt(mse)
    nrmse = rmse / (ground_truth.max() - ground_truth.min())

    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'mse': mse,
        'nrmse': nrmse
    }

# Function to convert NumPy arrays to PyTorch tensors for model input


def prepare_for_model(image):
    """Convert a NumPy array to a PyTorch tensor with appropriate dimensions."""
    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    return tensor

# Main evaluation function


def evaluate_model(model_path, num_samples=50, conditions=None, device='cuda'):
    """
    Evaluate the model under different degradation conditions.

    Parameters:
    - model_path: Path to the saved model (.pt file)
    - num_samples: Number of samples to evaluate
    - conditions: List of condition names to evaluate (default: all conditions)
    - device: Device to run the model on ('cuda' or 'cpu')

    Returns:
    - results: Dictionary with evaluation results for each condition
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load the model
    print(f"Loading model from {model_path}")
    model = SequentialHybridModel2(
        in_channels=1, out_channels=1, base_filters=64, depth=4, use_fusion_skips=False)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get all available conditions if not specified
    degradation = DegradationConditions()
    if conditions is None:
        conditions = list(degradation.conditions.keys())

    # Generate evaluation dataset
    print("Generating evaluation dataset...")
    ground_truth_slices = generate_evaluation_dataset(num_samples=num_samples)
    print(f"Generated {len(ground_truth_slices)} slices for evaluation")

    # Initialize results dictionary
    results = {condition: {
        'psnr': [], 'ssim': [], 'mse': [], 'nrmse': []
    } for condition in conditions}

    # Example images for visualization
    example_images = {
        'ground_truth': [],
        'degraded': {},
        'reconstructed': {}
    }
    for condition in conditions:
        example_images['degraded'][condition] = []
        example_images['reconstructed'][condition] = []

    # Evaluate the model on each condition
    for condition in conditions:
        print(f"Evaluating condition: {condition}")

        for i, gt_slice in enumerate(tqdm(ground_truth_slices, desc=f"Processing {condition}")):
            # Apply degradation condition
            degraded_slice, _ = degradation.apply_condition(
                gt_slice, condition)

            # Prepare input for the model
            input_tensor = prepare_for_model(degraded_slice).to(device)

            # Model inference
            with torch.no_grad():
                output = model(input_tensor)

            # Convert output to NumPy array
            if isinstance(output, dict):
                # If model returns a dictionary, use the 'output' key
                reconstructed = output['output'].cpu().squeeze().numpy()
            else:
                # If model returns a tensor directly
                reconstructed = output.cpu().squeeze().numpy()

            # Calculate metrics
            metrics = calculate_metrics(gt_slice, reconstructed)
            for metric_name, metric_value in metrics.items():
                results[condition][metric_name].append(metric_value)

            # Save example images (first 5 samples)
            if i < 5:
                if len(example_images['ground_truth']) < 5:
                    example_images['ground_truth'].append(gt_slice)
                example_images['degraded'][condition].append(degraded_slice)
                example_images['reconstructed'][condition].append(
                    reconstructed)

    # Calculate average metrics for each condition
    for condition in conditions:
        # Get a list of original metrics before adding the new avg_ and std_ keys
        original_metrics = list(results[condition].keys())

        # Now calculate average and std for each original metric
        for metric in original_metrics:
            results[condition][f'avg_{metric}'] = np.mean(
                results[condition][metric])
            results[condition][f'std_{metric}'] = np.std(
                results[condition][metric])

    # Save example images for visualization
    save_example_images(example_images, 'evaluation_examples')

    # Print results summary
    print("\nEvaluation Results Summary:")
    print("=" * 80)
    print(f"{'Condition':<20} {'PSNR':<10} {'SSIM':<10} {'MSE':<10} {'NRMSE':<10}")
    print("-" * 80)
    for condition in conditions:
        print(f"{condition:<20} "
              f"{results[condition]['avg_psnr']:<10.4f} "
              f"{results[condition]['avg_ssim']:<10.4f} "
              f"{results[condition]['avg_mse']:<10.4e} "
              f"{results[condition]['avg_nrmse']:<10.4f}")
    print("=" * 80)

    # Generate summary plots
    generate_summary_plots(results, 'evaluation_results')

    return results


def save_example_images(example_images, output_dir):
    """Save example images for visualization."""
    os.makedirs(output_dir, exist_ok=True)

    # Get conditions
    conditions = list(example_images['degraded'].keys())

    # For each sample
    for i in range(len(example_images['ground_truth'])):
        # Create a separate figure for each sample
        # Using a larger figure size and a smarter layout
        num_conditions = len(conditions)
        # Scale width based on conditions
        fig_width = min(15, 5 + num_conditions * 2)
        fig_height = 8 if num_conditions <= 5 else 12

        plt.figure(figsize=(fig_width, fig_height))

        # Calculate rows and columns for a better layout
        if num_conditions <= 5:
            # Use 2 rows: ground truth on top, then pairs of degraded/reconstructed below
            rows, cols = 3, num_conditions
        else:
            # Use more rows for many conditions
            rows, cols = 2 + 2*((num_conditions+2)//3), 3

        # Plot ground truth in the first position
        plt.subplot(rows, cols, 1)
        plt.imshow(example_images['ground_truth'][i], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # Plot each condition (degraded and reconstructed)
        for j, condition in enumerate(conditions):
            # Calculate positions for degraded and reconstructed images
            if num_conditions <= 5:
                # For fewer conditions, arrange in rows
                pos_degraded = cols + j + 1
                pos_reconstructed = 2*cols + j + 1
            else:
                # For many conditions, arrange in a grid
                pos_degraded = (j//3)*2*cols + (j % 3) + cols + 1
                pos_reconstructed = (j//3)*2*cols + (j % 3) + cols + 4

            # Plot degraded image
            plt.subplot(rows, cols, pos_degraded)
            plt.imshow(example_images['degraded'][condition][i], cmap='gray')
            plt.title(f"{condition} - Degraded")
            plt.axis('off')

            # Plot reconstructed image
            plt.subplot(rows, cols, pos_reconstructed)
            plt.imshow(example_images['reconstructed']
                       [condition][i], cmap='gray')
            plt.title(f"{condition} - Reconstructed")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'example_{i+1}.png'), dpi=300)
        plt.close()


def generate_summary_plots(results, output_dir):
    """Generate summary plots for evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # Get conditions
    conditions = list(results.keys())

    # Plot PSNR and SSIM
    metrics = ['psnr', 'ssim', 'mse', 'nrmse']
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        avg_values = [results[condition]
                      [f'avg_{metric}'] for condition in conditions]
        std_values = [results[condition]
                      [f'std_{metric}'] for condition in conditions]

        # Use bar plot for better visualization
        plt.bar(conditions, avg_values, yerr=std_values, capsize=5, alpha=0.7)

        if metric in ['mse', 'nrmse']:
            # For error metrics, lower is better
            plt.ylabel(f"{metric.upper()} (lower is better)")
        else:
            # For quality metrics, higher is better
            plt.ylabel(f"{metric.upper()} (higher is better)")

        plt.title(f"Average {metric.upper()} across Different Conditions")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'{metric}_summary.png'), dpi=300)
        plt.close()

    # Create a comparison plot for normalized values
    plt.figure(figsize=(12, 8))

    # Normalize all metrics to [0,1] range for comparison
    normalized_metrics = {}
    for metric in metrics:
        values = [results[condition][f'avg_{metric}']
                  for condition in conditions]

        if metric in ['mse', 'nrmse']:
            # For error metrics, normalize and invert (so 1 is best)
            min_val, max_val = min(values), max(values)
            norm_values = [1 - (val - min_val) / (max_val - min_val)
                           if max_val > min_val else 0.5 for val in values]
        else:
            # For quality metrics, normalize directly
            min_val, max_val = min(values), max(values)
            norm_values = [(val - min_val) / (max_val - min_val)
                           if max_val > min_val else 0.5 for val in values]

        normalized_metrics[metric] = norm_values

    # Plot normalized metrics
    x = np.arange(len(conditions))
    width = 0.2
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    for i, metric in enumerate(metrics):
        plt.bar(x + offsets[i], normalized_metrics[metric],
                width, label=metric.upper())

    plt.xlabel('Condition')
    plt.ylabel('Normalized Score (higher is better)')
    plt.title('Comparison of All Metrics Across Conditions (Normalized)')
    plt.xticks(x, conditions, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'normalized_comparison.png'), dpi=300)
    plt.close()

# Additional analysis functions


def analyze_frequency_response(ground_truth, degraded, reconstructed):
    """Analyze the frequency response of degradation and reconstruction."""
    # Compute FFT of all images
    gt_fft = np.fft.fftshift(np.fft.fft2(ground_truth))
    degraded_fft = np.fft.fftshift(np.fft.fft2(degraded))
    reconstructed_fft = np.fft.fftshift(np.fft.fft2(reconstructed))

    # Compute magnitude
    gt_mag = np.log(np.abs(gt_fft) + 1e-10)
    degraded_mag = np.log(np.abs(degraded_fft) + 1e-10)
    reconstructed_mag = np.log(np.abs(reconstructed_fft) + 1e-10)

    # Plot
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(gt_mag, cmap='viridis')
    plt.title('Ground Truth FFT (log magnitude)')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(degraded_mag, cmap='viridis')
    plt.title('Degraded FFT (log magnitude)')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_mag, cmap='viridis')
    plt.title('Reconstructed FFT (log magnitude)')
    plt.colorbar()

    plt.tight_layout()

    # Also compute the error in frequency domain
    error_fft = gt_fft - reconstructed_fft
    error_mag = np.log(np.abs(error_fft) + 1e-10)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(error_mag, cmap='hot')
    plt.title('Reconstruction Error in Frequency Domain (log magnitude)')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(ground_truth - reconstructed), cmap='hot')
    plt.title('Reconstruction Error in Spatial Domain')
    plt.colorbar()

    plt.tight_layout()

    return {
        'gt_fft': gt_fft,
        'degraded_fft': degraded_fft,
        'reconstructed_fft': reconstructed_fft,
        'error_fft': error_fft
    }


# Main execution
if __name__ == "__main__":
    # Update with your model path
    model_path = "./best_mri_SequentialHybrid_model.pt"

    # Evaluate model performance under different conditions
    results = evaluate_model(
        model_path=model_path,
        num_samples=50,  # Adjust based on computational resources
        conditions=None,  # Evaluate all conditions
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Evaluation complete!")
