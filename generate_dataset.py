import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

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
        r = np.random.random() * 0.8 * main_radius  # Limit to 80% of main radius distance
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
        small_sphere = create_sphere(shape, small_center, small_radius, value=1.0)
        
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

def undersample_kspace(slice_2d, undersample_factor=4, central_lines_zero_percent=0.05):
    """
    Apply FFT, undersample, and apply IFFT to simulate undersampled MRI data.
    
    Parameters:
    - slice_2d: 2D array representing a fully sampled slice
    - undersample_factor: Keep 1 out of every N lines
    - central_lines_zero_percent: Percentage of central lines to set to zero
    
    Returns:
    - undersampled_slice: 2D array after undersampling and reconstruction
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
    
    # Apply IFFT to reconstruct
    undersampled_slice = np.fft.ifft2(np.fft.ifftshift(undersampled_kspace))
    
    # Take the absolute value to get a real image
    undersampled_slice = np.abs(undersampled_slice)
    
    return undersampled_slice

def generate_dataset(num_samples=100, volume_shape=(64, 64, 64), slices_per_sample=3, 
                     undersample_factor=4, central_lines_zero_percent=0.05):
    """
    Generate a dataset of fully sampled and undersampled MRI slices.
    
    Parameters:
    - num_samples: Number of 3D structures to generate
    - volume_shape: Shape of the 3D volumes
    - slices_per_sample: Number of slices to extract per sample
    - undersample_factor: Keep 1 out of every N lines in k-space
    - central_lines_zero_percent: Percentage of central lines to set to zero
    
    Returns:
    - fully_sampled_data: List of fully sampled slices
    - undersampled_data: List of corresponding undersampled slices
    """
    fully_sampled_data = []
    undersampled_data = []
    
    for _ in range(num_samples):
        # Generate a new brain structure
        volume = generate_brain_structure(shape=volume_shape)
        
        # Get 2D slices containing small spheres
        slices = get_2d_slices(volume)
        
        # If we have enough slices, use them
        if len(slices) >= slices_per_sample:
            # Randomly select slices_per_sample slices
            selected_indices = np.random.choice(len(slices), slices_per_sample, replace=False)
            selected_slices = [slices[i] for i in selected_indices]
            
            for slice_2d in selected_slices:
                # Add the fully sampled slice
                fully_sampled_data.append(slice_2d)
                
                # Generate the corresponding undersampled slice
                undersampled_slice = undersample_kspace(slice_2d, undersample_factor, central_lines_zero_percent)
                undersampled_data.append(undersampled_slice)
    
    return np.array(fully_sampled_data), np.array(undersampled_data)

def visualize_sample(fully_sampled, undersampled, index=0):
    """Visualize a pair of fully sampled and undersampled slices."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(fully_sampled[index], cmap='gray')
    plt.title('Fully Sampled')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(undersampled[index], cmap='gray')
    plt.title('Undersampled')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def visualize_kspace(slice_2d):
    """Visualize k-space of a slice."""
    kspace = np.fft.fftshift(np.fft.fft2(slice_2d))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(kspace), norm=plt.Normalize(0, 100), cmap='viridis')
    plt.title('K-space Magnitude (log scale)')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(kspace), cmap='hsv')
    plt.title('K-space Phase')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

