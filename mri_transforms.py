import numpy as np
import torch
import torch.nn.functional as F


class MRITransforms:
    """
    Class to handle various MRI data transformations for data augmentation and realistic simulation.
    This includes:
    - Various k-space undersampling patterns (radial, cartesian, random, etc.)
    - Noise addition (complex Gaussian, Rician, etc.)
    - Motion artifacts
    - Coil sensitivity simulation
    """
    
    def __init__(self, image_size=(256, 256)):
        """
        Initialize the transform class.
        
        Args:
            image_size (tuple): Size of the input images (height, width)
        """
        self.image_size = image_size
        self._create_sampling_masks()
        
    def _create_sampling_masks(self):
        """
        Pre-create various sampling masks for efficiency
        """
        h, w = self.image_size
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        y, x = y - center_y, x - center_x
        
        # Create distance map from center
        radius = np.sqrt(x**2 + y**2)
        
        # Create angle map
        theta = np.arctan2(y, x)
        
        # Store for later use
        self.radius_map = radius
        self.theta_map = theta
        
    def apply_transform(self, image, transform_type='random', params=None):
        """
        Apply a specific transformation to the input image
        
        Args:
            image (torch.Tensor): Input image tensor [batch, channel, height, width]
            transform_type (str): Type of transform to apply ('random', 'undersample', 'noise', etc.)
            params (dict): Parameters specific to the transform type
            
        Returns:
            torch.Tensor: Transformed image
        """
        if transform_type == 'random':
            # Apply a random combination of transforms
            transforms = ['undersample', 'noise', 'motion']
            selected = np.random.choice(transforms, size=np.random.randint(1, len(transforms)+1), replace=False)
            
            transformed = image
            for t in selected:
                transformed = self.apply_transform(transformed, t, params)
            return transformed
        
        elif transform_type == 'undersample':
            return self.apply_undersampling(image, params)
        
        elif transform_type == 'noise':
            return self.apply_noise(image, params)
        
        elif transform_type == 'motion':
            return self.apply_motion_artifact(image, params)
            
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    def apply_undersampling(self, image, params=None):
        """
        Apply k-space undersampling to simulate accelerated MRI acquisition
        
        Args:
            image (torch.Tensor): Input image tensor [batch, channel, height, width]
            params (dict): Undersampling parameters including:
                - pattern: 'cartesian', 'radial', 'random', 'variable_density'
                - acceleration: Undersampling factor (higher = more undersampling)
                - center_fraction: Fraction of center k-space to fully sample
                
        Returns:
            torch.Tensor: Undersampled image
        """
        if params is None:
            params = {
                'pattern': np.random.choice(['cartesian', 'radial', 'random', 'variable_density']),
                'acceleration': np.random.uniform(2, 8),
                'center_fraction': np.random.uniform(0.01, 0.2)
            }
        
        pattern = params.get('pattern', 'cartesian')
        acceleration = params.get('acceleration', 4)
        center_fraction = params.get('center_fraction', 0.1)
        
        # Get image dimensions
        batch_size, channels, h, w = image.shape
        
        # Convert to complex domain for FFT
        image_complex = torch.complex(image, torch.zeros_like(image))
        
        # Apply FFT to get k-space data
        kspace = torch.fft.fftshift(torch.fft.fft2(image_complex))
        
        # Create appropriate sampling mask
        mask = self._get_mask(pattern, acceleration, center_fraction, batch_size, h, w)
        mask = torch.tensor(mask, dtype=torch.float32, device=image.device)
        mask = mask.unsqueeze(1).expand(-1, channels, -1, -1)
        
        # Apply mask to k-space
        masked_kspace = kspace * mask
        
        # Convert back to image domain
        undersampled_image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(masked_kspace)))
        
        return undersampled_image
    
    def _get_mask(self, pattern, acceleration, center_fraction, batch_size, h, w):
        """
        Create a sampling mask for k-space undersampling
        
        Args:
            pattern (str): Undersampling pattern type
            acceleration (float): Acceleration factor
            center_fraction (float): Fraction of center k-space to fully sample
            batch_size (int): Batch size
            h, w (int): Height and width of the image
            
        Returns:
            np.ndarray: Sampling mask [batch_size, h, w]
        """
        masks = np.zeros((batch_size, h, w), dtype=np.float32)
        
        for i in range(batch_size):
            if pattern == 'cartesian':
                # Cartesian undersampling (regularly spaced lines)
                mask = np.zeros((h, w), dtype=np.float32)
                
                # Determine line spacing based on acceleration factor
                skip = int(acceleration)
                
                # Sample regular lines (every 'skip' lines)
                mask[:, ::skip] = 1.0
                
                # Fully sample the center region
                center_width = int(w * center_fraction)
                center_start = (w - center_width) // 2
                mask[:, center_start:center_start+center_width] = 1.0
                
            elif pattern == 'random':
                # Random undersampling
                mask = np.random.uniform(size=(h, w)) < (1.0 / acceleration)
                mask = mask.astype(np.float32)
                
                # Ensure center is sampled
                center_h, center_w = h // 2, w // 2
                center_size_h = int(h * center_fraction)
                center_size_w = int(w * center_fraction)
                
                start_h = center_h - center_size_h // 2
                start_w = center_w - center_size_w // 2
                
                mask[start_h:start_h+center_size_h, start_w:start_w+center_size_w] = 1.0
                
            elif pattern == 'radial':
                # Radial/spoke undersampling
                mask = np.zeros((h, w), dtype=np.float32)
                
                # Number of spokes based on acceleration
                num_spokes = int((h + w) / (2 * acceleration))
                
                # Generate random angles for spokes
                theta_values = np.linspace(0, np.pi, num_spokes, endpoint=False)
                
                # Create radial lines
                for theta in theta_values:
                    # Line equation in k-space: y = (tan theta) * x
                    x_centers = np.arange(-w//2, w//2)
                    y_centers = np.round(np.tan(theta) * x_centers).astype(int)
                    
                    valid_indices = (y_centers >= -h//2) & (y_centers < h//2)
                    y_centers = y_centers[valid_indices]
                    x_centers = x_centers[valid_indices]
                    
                    # Shift to image coordinates
                    y_centers += h // 2
                    x_centers += w // 2
                    
                    mask[y_centers, x_centers] = 1.0
                    
                # Ensure the center is fully sampled
                center_radius = int(min(h, w) * center_fraction / 2)
                y_grid, x_grid = np.ogrid[:h, :w]
                center_mask = ((y_grid - h//2)**2 + (x_grid - w//2)**2) <= center_radius**2
                mask[center_mask] = 1.0
                
            elif pattern == 'variable_density':
                # Variable density random sampling (denser in center)
                mask = np.zeros((h, w), dtype=np.float32)
                
                center_y, center_x = h // 2, w // 2
                y_grid, x_grid = np.mgrid[:h, :w]
                
                # Distance from center, normalized to [0, 1]
                radius = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
                radius = radius / np.max(radius)
                
                # Probability decreases with distance from center
                prob = (1.0 - radius)**2  # Quadratic falloff
                
                # Scale probabilities to achieve target acceleration
                scaling = (h * w) / (acceleration * np.sum(prob))
                prob = prob * scaling
                prob = np.minimum(prob, 1.0)  # Cap at 1.0
                
                # Random sampling based on probabilities
                mask = np.random.uniform(size=(h, w)) < prob
                mask = mask.astype(np.float32)
                
                # Ensure center is fully sampled
                center_radius = int(min(h, w) * center_fraction / 2)
                center_mask = radius <= (center_fraction / 2)
                mask[center_mask] = 1.0
                
            else:
                raise ValueError(f"Unknown undersampling pattern: {pattern}")
            
            masks[i] = mask
            
        return masks
    
    def apply_noise(self, image, params=None):
        """
        Apply realistic MRI noise to images
        
        Args:
            image (torch.Tensor): Input image tensor [batch, channel, height, width]
            params (dict): Noise parameters including:
                - noise_type: 'gaussian', 'rician', 'complex_gaussian'
                - noise_level: Standard deviation of the noise (relative to image intensity)
                
        Returns:
            torch.Tensor: Noisy image
        """
        if params is None:
            params = {
                'noise_type': np.random.choice(['gaussian', 'rician', 'complex_gaussian']),
                'noise_level': np.random.uniform(0.01, 0.1)
            }
        
        noise_type = params.get('noise_type', 'gaussian')
        noise_level = params.get('noise_level', 0.05)
        
        if noise_type == 'gaussian':
            # Add standard Gaussian noise
            noise = torch.randn_like(image) * noise_level
            noisy_image = image + noise
            
        elif noise_type == 'rician':
            # Rician noise (realistic for magnitude MRI)
            # First convert to complex domain
            image_complex = torch.complex(image, torch.zeros_like(image))
            
            # Add complex Gaussian noise to real and imaginary components
            noise_real = torch.randn_like(image) * noise_level
            noise_imag = torch.randn_like(image) * noise_level
            
            noisy_complex = torch.complex(image + noise_real, noise_imag)
            
            # Take magnitude to get Rician noise
            noisy_image = torch.abs(noisy_complex)
            
        elif noise_type == 'complex_gaussian':
            # Complex Gaussian noise (k-space noise)
            # Convert to k-space
            image_complex = torch.complex(image, torch.zeros_like(image))
            kspace = torch.fft.fftshift(torch.fft.fft2(image_complex))
            
            # Add complex noise in k-space
            noise_real = torch.randn_like(image) * noise_level 
            noise_imag = torch.randn_like(image) * noise_level
            noise_complex = torch.complex(noise_real, noise_imag)
            
            noisy_kspace = kspace + noise_complex
            
            # Convert back to image domain
            noisy_image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(noisy_kspace)))
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
        return noisy_image
    
    def apply_motion_artifact(self, image, params=None):
        """
        Apply motion artifacts to simulate patient movement during acquisition
        
        Args:
            image (torch.Tensor): Input image tensor [batch, channel, height, width]
            params (dict): Motion parameters including:
                - motion_type: 'translation', 'rotation', 'random'
                - severity: Controls the amount of motion (0.0 to 1.0)
                
        Returns:
            torch.Tensor: Image with motion artifacts
        """
        if params is None:
            params = {
                'motion_type': np.random.choice(['translation', 'rotation', 'random']),
                'severity': np.random.uniform(0.1, 0.5)
            }
        
        motion_type = params.get('motion_type', 'random')
        severity = params.get('severity', 0.3)
        
        # Convert to complex domain for FFT
        image_complex = torch.complex(image, torch.zeros_like(image))
        
        # Apply FFT to get k-space data
        kspace = torch.fft.fftshift(torch.fft.fft2(image_complex))
        
        batch_size, channels, h, w = image.shape
        
        # Create empty corrupted k-space
        corrupted_kspace = kspace.clone()
        
        for i in range(batch_size):
            for c in range(channels):
                if motion_type == 'translation' or (motion_type == 'random' and np.random.choice([True, False])):
                    # Phase shifts in k-space (equivalent to translations in image domain)
                    # Randomly shift different k-space lines to simulate inconsistent motion
                    
                    # Determine how many lines to corrupt (based on severity)
                    num_lines_to_corrupt = int(h * severity)
                    lines_to_corrupt = np.random.choice(h, size=num_lines_to_corrupt, replace=False)
                    
                    # Apply random phase shifts to corrupted lines
                    for line in lines_to_corrupt:
                        # Random translation amount
                        shift_x = np.random.uniform(-10, 10) * severity
                        
                        # Create phase ramp (equivalent to shift in image domain)
                        x_indices = np.arange(-w//2, w//2) / w
                        phase_ramp = np.exp(1j * 2 * np.pi * x_indices * shift_x)
                        
                        # Apply phase ramp to this k-space line
                        corrupted_kspace[i, c, line] = kspace[i, c, line] * torch.tensor(
                            phase_ramp, dtype=torch.complex64, device=kspace.device)
                
                if motion_type == 'rotation' or (motion_type == 'random' and np.random.choice([True, False])):
                    # Simulate rotation motion by corrupting angular segments in k-space
                    
                    # Get polar coordinates for each point in k-space
                    y_indices, x_indices = np.indices((h, w))
                    y_center, x_center = h // 2, w // 2
                    y_indices = y_indices - y_center
                    x_indices = x_indices - x_center
                    
                    angles = np.arctan2(y_indices, x_indices)
                    
                    # Divide k-space into angular segments
                    num_segments = 18  # Number of angular segments
                    segments = np.floor((angles + np.pi) / (2 * np.pi / num_segments)).astype(int)
                    
                    # Corrupt some segments (based on severity)
                    num_segments_to_corrupt = int(num_segments * severity)
                    segments_to_corrupt = np.random.choice(num_segments, size=num_segments_to_corrupt, replace=False)
                    
                    # Create a mask for the segments to corrupt
                    segment_mask = np.zeros((h, w), dtype=bool)
                    for segment in segments_to_corrupt:
                        segment_mask |= (segments == segment)
                    
                    # Apply random phase to corrupted segments
                    for segment in segments_to_corrupt:
                        phase_shift = np.random.uniform(-np.pi, np.pi)
                        mask = (segments == segment)
                        
                        # Get indices where mask is True
                        y_idx, x_idx = np.where(mask)
                        
                        # Apply phase shifts
                        phase_factor = np.exp(1j * phase_shift)
                        corrupted_kspace[i, c, y_idx, x_idx] = kspace[i, c, y_idx, x_idx] * phase_factor
        
        # Convert back to image domain
        motion_image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(corrupted_kspace)))
        
        return motion_image
    
    def apply_random_transforms(self, image, probability=0.5):
        """
        Apply random combinations of transforms with given probability
        
        Args:
            image (torch.Tensor): Input image tensor
            probability (float): Probability of applying each transform
            
        Returns:
            torch.Tensor: Transformed image
        """
        transformed = image
        
        # Randomly apply undersampling
        if np.random.random() < probability:
            params = {
                'pattern': np.random.choice(['cartesian', 'radial', 'random', 'variable_density']),
                'acceleration': np.random.uniform(2, 6),
                'center_fraction': np.random.uniform(0.05, 0.15)
            }
            transformed = self.apply_undersampling(transformed, params)
        
        # Randomly apply noise
        if np.random.random() < probability:
            params = {
                'noise_type': np.random.choice(['gaussian', 'rician', 'complex_gaussian']),
                'noise_level': np.random.uniform(0.01, 0.07)
            }
            transformed = self.apply_noise(transformed, params)
        
        # Randomly apply motion artifacts (less frequently)
        if np.random.random() < (probability * 0.5):
            params = {
                'motion_type': np.random.choice(['translation', 'rotation', 'random']),
                'severity': np.random.uniform(0.1, 0.3)
            }
            transformed = self.apply_motion_artifact(transformed, params)
        
        return transformed


class DataAugmentationWrapper:
    """
    Wrapper class to apply MRITransforms during training
    Can be integrated with PyTorch DataLoader
    """
    
    def __init__(self, dataset, transform_probability=0.8, augmentation_probability=0.5, 
                 return_kspace=False, device='cpu'):
        """
        Initialize the data augmentation wrapper
        
        Args:
            dataset: Original PyTorch dataset
            transform_probability: Probability of applying MRI-specific transforms
            augmentation_probability: Probability of applying each specific transform
            return_kspace: Whether to return k-space data as well
            device: Device to perform computations on
        """
        self.dataset = dataset
        self.transform_probability = transform_probability
        self.augmentation_probability = augmentation_probability
        self.transforms = MRITransforms()
        self.return_kspace = return_kspace
        self.device = device
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original sample
        undersampled, fully_sampled = self.dataset[idx]
        
        # Ensure data is on the correct device
        undersampled = undersampled.to(self.device)
        fully_sampled = fully_sampled.to(self.device)
        
        # Apply transformations with certain probability
        if np.random.random() < self.transform_probability:
            undersampled = self.transforms.apply_random_transforms(
                undersampled, probability=self.augmentation_probability)
        
        # Convert to k-space if requested
        if self.return_kspace:
            # Convert to complex domain
            undersampled_complex = torch.complex(undersampled, torch.zeros_like(undersampled))
            fully_sampled_complex = torch.complex(fully_sampled, torch.zeros_like(fully_sampled))
            
            # Calculate k-space
            undersampled_kspace = torch.fft.fftshift(torch.fft.fft2(undersampled_complex))
            fully_sampled_kspace = torch.fft.fftshift(torch.fft.fft2(fully_sampled_complex))
            
            return (undersampled, undersampled_kspace), (fully_sampled, fully_sampled_kspace)
        
        return undersampled, fully_sampled



