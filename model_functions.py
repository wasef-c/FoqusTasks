import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

from generate_dataset import generate_dataset, visualize_sample

# Custom PyTorch Dataset


class MRIDataset(Dataset):
    def __init__(self, undersampled_images, fully_sampled_images):
        self.undersampled = torch.tensor(
            undersampled_images, dtype=torch.float32)
        self.fully_sampled = torch.tensor(
            fully_sampled_images, dtype=torch.float32)

    def __len__(self):
        return len(self.undersampled)

    def __getitem__(self, idx):
        return self.undersampled[idx], self.fully_sampled[idx]

# FFT/IFFT Conversion Module


class FFTModule(nn.Module):
    def __init__(self):
        super(FFTModule, self).__init__()

    def forward(self, x):
        # Convert to complex domain by adding zeros as imaginary part
        x_complex = torch.complex(x, torch.zeros_like(x))

        # Apply FFT2
        x_fft = torch.fft.fft2(x_complex)

        # Get magnitude and phase
        x_magnitude = torch.log(torch.abs(x_fft) + 1e-10)
        x_phase = torch.angle(x_fft)

        # Normalize magnitude and phase for easier processing
        batch_size, channels, h, w = x_magnitude.shape

        # Normalize each sample individually
        for b in range(batch_size):
            for c in range(channels):
                # Normalize magnitude
                x_min = x_magnitude[b, c].min()
                x_max = x_magnitude[b, c].max()
                if x_max > x_min:  # Avoid division by zero
                    x_magnitude[b, c] = (
                        x_magnitude[b, c] - x_min) / (x_max - x_min)

                # Normalize phase to [0,1]
                x_phase[b, c] = (x_phase[b, c] + np.pi) / (2 * np.pi)

        # Concatenate along channel dimension
        return torch.cat([x_magnitude, x_phase], dim=1)


class IFFTModule(nn.Module):
    def __init__(self):
        super(IFFTModule, self).__init__()

    def forward(self, x):
        # Split the input back into magnitude and phase
        batch_size, channels, h, w = x.shape
        assert channels % 2 == 0, "Channels must be even (half for magnitude, half for phase)"

        half_channels = channels // 2
        x_magnitude = x[:, :half_channels]
        x_phase = x[:, half_channels:]

        # Denormalize
        x_magnitude = torch.exp(x_magnitude)  # Reverse the log transform
        x_phase = x_phase * (2 * np.pi) - np.pi  # Scale back to [-π, π]

        # Convert to complex
        real = x_magnitude * torch.cos(x_phase)
        imag = x_magnitude * torch.sin(x_phase)
        x_complex = torch.complex(real, imag)

        # Apply IFFT2
        x_ifft = torch.fft.ifft2(x_complex)

        # Return absolute value (real part of the spatial domain image)
        return torch.abs(x_ifft)

# UNet building blocks


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, padding=padding))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.LeakyReLU(0.1))

        block.append(nn.Conv2d(out_channels, out_channels,
                     kernel_size=3, padding=padding))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.LeakyReLU(0.1))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='upconv'):
        super(UNetUpBlock, self).__init__()

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2,
                            align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        self.conv_block = UNetConvBlock(in_channels, out_channels)

    def forward(self, x, bridge):
        up = self.up(x)

        # Handle cases where dimensions don't match
        diffY = bridge.size()[2] - up.size()[2]
        diffX = bridge.size()[3] - up.size()[3]

        up = F.pad(up, [diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2])

        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out

# 1. Spatial Domain Only Model (Standard U-Net)


class SpatialDomainUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, depth=4):
        super(SpatialDomainUNet, self).__init__()

        self.depth = depth
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        # Downsampling path
        prev_channels = in_channels
        for i in range(depth):
            current_channels = base_filters * (2 ** i)
            self.down_path.append(UNetConvBlock(
                prev_channels, current_channels))
            prev_channels = current_channels

        # Bottom (bottleneck)
        bottom_channels = base_filters * (2 ** depth)
        self.bottom = UNetConvBlock(prev_channels, bottom_channels)

        # Upsampling path
        prev_channels = bottom_channels
        for i in reversed(range(depth)):
            current_channels = base_filters * (2 ** i)
            self.up_path.append(UNetUpBlock(prev_channels, current_channels))
            prev_channels = current_channels

        # Final output layer
        self.final = nn.Conv2d(prev_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        blocks = []

        # Downsampling
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i < self.depth - 1:
                x = nn.MaxPool2d(2)(x)

        # Bottom
        x = self.bottom(x)

        # Upsampling
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        # Final output
        x = self.final(x)

        return {
            'output': self.sigmoid(x),
            'spatial_features': x    # For visualization/analysis
        }


# 2. Frequency Domain Only Model
class FrequencyDomainUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, depth=4):
        super(FrequencyDomainUNet, self).__init__()

        # FFT conversion module
        self.fft_module = FFTModule()

        # IFFT conversion module
        self.ifft_module = IFFTModule()

        # Actual U-Net in frequency domain (operates on magnitude and phase)
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        # Downsampling path (input has 2 channels: magnitude and phase)
        prev_channels = in_channels * 2  # 2 channels for mag and phase
        for i in range(depth):
            current_channels = base_filters * (2 ** i)
            self.down_path.append(UNetConvBlock(
                prev_channels, current_channels))
            prev_channels = current_channels

        # Bottom (bottleneck)
        bottom_channels = base_filters * (2 ** depth)
        self.bottom = UNetConvBlock(prev_channels, bottom_channels)

        # Upsampling path
        prev_channels = bottom_channels
        for i in reversed(range(depth)):
            current_channels = base_filters * (2 ** i)
            self.up_path.append(UNetUpBlock(prev_channels, current_channels))
            prev_channels = current_channels

        # Final output layer (outputs 2 channels: magnitude and phase)
        self.final = nn.Conv2d(prev_channels, in_channels * 2, kernel_size=1)

    def forward(self, x):
        # Convert input to frequency domain
        x_kspace = self.fft_module(x)

        blocks = []

        # Downsampling in frequency domain
        for i, down in enumerate(self.down_path):
            x_kspace = down(x_kspace)
            blocks.append(x_kspace)
            if i < self.depth - 1:
                x_kspace = nn.MaxPool2d(2)(x_kspace)

        # Bottom
        x_kspace = self.bottom(x_kspace)

        # Upsampling in frequency domain
        for i, up in enumerate(self.up_path):
            x_kspace = up(x_kspace, blocks[-i-1])

        # Final output in frequency domain
        x_kspace = self.final(x_kspace)

        # Convert back to spatial domain
        x_spatial = self.ifft_module(x_kspace)

        # return x_spatial

        return {
            'output': x_spatial,
            'k_space_features': x_kspace,  # For visualization/analysis
        }

# 3. Dual-Domain U-Net


class DualDomainUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super(DualDomainUNet, self).__init__()

        # FFT module to convert to k-space
        self.fft_module = FFTModule()

        # Spatial domain branch (simplified U-Net)
        # Encoder
        self.s_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.1)
        )
        self.s_pool1 = nn.MaxPool2d(2)

        self.s_conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.1)
        )
        self.s_pool2 = nn.MaxPool2d(2)

        self.s_conv3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.LeakyReLU(0.1)
        )
        self.s_pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.s_bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters *
                      8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.s_upconv3 = nn.ConvTranspose2d(
            base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.s_upblock3 = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.LeakyReLU(0.1)
        )

        self.s_upconv2 = nn.ConvTranspose2d(
            base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.s_upblock2 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.1)
        )

        self.s_upconv1 = nn.ConvTranspose2d(
            base_filters*2, base_filters, kernel_size=2, stride=2)
        self.s_upblock1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.1)
        )

        # K-space domain branch (simplified U-Net)
        # Encoder
        self.k_conv1 = nn.Sequential(
            nn.Conv2d(in_channels*2, base_filters, kernel_size=3,
                      padding=1),  # *2 for mag and phase
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.1)
        )
        self.k_pool1 = nn.MaxPool2d(2)

        self.k_conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.1)
        )
        self.k_pool2 = nn.MaxPool2d(2)

        self.k_conv3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.LeakyReLU(0.1)
        )
        self.k_pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.k_bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters *
                      8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.k_upconv3 = nn.ConvTranspose2d(
            base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.k_upblock3 = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.LeakyReLU(0.1)
        )

        self.k_upconv2 = nn.ConvTranspose2d(
            base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.k_upblock2 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.1)
        )

        self.k_upconv1 = nn.ConvTranspose2d(
            base_filters*2, base_filters, kernel_size=2, stride=2)
        self.k_upblock1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.1)
        )

        # Merge and final output
        self.merge_conv = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.1)
        )

        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spatial domain branch
        s1 = self.s_conv1(x)
        s2 = self.s_conv2(self.s_pool1(s1))
        s3 = self.s_conv3(self.s_pool2(s2))

        s_bottleneck = self.s_bottleneck(self.s_pool3(s3))

        s_up3 = self.s_upconv3(s_bottleneck)
        s_up3 = torch.cat([s_up3, s3], dim=1)
        s_up3 = self.s_upblock3(s_up3)

        s_up2 = self.s_upconv2(s_up3)
        s_up2 = torch.cat([s_up2, s2], dim=1)
        s_up2 = self.s_upblock2(s_up2)

        s_up1 = self.s_upconv1(s_up2)
        s_up1 = torch.cat([s_up1, s1], dim=1)
        s_up1 = self.s_upblock1(s_up1)

        # K-space domain branch
        k_in = self.fft_module(x)

        k1 = self.k_conv1(k_in)
        k2 = self.k_conv2(self.k_pool1(k1))
        k3 = self.k_conv3(self.k_pool2(k2))

        k_bottleneck = self.k_bottleneck(self.k_pool3(k3))

        k_up3 = self.k_upconv3(k_bottleneck)
        k_up3 = torch.cat([k_up3, k3], dim=1)
        k_up3 = self.k_upblock3(k_up3)

        k_up2 = self.k_upconv2(k_up3)
        k_up2 = torch.cat([k_up2, k2], dim=1)
        k_up2 = self.k_upblock2(k_up2)

        k_up1 = self.k_upconv1(k_up2)
        k_up1 = torch.cat([k_up1, k1], dim=1)
        k_up1 = self.k_upblock1(k_up1)

        # Merge branches
        merged = torch.cat([s_up1, k_up1], dim=1)
        merged = self.merge_conv(merged)

        # Final output
        output = self.final(merged)
        return {
            'output': output,
            'k_space_features': k_up1,  # For visualization/analysis
            'spatial_features': s_up1    # For visualization/analysis
        }


# 4. Sequential Hybrid Domain Model
'''
Inspired by the work of: 
A Hybrid Frequency-domain/Image-domain Deep
Network for Magnetic Resonance Image
Reconstruction
Roberto Souza, Member, IEEE, and Richard Frayne
'''


class SequentialHybridModel2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, depth=4, use_fusion_skips=False):
        super(SequentialHybridModel2, self).__init__()
        self.use_fusion_skips = use_fusion_skips

        # FFT and IFFT modules for domain conversion
        self.fft_module = FFTModule()
        self.ifft_module = IFFTModule()

        # Stage 1: Frequency Domain U-Net
        self.freq_down_path = nn.ModuleList()
        self.freq_up_path = nn.ModuleList()

        prev_channels = in_channels * 2  # magnitude + phase
        for i in range(depth):
            current_channels = base_filters * (2 ** i)
            self.freq_down_path.append(
                UNetConvBlock(prev_channels, current_channels))
            prev_channels = current_channels

        self.freq_bottom = UNetConvBlock(
            prev_channels, base_filters * (2 ** depth))

        prev_channels = base_filters * (2 ** depth)
        for i in reversed(range(depth)):
            current_channels = base_filters * (2 ** i)
            self.freq_up_path.append(UNetUpBlock(
                prev_channels, current_channels))
            prev_channels = current_channels

        self.freq_final = nn.Conv2d(
            prev_channels, in_channels * 2, kernel_size=1)

        # Optional bridge from frequency to spatial domain
        self.bridge_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Optional fusion blocks if using intermediate skip connections from frequency
        if self.use_fusion_skips:
            self.fusion_blocks = nn.ModuleList([
                nn.Conv2d(base_filters * (2 ** i),
                          base_filters * (2 ** i), kernel_size=1)
                for i in range(depth)
            ])

        # Stage 2: Spatial Domain U-Net
        self.spatial_down_path = nn.ModuleList()
        self.spatial_up_path = nn.ModuleList()

        prev_channels = in_channels
        for i in range(depth):
            current_channels = base_filters * (2 ** i)
            self.spatial_down_path.append(
                UNetConvBlock(prev_channels, current_channels))
            prev_channels = current_channels

        self.spatial_bottom = UNetConvBlock(
            prev_channels, base_filters * (2 ** depth))

        prev_channels = base_filters * (2 ** depth)
        for i in reversed(range(depth)):
            current_channels = base_filters * (2 ** i)
            self.spatial_up_path.append(
                UNetUpBlock(prev_channels, current_channels))
            prev_channels = current_channels

        self.spatial_final = nn.Conv2d(
            prev_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # === Stage 1: Frequency Domain ===
        x_kspace = self.fft_module(x)

        freq_blocks = []
        freq_x = x_kspace
        for i, down in enumerate(self.freq_down_path):
            freq_x = down(freq_x)
            freq_blocks.append(freq_x)
            if i < len(self.freq_down_path) - 1:
                freq_x = F.max_pool2d(freq_x, 2)

        freq_x = self.freq_bottom(freq_x)

        for i, up in enumerate(self.freq_up_path):
            freq_x = up(freq_x, freq_blocks[-i - 1])

        freq_x = self.freq_final(freq_x)

        # === Convert back to spatial domain ===
        initial_reconstruction = self.ifft_module(freq_x)

        # === Bridge connection to spatial domain ===
        spatial_x = x + self.bridge_conv(initial_reconstruction)

        # === Stage 2: Spatial Domain ===
        spatial_blocks = []
        for i, down in enumerate(self.spatial_down_path):
            spatial_x = down(spatial_x)

            # Optional fusion from frequency domain
            if self.use_fusion_skips:
                freq_feat = self.fusion_blocks[i](freq_blocks[i])
                spatial_x = spatial_x + \
                    F.interpolate(freq_feat, size=spatial_x.shape[-2:])

            spatial_blocks.append(spatial_x)
            if i < len(self.spatial_down_path) - 1:
                spatial_x = F.max_pool2d(spatial_x, 2)

        spatial_x = self.spatial_bottom(spatial_x)

        for i, up in enumerate(self.spatial_up_path):
            spatial_x = up(spatial_x, spatial_blocks[-i - 1])

        final_reconstruction = self.spatial_final(spatial_x)

        return {
            # residual connection to original input
            'output': self.sigmoid(final_reconstruction + x),
            'k_space_reconstructed': freq_x,
            'initial_reconstruction': initial_reconstruction
        }


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=100, patience=20, model_name="model", combined_loss=False):
    """
    Train the PyTorch model.

    Parameters:yes 
    - model: PyTorch model
    - train_loader, val_loader: Data loaders
    - criterion: Loss function
    - optimizer: Optimizer
    - device: Device to run on (cuda/cpu)
    - num_epochs: Maximum number of epochs
    - patience: Early stopping patience
    - model_name: Name for saving the model

    Returns:
    - model: Trained model
    - history: Training history
    """
    model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }

    best_val_loss = float('inf')
    no_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            outputs_dict = model(inputs)
            outputs = outputs_dict['output']

            if combined_loss == True:
                loss = 0.3 * criterion(outputs_dict['initial_reconstruction'], targets) + \
                    0.7 * criterion(outputs, targets)

            else:
                loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs.size(0)
            train_mae += F.l1_loss(outputs, targets).item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_mae = train_mae / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs_dict = model(inputs)
                outputs = outputs_dict['output']

                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_mae += F.l1_loss(outputs, targets).item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_mae = val_mae / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')

        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), f'best_{model_name}.pt')
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history

# Evaluation metrics


def compute_metrics(model, data_loader, device):
    """
    Compute various image quality metrics.

    Parameters:
    - model: Trained PyTorch model
    - data_loader: Test data loader
    - device: Device to run on (cuda/cpu)

    Returns:
    - metrics: Dictionary of metrics
    - test_input: Sample of test inputs
    - test_output: Corresponding model outputs
    - test_target: Corresponding ground truth
    """
    model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    ssim_sum = 0.0
    num_samples = 0

    test_input = []
    test_output = []
    test_target = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_dict = model(inputs)
            outputs = outputs_dict['output']

            # Move tensors to CPU for numpy operations
            inputs_np = inputs.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Store some samples for visualization
            if len(test_input) < 5:  # Store up to 5 samples
                test_input.append(inputs_np)
                test_output.append(outputs_np)
                test_target.append(targets_np)

            # MSE
            mse = np.mean((outputs_np - targets_np) ** 2)
            mse_sum += mse * inputs.size(0)

            # MAE
            mae = np.mean(np.abs(outputs_np - targets_np))
            mae_sum += mae * inputs.size(0)

            # SSIM (simplified)
            for i in range(inputs.size(0)):
                ssim_val = ssim(targets_np[i, 0], outputs_np[i, 0])
                ssim_sum += ssim_val
                num_samples += 1

    # Combine test samples
    test_input = np.concatenate(test_input, axis=0)
    test_output = np.concatenate(test_output, axis=0)
    test_target = np.concatenate(test_target, axis=0)

    # Calculate average metrics
    avg_mse = mse_sum / num_samples
    avg_mae = mae_sum / num_samples
    avg_ssim = ssim_sum / num_samples

    # PSNR
    max_pixel = 1.0
    psnr = 10.0 * np.log10((max_pixel ** 2) / (avg_mse + 1e-10))

    return {
        'MSE': avg_mse,
        'MAE': avg_mae,
        'PSNR': psnr,
        'SSIM': avg_ssim
    }, test_input, test_output, test_target

# SSIM calculation


def ssim(img1, img2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    Simplified version.
    """
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim_score = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
        ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_score

# Visualization function


def visualize_results(test_input, test_output, test_target, num_samples=5, title=None):
    """
    Visualize the reconstruction results.

    Parameters:
    - test_input: Undersampled test images
    - test_output: Reconstructed test images by the model
    - test_target: Ground truth test images
    - num_samples: Number of samples to visualize
    - title: Optional title for the plot
    """
    plt.figure(figsize=(15, 5*num_samples))

    if title:
        plt.suptitle(title, fontsize=16)

    for i in range(min(num_samples, test_input.shape[0])):
        plt.subplot(num_samples, 3, i*3+1)
        plt.imshow(test_input[i, 0], cmap='gray')
        plt.title(f'Undersampled {i+1}')
        plt.colorbar()

        plt.subplot(num_samples, 3, i*3+2)
        plt.imshow(test_output[i, 0], cmap='gray')
        plt.title(f'Reconstructed {i+1}')
        plt.colorbar()

        plt.subplot(num_samples, 3, i*3+3)
        plt.imshow(test_target[i, 0], cmap='gray')
        plt.title(f'Ground Truth {i+1}')
        plt.colorbar()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Plot training history


def plot_history(history, title=None):
    """
    Plot the training history.

    Parameters:
    - history: Dictionary with training and validation metrics
    - title: Optional title for the plot
    """
    plt.figure(figsize=(12, 5))

    if title:
        plt.suptitle(title, fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train')
    plt.plot(history['val_mae'], label='Validation')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

# Compare models


def compare_models(models_metrics, models_names):
    """
    Compare metrics across different models with improved visualization.

    Parameters:
    - models_metrics: List of metric dictionaries from different models
    - models_names: List of model names for labeling
    """
    # Collect all metrics
    metrics = list(models_metrics[0].keys())

    # Create figure with subplots - increased figure size
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 8))

    # Set a consistent color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    # Plot each metric as a bar chart
    for i, metric in enumerate(metrics):
        values = [m[metric] for m in models_metrics]

        # Use consistent bar width with spacing
        bar_positions = range(len(models_names))
        axes[i].bar(bar_positions, values, width=0.7,
                    color=colors[:len(models_names)])

        # Set metric name as title with larger font
        axes[i].set_title(metric, fontsize=16, fontweight='bold', pad=20)

        # Set x-axis labels with angled text for better readability
        axes[i].set_xticks(bar_positions)
        axes[i].set_xticklabels(
            models_names, rotation=45, ha='right', fontsize=12)

        # Add grid lines for better readability
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        # Add actual values on top of bars with improved positioning
        for j, v in enumerate(values):
            # Format numbers based on their magnitude for cleaner display
            if v < 0.01:
                value_text = f"{v:.6f}"
            elif v < 1:
                value_text = f"{v:.4f}"
            else:
                value_text = f"{v:.2f}"

            axes[i].text(j, v + 0.02 * max(values), value_text,
                         ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Adjust y-axis to leave room for the value labels
        y_limit = axes[i].get_ylim()
        axes[i].set_ylim(0, y_limit[1] * 1.15)

    # Add a main title to the figure
    fig.suptitle('Model Performance Comparison',
                 fontsize=20, fontweight='bold', y=0.98)

    # Adjust the layout to prevent overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig  # Return the figure object so it can be saved if needed
