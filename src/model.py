"""
3D Sparse U-Net for LiDAR Weather Hallucination.
Uses simplified sparse 3D convolution (voxel-based).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VoxelEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VoxelDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd dims
        diff = [s - x.size(i + 2) for i, s in enumerate(skip.shape[2:])]
        x = F.pad(x, [diff[2] // 2, diff[2] - diff[2] // 2,
                       diff[1] // 2, diff[1] - diff[1] // 2,
                       diff[0] // 2, diff[0] - diff[0] // 2])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class LiDARWeatherHallucinationNet(nn.Module):
    """
    3D U-Net that takes a corrupted (weather-affected) voxelized point cloud
    and reconstructs the clean point cloud occupancy grid.

    Input:  (B, 1, D, H, W) voxelized occupancy of corrupted LiDAR
    Output: (B, 1, D, H, W) reconstructed clean occupancy probability
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = VoxelEncoderBlock(1, c)        # (D, H, W)
        self.enc2 = VoxelEncoderBlock(c, c * 2)    # (D/2, H/2, W/2)
        self.enc3 = VoxelEncoderBlock(c * 2, c * 4)
        self.enc4 = VoxelEncoderBlock(c * 4, c * 8)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(c * 8, c * 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(c * 16),
            nn.LeakyReLU(0.2),
            nn.Conv3d(c * 16, c * 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(c * 16),
            nn.LeakyReLU(0.2),
        )

        # Decoder
        self.dec4 = VoxelDecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = VoxelDecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = VoxelDecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = VoxelDecoderBlock(c * 2, c, c)

        # Output head: occupancy probability
        self.out = nn.Sequential(
            nn.Conv3d(c, c // 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(c // 2, 1, 1),
            nn.Sigmoid(),  # occupancy probability per voxel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, D, H, W) corrupted voxel grid."""
        # Encoder with skip connections
        e1 = self.enc1(x)    # (B, c, D, H, W)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.out(d1)


# ── Loss Functions ──────────────────────────────────────────────────────────────

class ChamferDistanceLoss(nn.Module):
    """
    Chamfer distance between two point clouds.
    Approximated on voxel grids: penalizes occupancy mismatch.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Symmetric binary cross-entropy on voxel occupancy
        loss = F.binary_cross_entropy(pred, target, reduction="mean")
        return loss


class DensityRegularizationLoss(nn.Module):
    """Penalizes unrealistic occupancy density in reconstruction."""

    def __init__(self, target_density: float = 0.05):
        super().__init__()
        self.target = target_density

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        density = pred.mean()
        return F.relu(density - self.target * 2) + F.relu(self.target * 0.5 - density)


class CombinedHallucinationLoss(nn.Module):
    """Chamfer + Density + Focal loss for class imbalance (empty >> occupied voxels)."""

    def __init__(self, focal_alpha: float = 0.75, focal_gamma: float = 2.0,
                 density_weight: float = 0.1):
        super().__init__()
        self.cd = ChamferDistanceLoss()
        self.dens = DensityRegularizationLoss()
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.density_weight = density_weight

    def _focal_bce(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        loss_focal = self._focal_bce(pred, target)
        loss_dens = self.dens(pred)
        total = loss_focal + self.density_weight * loss_dens
        return {
            "total": total,
            "focal": loss_focal.item(),
            "density": loss_dens.item(),
        }
