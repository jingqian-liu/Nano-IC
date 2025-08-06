import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedChannelSpatialAttention(nn.Module):
    """Hybrid attention with medium kernel size"""
    def __init__(self, in_channels):
        super().__init__()
        # Keep Model A's channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, max(in_channels//8, 4), 1),  # Intermediate compression
            nn.ReLU(),
            nn.Conv3d(max(in_channels//8, 4), in_channels, 1),
            nn.Sigmoid()
        )
        # Compromise spatial kernel
        self.sa = nn.Sequential(
            nn.Conv3d(1, 1, 5, padding=2, bias=False),  # Middle ground 5×5×5
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.ca(x)
        sa = torch.mean(x, dim=1, keepdim=True)
        sa = self.sa(sa)
        return x * ca * sa

class BalancedDilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        # Use Model B's base but with adjusted channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3,
                              padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.attn = BalancedChannelSpatialAttention(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attn(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)



class GlobalContextPredictor(nn.Module):
    def __init__(self, input_nc=1):
        super().__init__()
        self.feature_net = nn.Sequential(
            # Intermediate channel counts
            BalancedDilatedResBlock(input_nc, 12, dilation=2),
            nn.MaxPool3d(2),

            BalancedDilatedResBlock(12, 24, dilation=3),  # Mix dilation rates
            nn.AdaptiveAvgPool3d(2),

            nn.Conv3d(24, 48, 2),  # Intermediate global mixer
            nn.ReLU(),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        x = self.feature_net(x)
        return self.regressor(x).squeeze(-1)
