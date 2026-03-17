import torch
import torch.nn as nn


class CardsCNN(nn.Module):
    """
    CNN for processing card information.
    Expects input of shape [B, 4, 4, 13] representing cards in a 2D grid.
    """

    net: nn.Sequential
    proj: nn.Sequential

    def __init__(self, out_dim: int = 256) -> None:
        """
        Initializes the CardsCNN.

        Args:
            out_dim (int): The dimension of the output vector.
        """
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=4, out_channels=32, kernel_size=3, padding=1
            ),  # [B,32,4,13]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B,64,4,13]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.15),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B,128,4,13]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            # Block 4 (deeper representation)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # [B,128,4,13]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),  # [B,128,1,1]
        )

        self.proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CardsCNN.

        Args:
            x (torch.Tensor): [B, 4, 4, 13] Input tensor.

        Returns:
            torch.Tensor: [B, out_dim] Output tensor.
        """
        x = self.net(x)
        x = x.flatten(1)  # [B,128]
        x = self.proj(x)  # [B,out_dim]
        return x
