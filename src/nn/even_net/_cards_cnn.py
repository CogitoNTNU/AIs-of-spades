import torch
import torch.nn as nn


class CardsCNN(nn.Module):
    """
    CNN for processing card information.
    Expects input of shape [B, 4, 4, 13] representing cards in a 2D grid.
    """

    net: nn.Sequential
    proj: nn.Linear

    def __init__(self, out_dim: int = 128) -> None:
        """
        Initializes the CardsCNN.

        Args:
            out_dim (int): The dimension of the output vector.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=4, out_channels=16, kernel_size=3, padding=1
            ),  # [B,16,4,13]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B,32,4,13]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B,32,1,1]
        )
        self.proj = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CardsCNN.

        Args:
            x (torch.Tensor): [B, 4, 4, 13] Input tensor.

        Returns:
            torch.Tensor: [B, out_dim] Output tensor.
        """
        x = self.net(x)
        x = x.flatten(1)  # [B,32]
        x = self.proj(x)  # [B,out_dim]
        return x
