"""Utility models and layers for construction of reinforcement learning agents."""

import math
from typing import Any

import torch
import torch.nn as nn


class ChannelNorm(nn.LayerNorm):
    """Layer normalization across the channel dimension."""
    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        """Initialize the ChannelNorm layer.

        :param args: Positional arguments for the LayerNorm.
        :type args: list[Any]
        :param kwargs: Keyword arguments for the LayerNorm.
        :type kwargs: dict[str, Any]

        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization across the channel dimension.

        Swaps the channel dimension to the end, applies layer normalization, then swaps it back.

        :param x: The input tensor of shape (batch_size, channels, height, width).
        :type x: torch.Tensor
        :return: The normalized tensor of the same shape.
        :rtype: torch.Tensor

        """
        # Permute channels to last dimension
        x = x.permute(0, 2, 3, 1)
        # Apply layer normalization
        x = super().forward(x)
        # Reverse permutation
        return x.permute(0, 3, 1, 2)


class CNNEncoder(nn.Module):
    """CNN encoder for processing image observations.

    Processes a (generally) 64x64 image to 4x4 using [2, 4, 8, 16] channels and interleaved
    layer normalization and SiLU activations.
    """
    def __init__(self, input_channels: int, image_size: tuple[int, int]) -> None:
        """Initialize the CNN encoder.

        :param input_channels: The number of input channels in the image.
        :type input_channels: int
        :param image_size: The size of the input image.
        :type image_size: tuple[int, int]

        """
        super().__init__()

        # Create model in blocks
        layers = []
        num_blocks = 4
        hidden_channels = [input_channels] + [2 * 2 ** i for i in range(num_blocks)]
        for i in range(num_blocks):
            layers += [
                # Layer, dropout, norm, activation
                nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
                ChannelNorm(hidden_channels[i + 1], eps=1e-3),
                nn.SiLU(),
            ]
        layers += [nn.Flatten()]
        self._output_dim = hidden_channels[-1] * math.prod(image_size) // 4 ** num_blocks
        self._model = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        """Output dimension of the CNN encoder.

        :type: int

        """
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the CNN encoder.

        :param x: The input tensor of shape (batch_size, channels, height, width).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_size, output_dim, height, width).
        :rtype: torch.Tensor

        """
        # NOTE: Difference here with SheepRL, where batch dims are flattened during forward
        return self._model(x)
