"""Utilities for state management and common operations."""

import numpy as np
import torch
import torch.nn as nn


class MovingMinMaxScaler(nn.Module):
    """Moving percentile-based min-max scaler for normalizing inputs."""
    def __init__(
        self,
        beta: float = .99,
        frac_low: float = .05,
        frac_high: float = .95,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the `MovingMinMaxScaler`.

        :param beta: The decay rate for the moving min and max. (Default: ``0.99``)
        :param eps: Minimal value for the computed high-low range. (Default: ``1e-8``)
        :param frac_low: The lower percentile for scaling. (Default: ``0.05``)
        :param frac_high: The upper percentile for scaling. (Default: ``0.95``)

        """
        super().__init__()

        # Parameters
        self._beta = beta
        self._frac_low = frac_low
        self._frac_high = frac_high
        self._eps = torch.tensor(eps, dtype=torch.get_default_dtype())

        # Initialize low and high buffers
        self.register_buffer('_low', torch.zeros((), dtype=torch.get_default_dtype()))
        self.register_buffer('_high', torch.zeros((), dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update and return the low and range estimates.

        :param x: The input tensor to use while updating the estimates.
        :return: The normalized tensor.
        :rtype: torch.Tensor

        """
        # Detatch input to avoid memory leaks
        x = x.detach()

        # Update low and high estimates
        low = torch.quantile(x, self._frac_low)
        high = torch.quantile(x, self._frac_high)
        self._low = self._beta * self._low + (1 - self._beta) * low
        self._high = self._beta * self._high + (1 - self._beta) * high
        # self._low, self._high = self._low.detach(), self._high.detach()

        # Return low and range
        return self._low, torch.max(self._high - self._low, self._eps)


# Taken from SheepRL
# https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/algos/dreamer_v3/utils.py#L143
def init_weights(m):  # noqa
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# Taken from SheepRL
# https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/algos/dreamer_v3/utils.py#L170
def uniform_init_weights(given_scale):  # noqa
    def f(m):  # noqa
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f
