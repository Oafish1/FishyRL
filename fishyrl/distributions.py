"""Utility distributions for reinforcement learning agents."""

import torch
import torch.nn.functional as F


class TwoHot:
    """Two hot distribution as described in Dreamer-V3.

    Takes a binned input tensor and computes log probabilities as if the queried value
    is a linear interpolation between two bins.

    """
    def __init__(
        self,
        logits: torch.Tensor,
        bins: int = None,
        low: float = -20.0,
        high: float = 20.0,
    ) -> None:
        """Create a TwoHot distribution from the input logits.

        :param logits: The input logits to create the distribution from, of shape (..., bins).
        :type logits: torch.Tensor
        :param bins: The number of bins for the distribution. Defaults to the final dimension of `logits`.
        :type bins: int
        :param low: The lower bound of the distribution. (Default: ``-20.0``)
        :type low: float
        :param high: The upper bound of the distribution. (Default: ``20.0``)
        :type high: float

        """
        # Parameters
        self._logits = logits
        self._probs = torch.softmax(logits, dim=-1)
        self._bins = bins if bins is not None else logits.shape[-1]
        self._low = low
        self._high = high

        # Compute bin values
        self._bin_values = torch.linspace(low, high, self._bins, device=logits.device)

    @property
    def mean(self) -> torch.Tensor:
        """Compute the mean of the distribution as a weighted sum of bin values.

        :return: The mean of the distribution.
        :rtype: torch.Tensor

        """
        return symexp((self._probs * self._bin_values).sum(dim=-1))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the given value under the distribution.

        :param value: The value to compute the log probability for, can be of any shape.
        :type value: torch.Tensor
        :return: The log probability of the value under the distribution, of the same shape as `value`.
        :rtype: torch.Tensor

        """
        # Reference: https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/utils/distribution.py#L253
        # Apply symlog transformation to the value
        value = symlog(value)

        # Get left and right bins
        left_bin = (value.unsqueeze(-1) >= self._bin_values).sum(dim=-1) - 1
        right_bin = left_bin + 1
        left_bin, right_bin = torch.clamp(left_bin, 0, self._bins - 1), torch.clamp(right_bin, 0, self._bins - 1)

        # Get weights for left and right bins
        bin_range = self._bin_values[right_bin] - self._bin_values[left_bin]
        left_weight = (self._bin_values[right_bin] - value) / bin_range
        right_weight = 1 - left_weight

        # Create target and compute log probability
        # NOTE: This unsqueeze might be suboptimal, and could be maintained from the beginning
        target = F.one_hot(left_bin, self._bins) * left_weight.unsqueeze(-1) + F.one_hot(right_bin, self._bins) * right_weight.unsqueeze(-1)
        log_dist = self._logits - self._logits.logsumexp(dim=-1, keepdims=True)  # logit distance from max
        return (target * log_dist).sum(dim=-1)


def uniform_mix(logits: torch.Tensor, ratio: float = .01) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix the input logits with a uniform distribution on the final dimension.

    :param logits: The input logits to mix, of shape (..., num_classes).
    :type logits: torch.Tensor
    :param ratio: The ratio of uniform distribution to mix with the input logits.
    :type ratio: float
    :return: The mixed logits, of shape (..., num_classes).
    :rtype: torch.Tensor

    """
    # Compute probabilities from logits and mix with uniform distribution
    probs = torch.softmax(logits, dim=-1)
    probs = (1 - ratio) * probs + ratio * (1 / probs.shape[-1])

    # Return probabilities and logits
    # NOTE: `torch.distributions.utils.probs_to_logits` is just log with clamping
    return torch.distributions.utils.probs_to_logits(probs), probs


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Apply the symmetric logarithm transformation to the input tensor.

    :param x: The input tensor to transform.
    :type x: torch.Tensor
    :return: The transformed tensor.
    :rtype: torch.Tensor

    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Apply the symmetric exponential transformation to the input tensor.

    :param x: The input tensor to transform.
    :type x: torch.Tensor
    :return: The transformed tensor.
    :rtype: torch.Tensor

    """
    return torch.sign(x) * (torch.expm1(torch.abs(x)))
