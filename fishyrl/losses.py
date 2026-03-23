"""Utility loss functions for reinforcement learning agents."""

import torch


def MSELoss(prior: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
    """Compute the mean squared error between the prior and posterior distributions on the final dimension.

    :param prior: The prior distribution.
    :type prior: torch.Tensor
    :param posterior: The posterior distribution.
    :type posterior: torch.Tensor
    :return: The mean squared error loss.
    :rtype: torch.Tensor

    """
    return (prior - posterior).square().mean(dim=-1)
