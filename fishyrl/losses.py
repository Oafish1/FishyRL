"""Utility loss functions for reinforcement learning agents."""

import torch


def mse_loss(prior: torch.Tensor, posterior: torch.Tensor, dims: int = 1) -> torch.Tensor:
    """Compute the mean squared error between the prior and posterior distributions on the final dimension.

    :param prior: The prior distribution.
    :type prior: torch.Tensor
    :param posterior: The posterior distribution.
    :type posterior: torch.Tensor
    :param dims: The number of final dimensions to compute the loss over. (Default: ``1``)
    :type dims: int
    :return: The mean squared error loss.
    :rtype: torch.Tensor

    """
    return (prior - posterior).square().mean(dim=[-i for i in range(1, dims + 1)])
