"""Utility loss functions for reinforcement learning agents."""

import scipy.optimize
import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor, dims: int = 1) -> torch.Tensor:
    """Compute the mean squared error between the predicted and target distributions on the final dimension.

    :param pred: The predicted distribution.
    :type pred: torch.Tensor
    :param target: The target distribution.
    :type target: torch.Tensor
    :param dims: The number of final dimensions to compute the loss over. (Default: ``1``)
    :type dims: int
    :return: The mean squared error loss.
    :rtype: torch.Tensor

    """
    return (pred - target).square().mean(dim=[-i for i in range(1, dims + 1)])


def hungarian_loss(pred: torch.Tensor, target: torch.Tensor, dims: int = 2) -> torch.Tensor:
    """Compute the Hungarian loss between the predicted and target distributions on the final dimension.

    This loss function is not vectorized and is computationally expensive.

    :param pred: The predicted distribution, of shape (batch_dim, entities, features).
    :type pred: torch.Tensor
    :param target: The target distribution, of shape (batch_dim, entities, features).
    :type target: torch.Tensor
    :param dims: The number of final dimensions to compute the loss over. (Default: ``1``)
    :type dims: int
    :return: The Hungarian loss.
    :rtype: torch.Tensor

    """
    # Compute pairwise distances
    pairwise_distances = torch.cdist(pred, target, p=2)

    # Compute optimal assignment using the Hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(pairwise_distances.detach().cpu().numpy())

    # Compute MSE loss based on optimal assignment
    ordered_pred = pred[row_ind]
    ordered_target = target[col_ind]
    return mse_loss(ordered_pred, ordered_target, dims=dims)


def attention_reconstruction_loss(
    entities_pred: torch.Tensor,
    entities_target: torch.Tensor,
    existence_logits: torch.Tensor,
    existence_target: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the attention reconstruction loss between the predicted and target distributions.

    :param entities_pred: The predicted entity features, of shape (batch_dim, entities, features).
    :type entities_pred: torch.Tensor
    :param entities_target: The target entity features, of shape (batch_dim, entities, features).
    :type entities_target: torch.Tensor
    :param existence_logits: The predicted entity existence logits, of shape (batch_dim, entities).
    :type existence_logits: torch.Tensor
    :param existence_target: The target entity existence probabilities (0 or 1), of shape (batch_dim, entities).
        If not provided, it will be computed as 1 for entities with any non-zero features.
    :type existence_target: torch.Tensor
    :return: A tuple containing the attention reconstruction loss and the existence loss.
    :rtype: tuple[torch.Tensor, torch.Tensor]

    """
    # Compute existence target if not provided
    if existence_target is None:
        existence_target = (entities_target.detach().abs().sum(dim=-1) > 0).float()

    # Zero pad existence target to match existence logits if needed
    if existence_target.shape[-1] < existence_logits.shape[-1]:
        pad_size = existence_logits.shape[-1] - existence_target.shape[-1]
        existence_target = F.pad(existence_target, (0, pad_size), value=0)  # Pad bottom of last dim

    # Get batch dims and flatten predictions
    batch_dims = entities_pred.shape[:-2]
    entities_pred_flat = entities_pred.view(-1, *entities_pred.shape[-2:])
    entities_target_flat = entities_target.view(-1, *entities_target.shape[-2:])
    existence_target_flat = existence_target.view(-1, existence_target.shape[-1])

    # Compute entity reconstruction loss
    reconstruction_loss = torch.zeros(entities_pred_flat.shape[0], device=entities_pred.device)
    for i in range(entities_pred_flat.shape[0]):
        # Extract pred and target based on actual number of entities
        num_entities = existence_target_flat[i].sum().int().item()
        pred = entities_pred_flat[i][:num_entities]
        target = entities_target_flat[i][:num_entities]
        loss = hungarian_loss(pred, target, dims=2)
        reconstruction_loss[i] = loss

    # Reshape reconstruction loss back to batch dimensions
    reconstruction_loss = reconstruction_loss.view(*batch_dims)

    # Compute binary cross-entropy loss for existence probabilities
    existence_loss = F.binary_cross_entropy_with_logits(
        existence_logits, existence_target, reduction='none').mean(dim=-1)

    return reconstruction_loss, existence_loss
