import torch

import fishyrl.losses


def test_mse_loss():
    prior = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    posterior = torch.tensor([[2.0, 3.0, 4.0], [4.5, 5.5, 6.5]], requires_grad=True)
    loss = fishyrl.losses.mse_loss(prior, posterior)
    assert loss.shape == (2,)
    assert loss.requires_grad
    assert torch.allclose(loss, torch.tensor([1.0, 0.25]))
