"""Utility distributions for reinforcement learning agents."""

import torch
import torch.nn.functional as F


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
        pre_func: callable = symlog,
        post_func: callable = symexp,
        tensor_log_prob: bool = False,
        event_dims: int = 1,
        eps: float = 1e-5,
    ) -> None:
        """Create a TwoHot distribution from the input logits.

        :param logits: The input logits to create the distribution from, of shape (..., bins).
        :type logits: torch.Tensor
        :param bins: The number of bins for the distribution. Defaults to the final dimension of ``logits``.
        :type bins: int
        :param low: The lower bound of the distribution. (Default: ``-20.0``)
        :type low: float
        :param high: The upper bound of the distribution. (Default: ``20.0``)
        :type high: float
        :param pre_func: A function to apply to the input value before computing log probabilities. (Default: ``symlog``)
        :type pre_func: callable
        :param post_func: A function to apply to the mean of the distribution before returning. (Default: ``symexp``)
        :type post_func: callable
        :param tensor_log_prob: Whether to compute log probabilities directly from the input tensor instead of constructing a target vector. (Default: ``False``)
        :type tensor_log_prob: bool
        :param event_dims: The number of final dimensions to treat as event dimensions when combining log probabilities and entropy. (Default: ``1``)
        :type event_dims: int
        :param eps: A small value to add when computing entropy to avoid numerical issues. (Default: ``1e-8``)
        :type eps: float

        """
        # Parameters
        self._logits = logits
        self._probs = torch.softmax(logits, dim=-1)
        self._bins = bins if bins is not None else logits.shape[-1]
        self._low = low
        self._high = high
        self._pre_func = pre_func
        self._post_func = post_func
        self._event_dims = [-i for i in range(1, event_dims + 1)]
        self._tensor_log_prob = tensor_log_prob
        self._eps = eps

        # Compute bin values
        self._bin_values = torch.linspace(low, high, self._bins, device=logits.device)

        # Compute inter-bin probabilities for sampling
        self._inter_bin_probs = self._probs[..., :-1] + self._probs[..., 1:]
        self._inter_bin_probs = self._inter_bin_probs / self._inter_bin_probs.sum(dim=-1, keepdim=True)
        self._inter_bin_cumprobs = torch.cumsum(self._inter_bin_probs, dim=-1)
        self._inter_bin_cumprobs = torch.cat([torch.zeros_like(self._inter_bin_cumprobs[..., :1]), self._inter_bin_cumprobs], dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """Compute the mean of the distribution as a weighted sum of bin values.

        :return: The mean of the distribution.
        :rtype: torch.Tensor

        """
        return self._post_func((self._probs * self._bin_values).sum(dim=-1))

    def rsample(self) -> torch.Tensor:
        """Sample by randomly sampling two bins and interpolating. Still WIP, and might be numerically unstable.

        :return: A sample from the distribution, matching the shape of ``mean``.
        :rtype: torch.Tensor

        """
        # Sample a bin interval based on inter-bin probabilities, then generate a vector with 1s in each of the two bins and 0s elsewhere, with straight-through gradient.
        dist = torch.distributions.OneHotCategoricalStraightThrough(logits=self._inter_bin_probs)
        interval_mask = dist.rsample()
        left_bin_mask = torch.cat([interval_mask, torch.zeros_like(interval_mask[..., :1])], dim=-1)
        right_bin_mask = torch.cat([torch.zeros_like(interval_mask[..., :1]), interval_mask], dim=-1)

        # Get weights for left and right bins
        left_weight = self._probs * left_bin_mask
        right_weight = self._probs * right_bin_mask
        weight_sum = (left_weight + right_weight).sum(dim=-1, keepdim=True) + self._eps
        two_discrete = (left_weight + right_weight) / weight_sum

        # Add straight-through gradient
        two_discrete = two_discrete.detach() + (self._probs - self._probs.detach())

        return two_discrete

        # # Sample between bins according to an estimated PDF
        # uniform_sample = torch.rand(self._logits.shape[:-1], device=self._logits.device)
        # right_bin = (uniform_sample.unsqueeze(-1) > self._inter_bin_cumprobs).sum(dim=-1)
        # right_bin = right_bin.clamp(1, self._bins - 1)  # Clipping avoids selecting 0 for right bin, which causes OOB error due to numerical instability
        # left_bin = right_bin - 1

        # # Get weights for left and right bins
        # # TODO: Maybe clean up the gathering here and leave unsqueezed
        # bin_range = self._inter_bin_probs.gather(-1, left_bin.unsqueeze(-1)).squeeze(-1) + self._eps  # Add eps to avoid division by zero
        # left_weight = (uniform_sample - self._inter_bin_cumprobs.gather(-1, left_bin.unsqueeze(-1)).squeeze(-1)) / bin_range
        # right_weight = (self._inter_bin_cumprobs.gather(-1, right_bin.unsqueeze(-1)).squeeze(-1) - uniform_sample) / bin_range
        # left_weight, right_weight = left_weight.clamp(0, 1), right_weight.clamp(0, 1)  # Clamp to avoid numerical issues

        # # Compute the two hot vectors with straight-through gradient
        # two_discrete = (
        #     F.one_hot(left_bin, num_classes=self._bins) * left_weight.unsqueeze(-1)
        #     + F.one_hot(right_bin, num_classes=self._bins) * right_weight.unsqueeze(-1))
        # two_discrete = two_discrete.detach() + (self._probs - self._probs.detach())

        # return two_discrete

        # # Sample two distinct bins according to the probabilities
        # dist = torch.distributions.OneHotCategoricalStraightThrough(logits=self._logits)
        # left_bin_mask = dist.rsample()  # Don't use these gradients for now
        # right_bin_mask = dist.rsample()

        # # Get weights for left and right bins
        # left_weight = self._probs * left_bin_mask
        # right_weight = self._probs * right_bin_mask
        # weight_sum = (left_weight + right_weight).sum(dim=-1, keepdim=True) + self._eps
        # two_discrete = (left_weight + right_weight) / weight_sum

        # # Add straight-through gradient
        # two_discrete = two_discrete.detach() + (self._probs - self._probs.detach())

        # return two_discrete

        # # Sample two distinct bins according to the probabilities
        # dist = torch.distributions.OneHotCategoricalStraightThrough(logits=self._logits)
        # left_bin_mask = dist.rsample()
        # right_bin_mask = dist.rsample()

        # # Get weights for left and right bins by using an isolated softmax, since summing probabilities leads to exploding gradients.
        # # left_weight = self._probs * left_bin_mask
        # # right_weight = self._probs * right_bin_mask
        # left_weight = torch.exp(self._logits * left_bin_mask)
        # right_weight = torch.exp(self._logits * right_bin_mask)
        # weight_sum = (left_weight + right_weight).sum(dim=-1, keepdim=True)
        # sample = (left_weight / weight_sum) * self._bin_values + (right_weight / weight_sum) * self._bin_values
        # sample = sample.sum(dim=-1)  # Sum over bins to get the final sample
        # return self._post_func(sample)

        # # Sample from between bin intervals according to an estimated PDF using the reparameterization trick.
        # # Take a uniform sample from 0-1 and find the corresponding inter-bin index
        # uniform_sample = torch.rand(self._logits.shape[:-1], device=self._logits.device)
        # right_bin = (uniform_sample.unsqueeze(-1) > self._inter_bin_cumprobs).sum(dim=-1)
        # right_bin = right_bin.clamp(1, self._bins - 1)  # Clipping avoids selecting 0 for right bin, which causes OOB error due to numerical instability
        # left_bin = right_bin - 1

        # # Get weights for left and right bins
        # # TODO: Maybe clean up the gathering here and leave unsqueezed
        # bin_range = self._inter_bin_probs.gather(-1, left_bin.unsqueeze(-1)).squeeze(-1) + self._eps  # Add eps to avoid division by zero
        # left_weight = (uniform_sample - self._inter_bin_cumprobs.gather(-1, left_bin.unsqueeze(-1)).squeeze(-1)) / bin_range
        # right_weight = (self._inter_bin_cumprobs.gather(-1, right_bin.unsqueeze(-1)).squeeze(-1) - uniform_sample) / bin_range
        # left_weight, right_weight = left_weight.clamp(0, 1), right_weight.clamp(0, 1)  # Clamp to avoid numerical issues

        # # Compute the sample as a weighted sum of the left and right bin values
        # sample = self._bin_values[left_bin] * left_weight + self._bin_values[right_bin] * right_weight
        # return self._post_func(sample)  # .unsqueeze(-1)

        # # Compute weighted sample from the two bins
        # # bin_range = self._bin_values[right_bin] - self._bin_values[left_bin]
        # # return self._post_func(self._bin_values[left_bin] + torch.rand_like(left_bin, dtype=torch.float) * bin_range)
        # left_weight = self._probs.gather(-1, left_bin)
        # right_weight = self._probs.gather(-1, right_bin)
        # weight_sum = left_weight + right_weight
        # sample = (left_weight / weight_sum) * self._bin_values[left_bin] + (right_weight / weight_sum) * self._bin_values[right_bin]
        # return self._post_func(sample)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the given value under the distribution.

        :param value: The value to compute the log probability for, can be of any shape.
        :type value: torch.Tensor
        :return: The log probability of the value under the distribution, of the same shape as ``value``.
        :rtype: torch.Tensor

        """
        # Construct target vector if not provided
        if not self._tensor_log_prob:
            # Reference: https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/utils/distribution.py#L253
            # Apply `pre_func` transformation to the value
            value = self._pre_func(value)

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

        # Otherwise, use the provided value as the target directly
        else:
            target = value

        # Compute log probability
        log_dist = self._logits - self._logits.logsumexp(dim=-1, keepdims=True)  # logit distance from max
        return (target * log_dist).sum(dim=self._event_dims)

    def entropy(self) -> torch.Tensor:
        """Compute the entropy of the distribution.

        :return: The entropy of the distribution, of the same shape as the mean without extra event dimensions.
        :rtype: torch.Tensor

        """
        return - (self._probs * torch.log(self._probs + self._eps)).sum(dim=self._event_dims)


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


def identity(x: torch.Tensor) -> torch.Tensor:
    """Identity function, returns the input tensor unchanged.

    :param x: The input tensor.
    :type x: torch.Tensor
    :return: The same input tensor.
    :rtype: torch.Tensor

    """
    return x
