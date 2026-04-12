"""Utility action definitions for reinforcement learning agents."""

import enum
from abc import abstractmethod

import torch
from torch import nn

from . import distributions as frl_distributions
from . import utilities as frl_utilities


class Action(nn.Module):
    """Base class for actions."""
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """The number of input features for the action.

        :type: int

        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """The number of output features for the action.

        :type: int

        """
        pass

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """The number of actions.

        :type: int

        """
        pass

    @abstractmethod
    def simplify(self, x: torch.Tensor) -> torch.Tensor:
        """Simplify each action to a single value.

        :param x: The action(s) of shape (batch_dim, output_dim).
        :type x: torch.Tensor
        :return: The simplified action of shape (batch_dim).
        :rtype: torch.Tensor

        """
        pass

    @abstractmethod
    def construct(self, action: torch.Tensor) -> torch.Tensor:
        """Construct the full action from the simplified action.

        :param action: The simplified action of shape (batch_dim).
        :type action: torch.Tensor
        :return: The full action of shape (batch_dim, output_dim).
        :rtype: torch.Tensor

        """
        pass

    @abstractmethod
    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        """Sample an action from the logits.

        :param logits: The base logits of shape (batch_dim, input_dim).
        :type logits: torch.Tensor
        :return: The sampled action of shape (batch_dim, output_dim) and the corresponding distribution.
        :rtype: tuple[torch.Tensor, torch.distributions.Distribution]

        """
        pass


class ContinuousActions(Action):
    """Continuous action definition using ``torch.distributionsNormal``.

    Computed using mean ``tanh(mean)`` and std
    ``(std_max - std_min) * sigmoid(std + std_init) + std_min``.

    """
    def __init__(
            self,
            num_actions: int = 1,
            std_init: float = 2,
            std_min: float = .1,
            std_max: float = 1,
            clip: float = 0,
        ) -> None:
        """Initialize the action definition.

        :param num_actions: The number of actions to initialize.
        :type num_actions: int
        :param std_init: The initial standard deviation.
        :type std_init: float
        :param std_min: The minimum standard deviation.
        :type std_min: float
        :param std_max: The maximum standard deviation.
        :type std_max: float
        :param clip: The maximum absolute value of the action.
        :type clip: float

        """
        super().__init__()

        # Parameters
        self._num_actions = num_actions
        self._std_init = std_init
        self._std_min = std_min
        self._std_max = std_max
        self._clip = clip

    @property
    def input_dim(self) -> int:
        """The number of input features for the continuous action, equal to 2 * num_actions.

        :type: int

        """
        return 2 * self._num_actions

    @property
    def output_dim(self) -> int:
        """The number of output features for the continuous action, equal to num_actions.

        :type: int

        """
        return self._num_actions

    @property
    def num_actions(self) -> int:
        """The number of actions.

        :type: int

        """
        return self._num_actions

    def simplify(self, x: torch.Tensor) -> torch.Tensor:
        """Simplify each action to a single value.

        This is a no-op for continuous actions.

        :param x: The action(s) of shape (batch_dim, output_dim).
        :type x: torch.Tensor
        :return: The same action(s) of shape (batch_dim, output_dim).
        :rtype: torch.Tensor

        """
        return x

    def construct(self, action: torch.Tensor) -> torch.Tensor:
        """Construct the full action from the simplified action.

        This is a no-op for continuous actions.

        :param action: The simplified action of shape (batch_dim, output_dim).
        :type action: torch.Tensor
        :return: The same action(s) of shape (batch_dim, output_dim).
        :rtype: torch.Tensor

        """
        return action

    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        """Sample logits using ``Normal``, clipped to [-clip, clip].

        :param logits: The base logits of shape (batch_dim, input_dim).
        :type logits: torch.Tensor
        :return: Tuple containing the sampled action of shape (batch_dim, output_dim)
            and the distribution.
        :rtype: tuple[torch.Tensor, torch.distributions.Distribution]

        """
        # Get mean and std
        mean, std = logits.chunk(2, dim=-1)

        # Process mean and std
        mean = torch.tanh(mean)
        std = (self._std_max - self._std_min) * torch.sigmoid(std + self._std_init) + self._std_min

        # Create distribution and sample
        dist = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
        actions = dist.rsample()

        # Clip actions
        if self._clip > 0:
            # TODO: Double-check against JAX implementation
            # Avoid using `torch.clamp` since it is not differentiable
            # From SheepRL (https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/algos/dreamer_v3/agent.py#L824-L825)
            action_clip = torch.full_like(actions, self._clip)
            actions = actions * (action_clip / torch.maximum(action_clip, actions.abs())).detach()

        return actions, dist


class DiscreteAction(Action):
    """Discrete action definition using ``OneHotCategoricalStraightThrough``."""
    def __init__(self, num_options: int) -> None:
        """Initialize the action definition.

        :param num_options: The number of options for the discrete action.
        :type num_options: int

        """
        super().__init__()

        # Parameters
        self._num_options = num_options

    @property
    def input_dim(self) -> int:
        """The number of input features for the discrete action, equal to the number of options.

        :type: int

        """
        return self._num_options

    @property
    def output_dim(self) -> int:
        """The number of output features for the discrete action, equal to the number of options.

        :type: int

        """
        return self._num_options

    @property
    def num_actions(self) -> int:
        """The number of actions.

        :type: int

        """
        return 1

    def simplify(self, x: torch.Tensor) -> torch.Tensor:
        """Simplify each action to a single value.

        Takes the argmax of the provided one-hot vector.

        :param x: One-hot action of shape (batch_dim, output_dim).
        :type x: torch.Tensor
        :return: The same action(s) of shape (batch_dim, 1).
        :rtype: torch.Tensor

        """
        return torch.argmax(x, dim=-1, keepdim=True)

    def construct(self, action: torch.Tensor) -> torch.Tensor:
        """Construct the full action from the simplified action.

        Converts the provided action index to a one-hot vector.

        :param action: The simplified action of shape (batch_dim, 1).
        :type action: torch.Tensor
        :return: One-hot action of shape (batch_dim, output_dim).
        :rtype: torch.Tensor

        """
        # Check that all values are close to integers
        if not torch.allclose(action, action.round()):
            raise ValueError("Discrete action values must be close to integers.")

        # Cast to long and convert to one-hot
        action = action.long()
        return nn.functional.one_hot(action.squeeze(-1), num_classes=self._num_options).to(torch.get_default_dtype())

    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        """Sample logits using ``OneHotCategoricalStraightThrough``.

        :param logits: The base logits of shape (batch_dim, input_dim).
        :type logits: torch.Tensor
        :return: Tuple containing the sampled action of shape (batch_dim, output_dim)
            and the distribution.
        :rtype: tuple[torch.Tensor, torch.distributions.Distribution]

        """
        # Create and sample from distribution
        dist = torch.distributions.OneHotCategoricalStraightThrough(
                logits=frl_distributions.uniform_mix(logits)[0])
        return dist.rsample(), dist


class TwoHotDiscretizedContinuousAction(Action):
    """Discretized continuous action definition using a two-hot encoding."""
    def __init__(
            self,
            bins: int = 32,
            low: float = -1.,
            high: float = 1.,
            pre_func: callable = frl_distributions.identity,
            post_func: callable = frl_distributions.identity,
            eps: float = 1e-8,
        ) -> None:
        """Initialize the action definition.

        :param num_actions: The number of actions to initialize.
        :type num_actions: int
        :param bins: The number of bins to use for discretization.
        :type bins: int
        :param low: The lower bound of the action values.
        :type low: float
        :param high: The upper bound of the action values.
        :type high: float
        :param pre_func: A function to apply to the input logits before creating the distribution. (Default: ``symlog``)
        :type pre_func: callable
        :param post_func: A function to apply to the output of the distribution. (Default: ``symexp``)
        :type post_func: callable
        :param eps: A small value to add when computing entropy to avoid numerical issues. (Default: ``1e-8``)
        :type eps: float

        """
        # Parameters
        self._bins = bins
        self._low = low
        self._high = high
        self._pre_func = pre_func
        self._post_func = post_func
        self._eps = eps

    @property
    def input_dim(self) -> int:
        """The number of input features for the discretized continuous action, equal to the number of bins.

        :type: int

        """
        return self._bins

    @property
    def output_dim(self) -> int:
        """The number of output features for the discretized continuous action, always 1.

        :type: int

        """
        return 1

    @property
    def num_actions(self) -> int:
        """The number of actions.

        :type: int

        """
        return 1

    def simplify(self, x: torch.Tensor) -> torch.Tensor:
        """Simplify each action to a single value. Is a no-op for discretized continuous actions.

        :param x: The action(s) of shape (batch_dim, output_dim).
        :type x: torch.Tensor
        :return: The input tensor ``x``.
        :rtype: torch.Tensor
        """
        return x

    def construct(self, action: torch.Tensor) -> torch.Tensor:
        """Construct the full action from the simplified action. Is a no-op for discretized continuous actions.

        :param action: The simplified action of shape (batch_dim, output_dim).
        :type action: torch.Tensor
        :return: The input tensor ``action``.
        :rtype: torch.Tensor
        """
        return action

    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, frl_distributions.TwoHot]:
        """Sample logits using a two-hot encoding.

        :param logits: The base logits of shape (batch_dim, input_dim).
        :type logits: torch.Tensor
        :return: Tuple containing the sampled action of shape (batch_dim, output_dim)
            and the distribution.
        :rtype: tuple[torch.Tensor, frl_distributions.TwoHot]

        """
        # Create and sample from distribution
        dist = frl_distributions.TwoHot(
            logits=frl_distributions.uniform_mix(logits.unsqueeze(-2))[0],  # Add output_dim dimension
            # bins=self._bins,
            low=self._low,
            high=self._high,
            pre_func=self._pre_func,
            post_func=self._post_func,
            event_dims=2,  # Last two dimensions are event dimensions (bins and output_dim)
            eps=self._eps)

        return dist.rsample(), dist


class DiscretizedContinuousAction(Action):
    """Discretized continuous action definition using a one-hot encoding."""
    def __init__(
            self,
            bins: int = 32,
            low: float = -1.,
            high: float = 1.,
            pre_func: callable = frl_distributions.identity,
            post_func: callable = frl_distributions.identity,
        ) -> None:
        """Initialize the action definition.

        :param num_actions: The number of actions to initialize.
        :type num_actions: int
        :param bins: The number of bins to use for discretization.
        :type bins: int
        :param low: The lower bound of the action values.
        :type low: float
        :param high: The upper bound of the action values.
        :type high: float
        :param pre_func: A function to apply to the input logits before creating the distribution. (Default: ``symlog``)
        :type pre_func: callable
        :param post_func: A function to apply to the output of the distribution. (Default: ``symexp``)
        :type post_func: callable

        """
        # Parameters
        self._bins = bins
        self._low = low
        self._high = high
        self._pre_func = pre_func
        self._post_func = post_func

        # Compute bin values
        self._bin_values = torch.linspace(low, high, bins)

    @property
    def input_dim(self) -> int:
        """The number of input features for the discretized continuous action, equal to the number of bins.

        :type: int

        """
        return self._bins

    @property
    def output_dim(self) -> int:
        """The number of output features for the discretized continuous action, equal to the number of bins.

        :type: int

        """
        return self._bins

    @property
    def num_actions(self) -> int:
        """The number of actions.

        :type: int

        """
        return 1

    def simplify(self, x: torch.Tensor) -> torch.Tensor:
        """Simplify each action to a single value.

        Takes the index of the one-hot encoded action and returns the corresponding bin value.

        :param x: The action(s) of shape (batch_dim, output_dim).
        :type x: torch.Tensor
        :return: The simplified action of shape (batch_dim, 1).
        :rtype: torch.Tensor

        """
        # Make sure bin values are on the same device as input
        self._bin_values = self._bin_values.to(x.device)

        # Get index of one-hot encoding
        indices = torch.argmax(x, dim=-1)

        # Get corresponding bin values
        return self._bin_values[indices].unsqueeze(-1)

    def construct(self, action: torch.Tensor) -> torch.Tensor:
        """Construct the full action from the simplified action.

        Takes the simplified action value and returns a one-hot encoding corresponding to the proper bin.

        :param action: The simplified action of shape (batch_dim, 1).
        :type action: torch.Tensor
        :return: The full action of shape (batch_dim, output_dim).
        :rtype: torch.Tensor

        """
        # Make sure bin values are on the same device as action
        self._bin_values = self._bin_values.to(action.device)

        # Get bin indices for each action value
        indices = torch.bucketize(action.squeeze(-1).contiguous(), self._bin_values)  # Will sometimes warn about input tensor being non-contiguous

        # Convert to one-hot encoding
        return nn.functional.one_hot(indices, num_classes=self._bins).to(torch.get_default_dtype())

    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        """Sample logits using a one-hot encoding.

        :param logits: The base logits of shape (batch_dim, input_dim).
        :type logits: torch.Tensor
        :return: Tuple containing the sampled action of shape (batch_dim, output_dim)
            and the distribution.
        :rtype: tuple[torch.Tensor, torch.distributions.Distribution]

        """
        # Create and sample from distribution
        dist = torch.distributions.OneHotCategoricalStraightThrough(
            logits=frl_distributions.uniform_mix(logits)[0])
        return dist.rsample(), dist


class ACTION_IDENTIFIERS(enum.Enum, metaclass=frl_utilities.CaseInsensitiveEnumMeta):
    """String identifiers for action definitions, mapped to their corresponding classes."""
    CONTINUOUS = ContinuousActions
    DISCRETE = DiscreteAction
    TWO_HOT_DISCRETIZED_CONTINUOUS = TwoHotDiscretizedContinuousAction
    DISCRETIZED_CONTINUOUS = DiscretizedContinuousAction


def simplify_actions(actions: torch.Tensor, model_actions: list[Action]) -> torch.Tensor:
    """Simplify actions using the action definitions provided.

    :param actions: The actions of shape (batch_dim, sum(output_dim)).
    :type actions: torch.Tensor
    :param model_actions: A list of action definitions.
    :type model_actions: list[Action]
    :return: The simplified actions of shape (batch_dim, sum(num_actions)).
    :rtype: torch.Tensor

    """
    return torch.cat([
        ma.simplify(a)
        for a, ma
        in zip(
            actions.split([ma.output_dim for ma in model_actions], dim=-1),
            model_actions)
    ], dim=-1)


def construct_actions(actions: torch.Tensor, model_actions: list[Action]) -> torch.Tensor:
    """Construct actions using the action definitions provided.

    :param actions: The simplified actions of shape (batch_dim, sum(num_actions)).
    :type actions: torch.Tensor
    :param model_actions: A list of action definitions.
    :type model_actions: list[Action]
    :return: The full actions of shape (batch_dim, sum(output_dim)).
    :rtype: torch.Tensor

    """
    return torch.cat([
        ma.construct(a)
        for a, ma
        in zip(
            actions.split([ma.num_actions for ma in model_actions], dim=-1),
            model_actions)
    ], dim=-1)
