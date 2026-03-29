"""Utility action definitions for reinforcement learning agents."""

from abc import abstractmethod

import torch
import torch.nn as nn

from . import distributions


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
    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action from the logits.

        :param logits: The base logits of shape (batch_dim, input_dim).
        :type logits: torch.Tensor
        :return: The sampled action of shape (batch_dim, output_dim).
        :rtype: torch.Tensor

        """
        pass


class ContinuousActions(Action):
    """Continuous action definition using `Normal`."""
    def __init__(
            self,
            num_actions: int = 1,
            std_min: float = 1,
            std_max: float = 1,
            clip: float = 0,
        ) -> None:
        """Initialize the action definition.

        :param num_actions: The number of actions to initialize.
        :type num_actions: int
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

    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample logits using `Normal`, clipped to [-clip, clip].

        :param logits: The base logits of shape (batch_dim, input_dim).
        :type logits: torch.Tensor
        :return: Tuple containing the sampled action of shape (batch_dim, output_dim)
            and the distribution.
        :rtype: tuple[torch.Tensor, torch.Tensor]

        """
        # Get mean and std
        mean, std = logits.chunk(2, dim=-1)

        # Process mean and std
        mean = torch.tanh(mean)
        std = (self._std_max - self._std_min) * (std + 1) + self._std_min
        std = torch.sigmoid(std)

        # Create distribution and sample
        dist = torch.distributions.Normal(mean, std)
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
    """Discrete action definition using `OneHotCategoricalStraightThrough`."""
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

    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample logits using `OneHotCategoricalStraightThrough`.

        :param logits: The base logits of shape (batch_dim, input_dim).
        :type logits: torch.Tensor
        :return: Tuple containing the sampled action of shape (batch_dim, output_dim)
            and the distribution.
        :rtype: tuple[torch.Tensor, torch.Tensor]

        """
        # Create and sample from distribution
        dist = torch.distributions.OneHotCategoricalStraightThrough(
            logits=distributions.uniform_mix(logits)[0])
        return dist.rsample(), dist


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
