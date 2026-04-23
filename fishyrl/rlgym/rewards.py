"""Custom reward functions for RLGym environments."""

from typing import Any

import numpy as np
import rlgym.api as rlapi
import rlgym.rocket_league.api as rlrlapi
import rlgym.rocket_league.common_values as rlcommon


class CloseReward(rlapi.RewardFunction):  # TODO: Move to separate file for dependency reasons
    """A reward function that rewards agents for being close to the ball in RLGym."""
    def __init__(self, *args: list[Any], use_diff: bool = True, **kwargs: dict[str, Any]) -> None:
        """Initialize the CloseReward.

        :param args: Positional arguments for the base class.
        :type args: Any
        :param use_diff: Whether to use the difference in distance for reward calculation. If ``False``, the
            reward is based on the absolute distance. (Default: ``False``)
        :type use_diff: bool
        :param kwargs: Keyword arguments for the base class.
        :type kwargs: dict[str, Any]

        """
        super().__init__(*args, **kwargs)

        # Parameters
        self._use_diff = use_diff

        # Get scaling factor based on max diagonal of field to scale rewards to a range of ~2
        self._max_dist = np.linalg.norm([rlcommon.SIDE_WALL_X, rlcommon.BACK_NET_Y, rlcommon.CEILING_Z])


    def _compute_dist(self, state: rlrlapi.GameState) -> dict[str, float]:
        """Compute the distance from each agent to the ball.

        :param state: The current game state.
        :type state: rlgym.rocket_league.api.GameState
        :return: A dictionary mapping each agent to their distance from the ball.
        :rtype: dict[str, float]

        """
        # Get ball position
        ball_pos = state.ball.position

        # Compute distance for each agent
        dist = {}
        for agent in state.cars:
            car_pos = state.cars[agent].physics.position
            dist[agent] = np.linalg.norm(car_pos - ball_pos)

        return dist

    def reset(self, agents: list[str], initial_state: rlrlapi.GameState, shared_info: dict[str, Any]) -> None:
        """Reset the reward function, currently does nothing.

        :param agents: A list of agent identifiers.
        :type agents: list[str]
        :param initial_state: The initial game state.
        :type initial_state: rlgym.rocket_league.api.GameState
        :param shared_info: Shared info.
        :type shared_info: dict[str, Any]

        """
        self._dist = self._compute_dist(initial_state)

    def get_rewards(self, agents: list[str], state: rlrlapi.GameState, is_terminated: dict[str, bool], is_truncated: dict[str, bool], shared_info: dict[str, Any]) -> dict[str, float]:
        """Compute the rewards for the given state based on closeness to the ball.

        :param agents: A list of agent identifiers.
        :type agents: list[str]
        :param state: The current game state.
        :type state: rlgym.rocket_league.api.GameState
        :param is_terminated: A dictionary indicating if each agent's episode has terminated.
        :type is_terminated: dict[str, bool]
        :param is_truncated: A dictionary indicating if each agent's episode has been truncated.
        :type is_truncated: dict[str, bool]
        :param shared_info: Shared info.
        :type shared_info: dict[str, Any]
        :return: A dictionary mapping each agent to their computed reward.
        :rtype: dict[str, float]

        """
        # Get new distances
        new_dist = self._compute_dist(state)

        # Compute reward for each agent based on distance to ball
        rewards = {}
        for agent in agents:
            # Compute reward and record
            reward = self._dist[agent] - new_dist[agent] if self._use_diff else - new_dist[agent]
            reward = reward / self._max_dist  # Scale reward to a range of ~2
            rewards[agent] = reward

        # Update stored distances
        self._dist = new_dist

        return rewards
