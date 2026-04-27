"""Environment definitions for FishyRL."""

import enum
import time
import warnings
from abc import abstractmethod
from typing import Any

import numpy as np

from . import utilities as frl_utilities

# Try importing Gymnasium
try:
    import gymnasium as gym
except ImportError:
    pass

# Try importing RLGym
# TODO: Maybe move import inside VectorizedRLGymEnvironment
try:
    import rlgym.api as rlapi
    import rlgym.rocket_league.action_parsers as rlact
    import rlgym.rocket_league.common_values as rlcommon
    import rlgym.rocket_league.done_conditions as rldone
    import rlgym.rocket_league.obs_builders as rlobs
    import rlgym.rocket_league.reward_functions as rlreward
    import rlgym.rocket_league.rlviser as rlviser
    import rlgym.rocket_league.sim as rlsim
    import rlgym.rocket_league.state_mutators as rlstate
except ImportError:
    pass


class VectorizedEnvironment:
    """Abstract base class for vectorized environments."""
    @property
    @abstractmethod
    def num_envs(self) -> int:
        """The number of environments.

        :type: int

        """
        pass

    @property
    @abstractmethod
    def render_fps(self) -> int:
        """The number of frames per second while rendering.

        :type: int

        """
        pass

    @property
    @abstractmethod
    def obs_shape(self) -> np.ndarray:
        """The shape of the observation space.

        :type: np.ndarray

        """
        pass

    @abstractmethod
    def action_sample(self) -> np.ndarray:
        """Sample an action from the action space.

        :rtype: np.ndarray

        """
        pass

    @abstractmethod
    def reset(self, seed: int = None, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.

        :param seed: The random seed for the environment. (Default: ``None``)
        :type seed: int
        :param kwargs: Additional keyword arguments for resetting the environment.
        :type kwargs: dict[str, Any]
        :return: A tuple containing the initial observations and additional info.
        :rtype: tuple[np.ndarray, dict[str, Any]]

        """
        pass

    @abstractmethod
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Perform a step in the environment.

        :param actions: The actions to perform in the environment.
        :type actions: np.ndarray
        :return: A tuple containing the next observations, rewards, terminations, truncations, and additional info.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]

        """
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        """Render the environment and return a frame.

        :return: The rendered frame.
        :rtype: np.ndarray

        """
        pass

    @abstractmethod
    def copy(self, **kwargs: dict[str, Any]) -> 'VectorizedEnvironment':
        """Copy the environment, overriding parameters in `kwargs`.

        :param kwargs: Additional keyword arguments for the new environment instance.
        :type kwargs: dict[str, Any]
        :return: A copy of the environment with overridden parameters.
        :rtype: VectorizedEnvironment

        """
        pass


class VectorizedGymEnvironment(VectorizedEnvironment):
    """A vectorized environment wrapper for Gymnasium environments."""
    def __init__(self, env_name: str, num_envs: int = 1, allow_rendering: bool = False, **init_kwargs: dict[str, Any]) -> None:
        """Initialize the vectorized Gymnasium environment.

        :param env_name: The name of the Gymnasium environment to create.
        :type env_name: str
        :param num_envs: The number of parallel environments to create. (Default: ``1``)
        :type num_envs: int
        :param allow_rendering: Whether to allow rendering of the environment. (Default: ``False``)
        :type allow_rendering: bool
        :param init_kwargs: Additional keyword arguments for initializing the Gymnasium environment.
        :type init_kwargs: dict[str, Any]

        """
        # Parameters
        self._env_name = env_name
        self._num_envs = num_envs
        self._allow_rendering = allow_rendering
        self._init_kwargs = init_kwargs

        # Initialize environments
        self._envs = gym.vector.AsyncVectorEnv([
            lambda: gym.make(
                self._env_name,
                render_mode='rgb_array' if self._allow_rendering else None,
                **self._init_kwargs) for _ in range(self._num_envs)])

    @property
    def num_envs(self) -> int:
        """The number of environments.

        :type: int

        """
        return self._envs.num_envs

    @property
    def render_fps(self) -> int:
        """The number of frames per second while rendering.

        :type: int

        """
        return self._envs.metadata['render_fps']

    @property
    def obs_shape(self) -> np.ndarray:
        """The shape of the observation space.

        :type: np.ndarray

        """
        return self._envs.observation_space.shape

    def action_sample(self) -> np.ndarray:
        """Sample an action from the action space.

        :return: A sampled action.
        :rtype: np.ndarray

        """
        return self._envs.action_space.sample()

    def reset(self, seed: int = None, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.

        :param seed: The random seed for the environment. (Default: ``None``)
        :type seed: int
        :param kwargs: Additional keyword arguments for resetting the environment.
        :type kwargs: dict[str, Any]
        :return: A tuple containing the initial observations and additional info.
        :rtype: tuple[np.ndarray, dict[str, Any]]

        """
        return self._envs.reset(seed=seed, **kwargs)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Take a step in the environment.

        :param actions: The actions to take in the environment.
        :type actions: np.ndarray
        :return: A tuple containing the next observations, rewards, dones, truncations, and additional info.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]

        """
        return self._envs.step(actions)

    def render(self) -> np.ndarray:
        """Render the environment and return a frame.

        :return: A rendered frame.
        :rtype: np.ndarray

        """
        if not self._allow_rendering:
            raise ValueError('Parameter `allow_rendering` is set to `False`.')
        return np.stack(self._envs.render(), axis=0)

    def copy(self, **kwargs: dict[str, Any]) -> 'VectorizedGymEnvironment':
        """Copy the environment, overriding parameters in `kwargs`.

        :param kwargs: Additional keyword arguments for the new environment instance.
        :type kwargs: dict[str, Any]
        :return: A copy of the environment with overridden parameters.
        :rtype: VectorizedGymEnvironment

        """
        new_kwargs = {
            'env_name': self._env_name,
            'num_envs': self._num_envs,
            'allow_rendering': self._allow_rendering,
            **self._init_kwargs}
        new_kwargs.update(kwargs)
        return VectorizedGymEnvironment(**new_kwargs)


class VectorizedRLGymEnvironment(VectorizedEnvironment):
    """A vectorized environment wrapper for RLGym environments."""
    def __init__(
        self,
        env_name: str = 'Soccar',
        num_envs: int = 1,
        team_size: int | list[int] = 2,
        frame_skip: int = 8,
        allow_rendering: bool = False,
        automatic_reset: bool = True,
        **init_kwargs: dict[str, Any],
    ) -> None:
        """Initialize the vectorized RLGym environment.

        :param env_name: The name of the RLGym environment to create, currently unused. (Default: ``'Soccar'``)
        :type env_name: str
        :param num_envs: The number of parallel environments to create. (Default: ``1``)
        :type num_envs: int
        :param team_size: The size of both teams. Can be an int or a list of ints for each environment. (Default: ``2``)
        :type team_size: int | list[int]
        :param allow_rendering: Whether to allow rendering of the environment, frame grabbing is currently unsupported in favor
            of live visualization. (Default: ``False``)
        :param frame_skip: The number of frames to skip for each action. (Default: ``8``)
        :type frame_skip: int
        :type allow_rendering: bool
        :param automatic_reset: Whether to automatically reset the environment after termination. (Default: ``True``)
        :type automatic_reset: bool
        :param init_kwargs: Additional keyword arguments for initializing the RLGym environment. (Default: ``{}``)
        :type init_kwargs: dict[str, Any]

        """
        # Parameters
        self._env_name = env_name
        # NOTE: We count each car as a separate environment for the purposes of tracking terminations and observations,
        #       but we still want to create the correct number of actual RLGym environments
        self._actual_num_envs = num_envs
        self._team_size = [team_size] * num_envs if isinstance(team_size, int) else team_size
        self._effective_num_envs = sum([ts * 2 for ts in self._team_size])  # Each team has `team_size` cars, and there are 2 teams
        self._frame_skip = frame_skip
        self._allow_rendering = allow_rendering
        self._automatic_reset = automatic_reset
        self._init_kwargs = init_kwargs

        # Import new rewards
        from .rlgym.rewards import CloseReward

        # Initialize environments
        self._envs = []
        for team_size in self._team_size:
            # Soccar environment
            if env_name.upper() == 'SOCCAR':
                # Actions
                action_parser = rlact.RepeatAction(
                    rlact.LookupTableAction(), repeats=self._frame_skip)  # TODO: Should scale rewards inversely with repeats
                # Truncation and termination
                termination_cond = rldone.GoalCondition()
                truncation_cond = rldone.AnyCondition(
                    rldone.NoTouchTimeoutCondition(30), rldone.TimeoutCondition(300))
                # Rewards
                reward_fn = rlreward.CombinedReward(
                    (rlreward.GoalReward(), 100), (rlreward.TouchReward(), 1), (CloseReward(), 10))  # TODO: Maybe remove `TouchReward`
                # Observations
                obs_builder = rlobs.DefaultObs(
                    zero_padding=None,
                    pos_coef=np.asarray([1 / rlcommon.SIDE_WALL_X, 1 / rlcommon.BACK_NET_Y, 1 / rlcommon.CEILING_Z]),
                    ang_coef=1 / np.pi,
                    lin_vel_coef=1 / rlcommon.CAR_MAX_SPEED,
                    ang_vel_coef=1 / rlcommon.CAR_MAX_ANG_VEL,
                    boost_coef=1 / 100.)
                # Training conditions
                state_mutator = rlstate.MutatorSequence(
                    rlstate.FixedTeamSizeMutator(team_size, team_size),
                    rlstate.KickoffMutator())
                # Create environment
                env = rlapi.RLGym(
                    action_parser=action_parser,
                    termination_cond=termination_cond,
                    truncation_cond=truncation_cond,
                    reward_fn=reward_fn,
                    obs_builder=obs_builder,
                    state_mutator=state_mutator,
                    transition_engine=rlsim.RocketSimEngine(),
                    renderer=rlviser.RLViserRenderer(tick_rate=120. / action_parser.repeats) if self._allow_rendering else None)

            # Other environments
            else:
                raise NotImplementedError(f"Environment '{env_name}' is not implemented for RLGym.")

            # Record
            self._envs.append(env)

        # Initialize render targets
        self._tick_duration = self._frame_skip / 120.

        # Call an initial reset to initialize observations and tracking
        self.reset()

    @property
    def num_envs(self) -> int:
        """The number of environments.

        :type: int

        """
        return self._effective_num_envs

    @property
    def render_fps(self) -> int:
        """The number of frames per second while rendering.

        :type: int

        """
        # return rlcommon.TICKS_PER_SECOND / self._envs[0].action_parser.repeats  # 1 frame per action
        # return rlcommon.TICKS_PER_SECOND  # Actual ticks
        return self._envs[0].renderer.tick_rate

    @property
    def obs_shape(self) -> np.ndarray:
        """The shape of the observation space.

        :type: np.ndarray

        """
        return list(self._envs[0].observation_spaces.values())[0][1]

    def _construct_actions(self, actions: np.ndarray) -> list[dict[str, np.ndarray]]:
        """Construct a list of dicts mapping agents to one-hot actions from an array of discrete actions.

        :param actions: An array of discrete actions, with shape (num_envs, num_agents).
        :type actions: np.ndarray
        :return: A list of dicts mapping agents to one-hot actions, with length num_envs.
        :rtype: list[dict[str, np.ndarray]]

        """
        # Get action space
        # NOTE: Assumes all environments have the same action space, which should be true for this implementation
        action_space_type, action_space_size = list(self._envs[0].action_spaces.values())[0]

        # Check that action space is correct
        if action_space_type != 'discrete':
            raise NotImplementedError("Only discrete action spaces are currently supported.")

        # Decompose array into list of dicts mapping agents to one-hot actions
        new_actions = []
        running_agent_index = 0
        for env in self._envs:
            # Get agents
            agents = env.agents

            # Create dict of actions for each agent, converting from discrete to one-hot
            env_new_actions = {agent: np.zeros(action_space_size, dtype=bool) for agent in agents}
            for agent in agents:
                env_new_actions[agent][actions[running_agent_index]] = 1
                running_agent_index += 1
            new_actions.append(env_new_actions)

        return new_actions

    def action_sample(self) -> np.ndarray:
        """Sample an action from the action space.

        :return: An array of sampled actions, with shape (num_envs, num_actions).
        :rtype: np.ndarray

        """
        # Get action space
        action_space_type, action_space_size = list(self._envs[0].action_spaces.values())[0]

        # Check that action space is correct
        if action_space_type != 'discrete':
            raise NotImplementedError("Only discrete action spaces are currently supported.")

        # Sample actions
        actions = []
        for env in self._envs:
            # Randomly sample one action per agent
            env_actions = np.random.randint(action_space_size, size=len(env.agents)).reshape(-1, 1)
            actions.append(env_actions)

        # Stack actions
        actions = np.concatenate(actions, axis=0)

        return actions

    def reset(self, seed: int = None, idx: int = None, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environments.

        :param seed: An optional seed for the environments, currently unsupported. (Default: ``None``)
        :type seed: int
        :param idx: An optional index to specify which environment to reset. (Default: ``None``)
        :type idx: int
        :param kwargs: Additional keyword arguments to pass to the environment reset method.
        :type kwargs: dict[str, Any]
        :return: A tuple containing the initial observations and an empty info dictionary.
        :rtype: tuple[np.ndarray, dict[str, Any]]

        """
        # Raise warning about ignoring seed
        if seed is not None:
            warnings.warn("RLGym environments do not currently support seeding, ignoring.")

        # Reset environments and record observations
        obs = []
        envs = self._envs if idx is None else [self._envs[idx]]
        for env in envs:
            env_obs = env.reset(**kwargs)
            env_keys = list(env.observation_spaces.keys())
            env_obs = np.stack(list(env_obs.values()), axis=0)
            obs.append(env_obs)
        obs = np.concatenate(obs, axis=0)  # We concat to count each car as a separate observation

        # Adjust terminated tracking for automatic reset
        if idx is not None:
            self._ended[idx] = False
        else:
            self._ended = np.zeros(self._actual_num_envs, dtype=bool)

        # Check that agents order matches key order
        for env in self._envs:
            if env_keys != env.agents:
                raise ValueError('The order of agents must match the order of keys in the observation space. Something went wrong with RLGym environment initialization.')

        # Reset render tracking
        self._rendering = False
        self._render_time = time.perf_counter()

        return obs, {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Take a step in the environments.

        :param actions: An array of actions to take in the environments, with shape (num_envs, num_agents).
        :type actions: np.ndarray
        :return: A tuple containing the next observations, rewards, terminations, truncations, and an empty info dictionary.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]

        """
        # Deconstruct actions into list of dicts
        actions = self._construct_actions(actions)

        # Step environments and record results
        obs, rewards, terminations, truncations = [], [], [], []
        for i, (env, env_actions, env_terminal) in enumerate(zip(self._envs, actions, self._ended)):
            # Reset if terminal and have automatic reset setting
            if self._automatic_reset and env_terminal:
                env_obs, _ = self.reset(idx=i)
                env_rewards = {agent: 0 for agent in env.agents}
                env_terminations = {agent: False for agent in env.agents}
                env_truncations = {agent: False for agent in env.agents}

                # Adjust ended tracking
                self._ended[i] = False

            # Otherwise, perform step
            else:
                env_obs, env_rewards, env_terminations, env_truncations = env.step(env_actions)
                env_obs = np.stack(list(env_obs.values()), axis=0)

            # Format results
            env_rewards = np.stack(list(env_rewards.values()), axis=0)
            env_terminations = np.stack(list(env_terminations.values()), axis=0)
            env_truncations = np.stack(list(env_truncations.values()), axis=0)

            # Record
            obs.append(env_obs)
            rewards.append(env_rewards)
            terminations.append(env_terminations)
            truncations.append(env_truncations)

            # Record terminations for automatic reset, record if any agent is terminated or truncated
            self._ended[i] = np.logical_or.reduce([self._ended[i], env_terminations.any(), env_truncations.any()])

        # Stack results
        obs = np.concatenate(obs, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminations = np.concatenate(terminations, axis=0)
        truncations = np.concatenate(truncations, axis=0)

        return obs, rewards, terminations, truncations, {}

    def render(self, delay: bool = True, speedup: float = 1.0, warn: bool = True) -> np.ndarray:
        """Render the environments and return a frame.

        Live visualization is currently supported, but frame grabbing is not due no support with RLViser. This method will render the
        first environment to the RLViser window, but will return an empty array.

        :param delay: Whether to delay rendering to match the render FPS. (Default: ``True``)
        :type delay: bool
        :param speedup: A factor to speed up rendering, applied to the delay. (Default: ``1.0``)
        :type speedup: float
        :param warn: Whether to display warnings on long computations. (Default: ``True``)
        :type warn: bool
        :return: An array of rendered frames, with shape (num_envs, height, width, channels).
        :rtype: np.ndarray

        """
        # Download from https://github.com/VirxEC/rlviser/releases/tag/v0.9.1, and install virtual display (Xvfb) if running headless
        # Initialize virtual display if running headless
        # if os.environ.get('DISPLAY') is None:
        #     import pyvirtualdisplay
        #     pyvirtualdisplay.Display(visible=True, size=(1280, 720), backend='xvfb').start()

        # # Take screenshot
        # def screenshot() -> np.ndarray:
        #     import mss
        #     with mss.mss() as sct:
        #         frame = np.array(sct.grab(sct.monitors[0]))
        #         frame = frame[:, :, [2, 1, 0]]  # Convert from BGRA to RGB
        #     return frame
        # NOTE: Rendering currently does not work for capture, only for visualization
        # TODO: Implement in RLBot and just record manually
        # raise NotImplementedError("Rendering is currently not implemented for RLGym environments. This is due to issues with RLViser.")

        # # Gather frames from each environment
        # frames = []
        # for env in self._envs:
        #     env_frame = env.render()
        #     frames.append(env_frame)

        # # Stack frames and return
        # return np.stack(frames, axis=0)

        # Delay to match render FPS
        if delay and self._rendering:
            elapsed = time.perf_counter() - self._render_time
            render_target = self._tick_duration / speedup
            sleep_time = max(0, render_target - elapsed)
            time.sleep(sleep_time)

            # Warn if sleep time is below zero
            if warn and sleep_time <= 0:
                warnings.warn(f'Rendering took {1000 * elapsed:.0f}ms, which is longer than the target frame time of {1000 * render_target:.0f}ms. '
                              f'Consider lowering the `speedup` parameter if this warning persists.')

        # Set rendering flag
        self._rendering = True

        # Render
        self._envs[0].render()  # We render just to update the RLViser window, but we can't capture frames from it, so we return an empty array

        # Update render time
        self._render_time = time.perf_counter()

        return np.array([])

    def copy(self, **kwargs: dict[str, Any]) -> 'VectorizedRLGymEnvironment':
        """Copy the environment, overriding parameters in `kwargs`.

        :param kwargs: Additional keyword arguments for the new environment instance.
        :type kwargs: dict[str, Any]
        :return: A copy of the environment with overridden parameters.
        :rtype: VectorizedRLGymEnvironment

        """
        new_kwargs = {
            'num_envs': self._actual_num_envs,
            'allow_rendering': self._allow_rendering,
            **self._init_kwargs}
        new_kwargs.update(kwargs)
        return VectorizedRLGymEnvironment(**new_kwargs)


class ENVIRONMENT_IDENTIFIERS(enum.Enum, metaclass=frl_utilities.CaseInsensitiveEnumMeta):
    """String identifiers for environment definitions, mapped to their corresponding classes."""
    GYMNASIUM = VectorizedGymEnvironment
    RLGYM = VectorizedRLGymEnvironment
