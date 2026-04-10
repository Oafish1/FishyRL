"""Main functions for training and running the Dreamer agent."""

import copy
from typing import Any

import gymnasium as gym
import torch

from . import actions as frl_actions
from . import buffers as frl_buffers
from . import models as frl_models
from . import utilities as frl_utilities


@frl_utilities.optional_flatten_cfg(exceptions=[])
def get_environments_and_actions(
    env_name: str,
    env_num: int = 1,
    env_actions: list[dict[str, Any]] = [],
    **kwargs: dict[str, Any],
) -> tuple[gym.vector.AsyncVectorEnv, list[frl_actions.Action]]:
    """Get the environments and actions for the Dreamer agent, given a configuration.

    :param env_name: The identifier of the environment to create.
    :type env_name: str
    :param env_num: The number of parallel environments to create. (Default: ``1``)
    :type env_num: int
    :param env_actions: A list of action configurations for the environment. (Default: ``[]``)
    :type env_actions: list[dict[str, Any]]
    :param kwargs: Catch keyword arguments for compatibility with ``utilities.optional_flatten_cfg``.
    :type kwargs: dict[str, Any]

    :return: A tuple containing the vectorized environments and a list of actions.

    """
    # Initialize environments
    envs = gym.vector.AsyncVectorEnv([
        lambda: gym.make(env_name) for _ in range(env_num)])

    # Make actions
    actions = []
    for action in env_actions:
        actions.append(
            frl_actions.ACTION_IDENTIFIERS[action['type']](
                **{k: v for k, v in action.items() if k != 'type'}))

    return envs, actions


@frl_utilities.optional_flatten_cfg(exceptions=[])
def construct_models(
    # Environment parameters
    env_obs_dim: int,
    env_actions: list[frl_actions.Action],
    env_num: int,
    # Model hidden parameters
    model_global_embedded: int = 1024,
    model_global_blocks: int = 5,
    model_global_dense: int = 1024,
    # Model binning parameters
    model_global_categorical_bins: int = 32,
    model_global_reward_bins: int = 255,
    # Model state parameters
    model_global_stochastic_dim: int = 32,
    model_global_deterministic_dim: int = 4096,
    # World parameters
    model_world_lr: float = 1e-4,
    model_world_eps: float = 1e-8,
    model_world_learnable_initial_state: bool = True,
    # Actor parameters
    model_actor_lr: float = 8e-5,
    model_actor_eps: float = 1e-5,
    # Critic parameters
    model_critic_lr: float = 8e-5,
    model_critic_eps: float = 1e-5,
    # Buffer parameters
    buffer_capacity: int = 10**6,
    # Normalizer parameters
    scaler_eps: float = 1.,
    # Replay parameters
    replay_ratio: float = 0.5,
    # Device
    device: torch.device | str = 'cuda' if torch.cuda.is_available() else 'cpu',
    # Compatibility
    **kwargs: dict[str, Any],
) -> None:
    """Construct the models for the Dreamer agent.

    :param env_obs_dim: The dimension of the environment observations.
    :type env_obs_dim: int
    :param env_actions: A list of actions for the environment.
    :type env_actions: list[frl_actions.Action]
    :param env_num: The number of parallel environments. (Default: ``1``)
    :type env_num: int
    :param model_global_embedded: The dimension of the embedded observation space. (Default: ``1024``)
    :type model_global_embedded: int
    :param model_global_blocks: The number of blocks in the MLP models. (Default: ``5``)
    :type model_global_blocks: int
    :param model_global_dense: The dimension of the dense layers in the MLP models. (Default: ``1024``)
    :type model_global_dense: int
    :param model_global_categorical_bins: The number of categorical bins for the stochastic state. (Default: ``32``)
    :type model_global_categorical_bins: int
    :param model_global_reward_bins: The number of categorical bins for the reward prediction. (Default: ``255``)
    :type model_global_reward_bins: int
    :param model_global_stochastic_dim: The dimension of the stochastic state. (Default: ``32``)
    :type model_global_stochastic_dim: int
    :param model_global_deterministic_dim: The dimension of the deterministic state. (Default: ``4096``)
    :type model_global_deterministic_dim: int
    :param model_world_lr: The learning rate for the world model. (Default: ``1e-4``)
    :type model_world_lr: float
    :param model_world_eps: The epsilon for the world model optimizer. (Default: ``1e-8``)
    :type model_world_eps: float
    :param model_world_learnable_initial_state: Whether the initial state of the ``RSSM`` model is learnable. (Default: ``True``)
    :type model_world_learnable_initial_state: bool
    :param model_actor_lr: The learning rate for the actor model. (Default: ``8e-5``)
    :type model_actor_lr: float
    :param model_actor_eps: The epsilon for the actor model optimizer. (Default: ``1e-5``)
    :type model_actor_eps: float
    :param model_critic_lr: The learning rate for the critic model. (Default: ``8e-5``)
    :type model_critic_lr: float
    :param model_critic_eps: The epsilon for the critic model optimizer. (Default: ``8e-5``)
    :type model_critic_eps: float
    :param buffer_capacity: The capacity of the replay buffer. (Default: ``10**6``)
    :type buffer_capacity: int
    :param scaler_eps: The epsilon for the lambda value normalizer. (Default: ``1.``)
    :type scaler_eps: float
    :param replay_ratio: The ratio of model updates to environment steps. (Default: ``0.5``)
    :type replay_ratio: float
    :param device: The device to use for the models. (Default: ``'cuda'`` if available, else ``'cpu'``)
    :type device: torch.device | str
    :param kwargs: Catch keyword arguments for compatibility with ``utilities.optional_flatten_cfg``.
    :type kwargs: dict[str, Any]
    :return: A tuple containing the world model, agent model, and utility modules.
    :rtype: tuple[frl_utilities.ContainerModule, frl_utilities.ContainerModule, frl_utilities.ContainerModule]

    """
    # TODO: Add per-model hyperparameters
    # Compute model action derivates
    model_action_output_dims = [action.output_dim for action in env_actions]  # TODO: Consider auto-generating environment if this is str
    action_dim = sum(model_action_output_dims)

    # Encoder-decoder models
    encoder_model = frl_models.MLPEncoder(env_obs_dim, model_global_embedded, num_blocks=model_global_blocks, hidden_dim=model_global_dense).to(device)
    decoder_model = frl_models.MLPDecoder(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, env_obs_dim, num_blocks=model_global_blocks, hidden_dim=model_global_dense).to(device)

    # RSSM models
    recurrent_model = frl_models.RecurrentModel(model_global_stochastic_dim * model_global_categorical_bins + action_dim, model_global_deterministic_dim).to(device)
    representation_model = frl_models.MLP(model_global_embedded + model_global_deterministic_dim, model_global_stochastic_dim * model_global_categorical_bins).to(device)
    transition_model = frl_models.MLP(model_global_deterministic_dim, model_global_stochastic_dim * model_global_categorical_bins).to(device)
    rssm_model = frl_models.RSSM(recurrent_model, representation_model, transition_model, model_global_categorical_bins, learnable_initial_state=model_world_learnable_initial_state).to(device)

    # Reward and continue models
    reward_model = frl_models.MLP(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, model_global_reward_bins, model_global_blocks * [model_global_dense]).to(device)
    continue_model = frl_models.MLP(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, 1, model_global_blocks * [model_global_dense]).to(device)

    # Actor and critic models
    actor_model = frl_models.Actor(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, env_actions).to(device).apply(frl_utilities.init_weights)
    critic_model = frl_models.MLP(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, model_global_reward_bins, model_global_blocks * [model_global_dense]).to(device).apply(frl_utilities.init_weights)
    target_critic_model = copy.deepcopy(critic_model)

    # Hafner weight initialization
    actor_model._model._model[-1].apply(frl_utilities.uniform_init_weights(1.))
    critic_model._model[-1].apply(frl_utilities.uniform_init_weights(0.))
    rssm_model._transition_model._model[-1].apply(frl_utilities.uniform_init_weights(1.))
    rssm_model._representation_model._model[-1].apply(frl_utilities.uniform_init_weights(1.))
    reward_model._model[-1].apply(frl_utilities.uniform_init_weights(0.))
    continue_model._model[-1].apply(frl_utilities.uniform_init_weights(1.))
    decoder_model._model._model[-1].apply(frl_utilities.uniform_init_weights(1.))

    # Buffer
    buffer = frl_buffers.IndependentVectorizedBuffer(env_num, buffer_capacity // env_num)  # Buffer size of 1M

    # Lambda value normalizer
    lambda_normalizer = frl_utilities.MovingMinMaxScaler(eps=scaler_eps)

    # Gradient update ratio
    ratio = frl_utilities.Ratio(replay_ratio)

    # Containerize world model and add optimizer
    # NOTE: All use Adam optimizer, https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/configs/algo/dreamer_v3.yaml
    world_model = frl_utilities.ContainerModule(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        rssm_model=rssm_model,
        reward_model=reward_model,
        continue_model=continue_model,
    )
    world_model.optimizer = torch.optim.Adam(world_model.parameters(), lr=model_world_lr, eps=model_world_eps)

    # Containerize actor-critic model and add optimizers
    agent_model = frl_utilities.ContainerModule(
        actor_model=actor_model,
        critic_model=critic_model,
        target_critic_model=target_critic_model,
    )
    agent_model.actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=model_actor_lr, eps=model_actor_eps)
    agent_model.critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=model_critic_lr, eps=model_critic_eps)

    # Containerize utilities
    utility_modules = frl_utilities.ContainerModule(
        buffer=buffer,
        lambda_normalizer=lambda_normalizer,
        ratio=ratio,
    )

    return world_model, agent_model, utility_modules
