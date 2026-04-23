"""Main functions for training and running the Dreamer agent."""

import copy
import os
from typing import Any, Union

import numpy as np
import torch
from torch import nn

from . import actions as frl_actions
from . import buffers as frl_buffers
from . import distributions as frl_distributions
from . import environments as frl_environments
from . import losses as frl_losses
from . import models as frl_models
from . import utilities as frl_utilities

# Try importing tensorboard
try:
    import torch.utils.tensorboard
    tensorboard_SummaryWriter_cls = torch.utils.tensorboard.SummaryWriter
except ImportError:
    tensorboard_SummaryWriter_cls = None


@frl_utilities.optional_flatten_cfg(exceptions=[])
def construct_envs(
    env_group: str,
    env_name: str,
    env_num: int = 1,
    **kwargs: dict[str, Any],
) -> frl_environments.VectorizedEnvironment:
    """Get the environments for the Dreamer agent, given a configuration.

    Can optionally take a configuration dictionary with argument ``cfg``.

    :param env_group: The group of environments to create.
    :type env_group: str
    :param env_name: The identifier of the environment to create.
    :type env_name: str
    :param env_num: The number of parallel environments to create. (Default: ``1``)
    :type env_num: int
    :param kwargs: Catch keyword arguments for compatibility with ``utilities.optional_flatten_cfg``.
    :type kwargs: dict[str, Any]

    :return: The vectorized environments.

    """
    # Get environment arguments by taking all kwargs with prefix `env_`
    env_prefix = 'env_'
    env_args = {
        k[len(env_prefix):]: v
        for k, v in kwargs.items()
        if k.startswith(env_prefix) and k != env_prefix + 'actions'}

    # Create environments
    envs = frl_environments.ENVIRONMENT_IDENTIFIERS[env_group](env_name=env_name, num_envs=env_num, **env_args)

    return envs


@frl_utilities.optional_flatten_cfg(exceptions=[])
def construct_actions(
    env_actions: list[dict[str, Any]] = [],
    **kwargs: dict[str, Any],
) -> list[frl_actions.Action]:
    """Get the actions for the Dreamer agent, given a configuration.

    Can optionally take a configuration dictionary with argument ``cfg``.

    :param env_actions: A list of action configurations for the environment. (Default: ``[]``)
    :type env_actions: list[dict[str, Any]]
    :param kwargs: Catch keyword arguments for compatibility with ``utilities.optional_flatten_cfg``.
    :type kwargs: dict[str, Any]

    :return: The list of actions.

    """
    # Make actions
    actions = []
    for action in env_actions:
        rep_num = action['num'] if 'num' in action else 1
        for _ in range(rep_num):
            actions.append(
                frl_actions.ACTION_IDENTIFIERS[action['type']](
                    **{k: v for k, v in action.items() if k not in ('type', 'num')}))

    return actions


@frl_utilities.optional_flatten_cfg(exceptions=[])
def construct_models(
    # Environment parameters
    env_actions: list[frl_actions.Action],
    env_num: int,
    # Model embedding parameters
    model_embedding: list[dict[str, Any]] = [],
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
) -> tuple[frl_utilities.ContainerModule, frl_utilities.ContainerModule, frl_utilities.Container]:
    """Construct the models for the Dreamer agent.

    Can optionally take a configuration dictionary with argument ``cfg``.

    :param env_actions: A list of actions for the environment.
    :type env_actions: list[frl_actions.Action]
    :param env_num: The number of parallel environments. (Default: ``1``)
    :type env_num: int
    :param model_embedding: A list of parameters defining the encoding inputs for the model. For more
        details, see the documentation for ``CompoundEncoder``. (Default: ``[]``)
    :type model_embedding: list[dict[str, Any]]
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
    :rtype: tuple[frl_utilities.ContainerModule, frl_utilities.ContainerModule, frl_utilities.Container]

    """
    # TODO: Add per-model hyperparameters
    # Compute model action derivates
    model_action_output_dims = [action.output_dim for action in env_actions]  # TODO: Consider auto-generating environment if this is str
    action_dim = sum(model_action_output_dims)

    # Encoder-decoder models
    # NOTE: This uses `model_global_embedded` for all encoder and decoder dims, as in the original implementation
    # encoder_model = frl_models.MLPEncoder(env_obs_dim, model_global_embedded, num_blocks=model_global_blocks, hidden_dim=model_global_embedded).to(device)
    # decoder_model = frl_models.MLPDecoder(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, env_obs_dim, num_blocks=model_global_blocks, hidden_dim=model_global_embedded).to(device)
    encoder_model = frl_models.CompoundEncoder(*model_embedding, output_dim=model_global_embedded, num_blocks=model_global_blocks, hidden_dim=model_global_embedded).to(device)
    decoder_model = frl_models.CompoundDecoder(*model_embedding, input_dim=model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, num_blocks=model_global_blocks, hidden_dim=model_global_embedded).to(device)

    # RSSM models
    recurrent_model = frl_models.RecurrentModel(model_global_stochastic_dim * model_global_categorical_bins + action_dim, model_global_deterministic_dim).to(device)
    representation_model = frl_models.MLP(encoder_model.output_dim + model_global_deterministic_dim, model_global_stochastic_dim * model_global_categorical_bins).to(device)
    transition_model = frl_models.MLP(model_global_deterministic_dim, model_global_stochastic_dim * model_global_categorical_bins).to(device)
    rssm_model = frl_models.RSSM(recurrent_model, representation_model, transition_model, model_global_categorical_bins, learnable_initial_state=model_world_learnable_initial_state).to(device)

    # Reward and continue models
    reward_model = frl_models.MLP(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, model_global_reward_bins, model_global_blocks * [model_global_dense]).to(device)
    continue_model = frl_models.MLP(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, 1, model_global_blocks * [model_global_dense]).to(device)

    # Actor and critic models
    actor_model = frl_models.Actor(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, env_actions).to(device).apply(frl_utilities.init_weights)
    critic_model = frl_models.MLP(model_global_stochastic_dim * model_global_categorical_bins + model_global_deterministic_dim, model_global_reward_bins, model_global_blocks * [model_global_dense]).to(device).apply(frl_utilities.init_weights)

    # Hafner weight initialization
    actor_model._model._model[-1].apply(frl_utilities.uniform_init_weights(1.))
    critic_model._model[-1].apply(frl_utilities.uniform_init_weights(0.))
    rssm_model._transition_model._model[-1].apply(frl_utilities.uniform_init_weights(1.))
    rssm_model._representation_model._model[-1].apply(frl_utilities.uniform_init_weights(1.))
    reward_model._model[-1].apply(frl_utilities.uniform_init_weights(0.))
    continue_model._model[-1].apply(frl_utilities.uniform_init_weights(1.))
    for model in decoder_model._decoders:
        if isinstance(model, frl_models.MLPDecoder) or isinstance(model, frl_models.CNNDecoder):
            model._model._model[-1].apply(frl_utilities.uniform_init_weights(1.))
        elif isinstance(model, frl_models.AttentionDecoder):
            model._output_heads.apply(frl_utilities.uniform_init_weights(1.))
            model._existence_heads.apply(frl_utilities.uniform_init_weights(1.))

    # Create target critic model as a copy of critic model, with `requires_grad` set to False
    target_critic_model = copy.deepcopy(critic_model)
    for param in target_critic_model.parameters():
        param.requires_grad = False

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
    actor_critic_model = frl_utilities.ContainerModule(
        actor_model=actor_model,
        critic_model=critic_model,
        target_critic_model=target_critic_model,
    )
    actor_critic_model.actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=model_actor_lr, eps=model_actor_eps)
    actor_critic_model.critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=model_critic_lr, eps=model_critic_eps)

    # Containerize utilities
    utility_modules = frl_utilities.Container(
        buffer=buffer,
        lambda_normalizer=lambda_normalizer,
        ratio=ratio,
    )

    return world_model, actor_critic_model, utility_modules


def save_models(
    path: str = '.',
    world_model: frl_utilities.ContainerModule | None = None,
    actor_critic_model: frl_utilities.ContainerModule | None = None,
    utility_modules: frl_utilities.Container | None = None,
    include_optimizers: bool = True,
) -> None:
    """Save the models and utilities to a file.

    Can optionally take a configuration dictionary with argument ``cfg``.

    :param path: The file path for saving the models. (Default: ``'.'``)
    :type path: str
    :param world_model: The world model to save. Excluded if not provided.
    :type world_model: frl_utilities.ContainerModule
    :param actor_critic_model: The actor-critic model to save. Excluded if not provided.
    :type actor_critic_model: frl_utilities.ContainerModule
    :param utility_modules: The utility modules to save. Excluded if not provided.
    :type utility_modules: frl_utilities.Container
    :param include_optimizers: Whether to include the optimizer states in the saved file. (Default: ``True``)
    :type include_optimizers: bool
    """
    # Initialize weights dictionary
    weights = {}

    # Add world model weights and optimizer state
    if world_model is not None:
        weights['world_model'] = world_model.state_dict()
        if include_optimizers:
            weights['world_optimizer'] = world_model.optimizer.state_dict()

    # Add actor-critic model weights and optimizer states
    if actor_critic_model is not None:
        weights['actor_critic_model'] = actor_critic_model.state_dict()
        if include_optimizers:
            weights['actor_optimizer'] = actor_critic_model.actor_optimizer.state_dict()
            weights['critic_optimizer'] = actor_critic_model.critic_optimizer.state_dict()

    # Add utility modules weights
    if utility_modules is not None:
        weights['utility_modules'] = utility_modules.state_dict()

    # Save models
    torch.save(weights, path)


def load_models(
    path: str = '.',
    world_model: frl_utilities.ContainerModule = None,
    actor_critic_model: frl_utilities.ContainerModule = None,
    utility_modules: frl_utilities.Container = None,
    include_optimizers: bool = True,
) -> None:
    """Load the models and utilities from a file.

    Uses ``weights_only=False`` to allow loading optimizer states, which is somewhat unsafe, so make sure to only load from trusted weight files.

    :param path: The path to the weights file. (Default: ``'.'``)
    :type path: str
    :param world_model: The world model for applying the loaded weights.
    :type world_model: frl_utilities.ContainerModule
    :param actor_critic_model: The actor-critic model for applying the loaded weights.
    :type actor_critic_model: frl_utilities.ContainerModule
    :param utility_modules: The utility modules for applying the loaded weights.
    :type utility_modules: frl_utilities.Container
    :param include_optimizers: Whether to load optimizer states or not. (Default: ``True``)
    :type include_optimizers: bool

    """
    # Load models
    with torch.serialization.safe_globals([np.ndarray, np.dtype, np._core.multiarray._reconstruct]):
        weights = torch.load(path, weights_only=False)  # Somewhat unsafe, make sure that only trusted files are loaded

    # Apply weights to world model
    if world_model is not None:
        world_model.load_state_dict(weights['world_model'])
        if include_optimizers and 'world_optimizer' in weights:
            world_model.optimizer.load_state_dict(weights['world_optimizer'])

    # Apply weights to actor-critic model
    if actor_critic_model is not None:
        actor_critic_model.load_state_dict(weights['actor_critic_model'])
        if include_optimizers and 'actor_optimizer' in weights:
            actor_critic_model.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        if include_optimizers and 'critic_optimizer' in weights:
            actor_critic_model.critic_optimizer.load_state_dict(weights['critic_optimizer'])

    # Apply weights to utility modules
    if utility_modules is not None:
        utility_modules.load_state_dict(weights['utility_modules'])


@frl_utilities.optional_flatten_cfg(exceptions=[])
def learning_step(
    # Manually supplied parameters
    batch: dict[str, torch.Tensor],
    world_model: frl_utilities.ContainerModule,
    actor_critic_model: frl_utilities.ContainerModule,
    utility_modules: frl_utilities.Container,
    tensorboard_writer: tensorboard_SummaryWriter_cls = None,
    environment_step: int = -1,
    # Training parameters
    training_imagination_horizon: int = 15,
    training_free_nats: float = 1.0,
    training_kl_dyn: float = 0.5,
    training_kl_rep: float = 0.1,
    training_kl_reg: float = 1.0,
    training_continue_reg: float = 1.0,
    training_gamma: float = 0.997,
    training_lmbda: float = 0.95,
    training_world_model_grad_clip: float = 1000.0,
    training_actor_model_grad_clip: float = 100.0,
    training_critic_model_grad_clip: float = 100.0,
    **kwargs: dict[str, Any],
) -> None:
    """Perform a learning step for the Dreamer agent.

    :param batch: A batch of data from the replay buffer, containing tensors for observations, actions, rewards, terminations, and truncations.
    :type batch: dict[str, torch.Tensor]
    :param world_model: The world model for the Dreamer agent.
    :type world_model: frl_utilities.ContainerModule
    :param actor_critic_model: The actor-critic model for the Dreamer agent.
    :type actor_critic_model: frl_utilities.ContainerModule
    :param utility_modules: The utility modules for the Dreamer agent.
    :type utility_modules: frl_utilities.Container
    :param tensorboard_writer: The TensorBoard writer for logging, if desired. (Default: ``None``)
    :type tensorboard_writer: torch.utils.tensorboard.SummaryWriter
    :param environment_step: The current environment step. Used for logging purposes. (Default: ``-1``)
    :type environment_step: int
    :param training_imagination_horizon: The number of steps to imagine into the future for actor-critic updates. (Default: ``15``)
    :type training_imagination_horizon: int
    :param training_free_nats: The number of free nats for KL balancing. (Default: ``1.0``)
    :type training_free_nats: float
    :param training_kl_dyn: The weight for the dynamic KL loss. (Default: ``0.5``)
    :type training_kl_dyn: float
    :param training_kl_rep: The weight for the representation KL loss. (Default: ``0.1``)
    :type training_kl_rep: float
    :param training_kl_reg: The overall weight for the KL regularization loss. (Default: ``1.0``)
    :type training_kl_reg: float
    :param training_continue_reg: The weight for the continue loss. (Default: ``1.0``)
    :type training_continue_reg: float
    :param training_gamma: The discount factor for future rewards. (Default: ``0.997``)
    :type training_gamma: float
    :param training_lmbda: The lambda parameter for computing lambda returns. (Default: ``0.95``)
    :type training_lmbda: float
    :param kwargs: Catch keyword arguments for compatibility with ``utilities.optional_flatten_cfg``.
    :type kwargs: dict[str, Any]

    """
    # Infer batch size and sequence length (batch, sequence_length, ...), as well as categorical bins
    batch_size, sequence_length = batch['obs'].shape[:2]
    categorical_bins = world_model.rssm_model._bins

    # Embed observations
    extracted_obs = frl_models.extract_representation(batch['obs'], world_model.encoder_model._encoder_specs)  # Format the observations into specified segments
    embedded_obs = world_model.encoder_model(extracted_obs)

    # Initialize storage
    hidden_states = []
    priors = []
    priors_logits = []
    posteriors = []
    posteriors_logits = []

    # Compute model outputs for each time step, starting with initial recurrent state and posteriors
    for i in range(sequence_length):
        # Run through recurrent model
        ret = world_model.rssm_model(
            batch['actions'][:, i - 1] if i > 0 else None,  # Use the action from the previous step, like in the environment loop
            posteriors[i - 1] if i > 0 else None,
            hidden_states[i - 1] if i > 0 else None,
            embedded_obs[:, i],
            batch['terminations'][:, i - 1] | batch['truncations'][:, i - 1] if i > 0 else None,  # Get initializations using result of previous step
            batch_dim=batch_size,
        )
        hidden_states.append(ret['hidden_state'])
        priors.append(ret['prior'])
        priors_logits.append(ret['prior_logits'])
        posteriors.append(ret['posterior'])
        posteriors_logits.append(ret['posterior_logits'])

    # Concatenate returned tensors
    hidden_states = torch.stack(hidden_states, dim=1)
    priors = torch.stack(priors, dim=1)
    priors_logits = torch.stack(priors_logits, dim=1)
    posteriors = torch.stack(posteriors, dim=1)
    posteriors_logits = torch.stack(posteriors_logits, dim=1)

    # Compute predicted observations, rewards, and continues
    pred_obs = world_model.decoder_model(torch.cat((posteriors, hidden_states), dim=-1))
    pred_rewards = world_model.reward_model(torch.cat((posteriors, hidden_states), dim=-1))
    pred_continues = world_model.continue_model(torch.cat((posteriors, hidden_states), dim=-1))

    # Compute MSE loss for reconstructed observations (po)
    # NOTE: Interestingly, the model in the original implementation does not predict symlog observations, but predicts observations in the natural space,
    #       despite the encoder taking symlog observations
    # observation_loss = frl.losses.mse_loss(pred_obs, XXX, dims=2)  # CNN, TODO: Detect
    # observation_loss = frl_losses.mse_loss(frl_distributions.symlog(pred_obs), frl_distributions.symlog(batch['obs'])).clip(min=1e-8)  # MLP
    # Bugfix here for symlog on `pred_obs`, April 11, 2026, 4:10 AM

    # Custom reconstruction losses for CNN, MLP, and attention
    mlp_observation_loss = cnn_observation_loss = attention_reconstruction_loss = attention_existence_loss = 0
    target_obs = frl_models.extract_representation(batch['obs'], world_model.decoder_model._decoder_specs)  # Format the observations into specified segments
    for pred_obs_segment, target_obs_segment, decoder_spec in zip(pred_obs, target_obs, world_model.decoder_model._decoder_specs):
        # MLP case
        if decoder_spec['type'].upper() == 'MLP':
            # TODO: The clip here is not in jax, apparently, but it is in SheepRL
            mlp_observation_loss = mlp_observation_loss + frl_losses.mse_loss(
                frl_distributions.symlog(pred_obs_segment), frl_distributions.symlog(target_obs_segment)).clip(min=1e-8)

        # CNN case
        elif decoder_spec['type'].upper() == 'CNN':
            cnn_observation_loss = cnn_observation_loss + frl_losses.mse_loss(pred_obs_segment, target_obs_segment, dims=3)

        # Attention case (Novel)
        elif decoder_spec['type'].upper() == 'ATTENTION':
            # Loop through all segments/entity types and sum reconstruction losses
            # NOTE: This weighs all entity types equally, which may not be ideal
            segment_reconstruction_loss = segment_existence_loss = 0
            entities_pred_list, existence_logits_list = pred_obs_segment
            for entities_pred, existence_logits, entities_target in zip(entities_pred_list, existence_logits_list, target_obs_segment):
                subsegment_reconstruction_loss, subsegment_existence_loss = frl_losses.attention_reconstruction_loss(
                    frl_distributions.symlog(entities_pred),
                    frl_distributions.symlog(entities_target),
                    existence_logits)
                segment_reconstruction_loss = segment_reconstruction_loss + subsegment_reconstruction_loss
                segment_existence_loss = segment_existence_loss + subsegment_existence_loss

            # Record
            attention_reconstruction_loss = attention_reconstruction_loss + segment_reconstruction_loss
            attention_existence_loss = attention_existence_loss + segment_existence_loss

        # Unsupported decoder spec type
        else:
            raise ValueError(f'Unsupported decoder spec type, "{decoder_spec['type']}", must be one of "MLP", "CNN", or "ATTENTION".')

    # Record
    observation_loss = mlp_observation_loss + cnn_observation_loss + attention_reconstruction_loss + attention_existence_loss

    # Compute reward loss (pr)
    dist_pred_rewards = frl_distributions.TwoHot(pred_rewards)  # NOTE: Default reward range is -20, 20 before symexp
    reward_loss = - dist_pred_rewards.log_prob(batch['rewards'])  # TODO: Verify that this is the correct timestep, same with continues

    # Compute continue loss (pc)
    # NOTE: We set `validate_args=False` so that we can use non-bool targets
    dist_pred_continues = torch.distributions.Bernoulli(logits=pred_continues.squeeze(-1), validate_args=False)
    # https://github.com/danijar/dreamerv3/blob/b65cf81a6fb13625af8722127459283f899a35d9/dreamerv3/agent.py#L174-L176
    # NOTE: `self.config.contdisc` : `con *= 1 - 1 / self.config.horizon(333)` is covered by gamma for horizon and lambda values,
    #       but SheepRL doesn't discount the continues while Jax does - equivalent of `contdisc = False`, where contdisc controls all gamma applications
    #       except for the lambda values used in the critic loss
    # Changed April 11, 2026, 11:57 AM to match Jax by multiplying continues by gamma
    continue_targets = (1 - 1. * batch['terminations']) * training_gamma
    continue_loss = - dist_pred_continues.log_prob(continue_targets)

    # Reshape priors and posteriors
    posteriors_logits = posteriors_logits.reshape([*posteriors_logits.shape[:-1], -1, categorical_bins])
    priors_logits = priors_logits.reshape([*priors_logits.shape[:-1], -1, categorical_bins])

    # KL balancing
    dynamic_loss = torch.distributions.kl_divergence(
        torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
        torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=priors_logits), 1))
    representation_loss = torch.distributions.kl_divergence(
        torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1))
    free_nats = torch.tensor(training_free_nats, device=dynamic_loss.device)
    dynamic_loss = torch.maximum(dynamic_loss, free_nats)  # Free nats
    representation_loss = torch.maximum(representation_loss, free_nats)  # Free nats
    kl_loss = training_kl_dyn * dynamic_loss + training_kl_rep * representation_loss

    # Total loss
    reconstruction_loss = (training_kl_reg * kl_loss + observation_loss + reward_loss + training_continue_reg * continue_loss).mean()

    # Log to tensorboard
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('Loss/World', reconstruction_loss.item(), environment_step)
        tensorboard_writer.add_scalar('World/Observation', observation_loss.mean().item(), environment_step)
        if isinstance(mlp_observation_loss, torch.Tensor):
            tensorboard_writer.add_scalar('Observation/MLP', mlp_observation_loss.mean().item(), environment_step)
        if isinstance(cnn_observation_loss, torch.Tensor):
            tensorboard_writer.add_scalar('Observation/CNN', cnn_observation_loss.mean().item(), environment_step)
        if isinstance(attention_reconstruction_loss, torch.Tensor):
            tensorboard_writer.add_scalar('Observation/Attention_Reconstruction', attention_reconstruction_loss.mean().item(), environment_step)
            tensorboard_writer.add_scalar('Observation/Attention_Existence', attention_existence_loss.mean().item(), environment_step)
        tensorboard_writer.add_scalar('World/Reward', reward_loss.mean().item(), environment_step)
        tensorboard_writer.add_scalar('World/Continue', continue_loss.mean().item(), environment_step)
        tensorboard_writer.add_scalar('World/Dynamic', dynamic_loss.mean().item(), environment_step)
        tensorboard_writer.add_scalar('World/Representation', representation_loss.mean().item(), environment_step)

    # Step world model
    world_model.optimizer.zero_grad()
    reconstruction_loss.backward()
    nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=training_world_model_grad_clip)
    world_model.optimizer.step()

    # Initialize storage for behavior learning
    imagined_prior, imagined_hidden_state = posteriors.detach(), hidden_states.detach()
    imagined_trajectories = [torch.cat((imagined_prior, imagined_hidden_state), dim=-1)]
    imagined_actions = []

    # Imagine for `imagination_horizon` steps
    for _ in range(training_imagination_horizon):
        # Compute action based on previous prior + hidden state
        # NOTE: This detach is very important, as it prevents the actor model fitting on the world model recursively, which is unstable
        actions, action_distributions = actor_critic_model.actor_model(imagined_trajectories[-1].detach())  # This detach was added April 8th, 2026, and after cartpole 19-07

        # Imagine future prior
        ret = world_model.rssm_model(actions, imagined_prior, imagined_hidden_state)
        imagined_prior = ret['prior']
        imagined_hidden_state = ret['hidden_state']

        # Record
        imagined_actions.append(actions)
        imagined_trajectories.append(torch.cat((imagined_prior, imagined_hidden_state), dim=-1))

    # Stack to (batch, sequence, horizon(-1), features)
    imagined_trajectories = torch.stack(imagined_trajectories, dim=2)
    imagined_actions = torch.stack(imagined_actions, dim=2)

    # Compute predicted rewards, values, and continues
    pred_rewards = world_model.reward_model(imagined_trajectories)
    pred_rewards = frl_distributions.TwoHot(pred_rewards).mean
    # Changed April 11, 2026, 4:23 AM, to match https://github.com/danijar/dreamerv3/blob/b65cf81a6fb13625af8722127459283f899a35d9/dreamerv3/agent.py#L397-L400
    # TODO: Maybe add parameter for slow target
    value_model = actor_critic_model.target_critic_model
    pred_values = value_model(imagined_trajectories)
    pred_values = frl_distributions.TwoHot(pred_values).mean
    pred_continues = world_model.continue_model(imagined_trajectories[:, :, 1:]).squeeze(-1)
    # Changed April 11, 2026, 4:36 AM, to use probs instead of binary, matching https://github.com/danijar/dreamerv3/blob/b65cf81a6fb13625af8722127459283f899a35d9/dreamerv3/agent.py#L411-L415
    # SheepRL contradicts the jax implementation here by using binary continues
    # pred_continues = torch.distributions.Bernoulli(logits=pred_continues.squeeze(-1)).mode
    pred_continues = torch.sigmoid(pred_continues)  # Could also do Bernoulli.log_prob(1).exp()
    pred_continues = torch.cat((continue_targets.unsqueeze(-1), pred_continues), dim=-1)  # Use actual continues where possible

    # Compute lambda values
    # TODO: Maybe at some point report these or anticipated rewards to dashboard
    # NOTE: These are not advantages
    # https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/algos/dreamer_v3/utils.py#L66
    # https://github.com/danijar/dreamerv3/blob/b65cf81a6fb13625af8722127459283f899a35d9/dreamerv3/agent.py#L482
    lambda_values = [pred_values[:, :, -1]]  # Add bootstrapping value
    interm = pred_rewards[:, :, 1:] + pred_continues[:, :, 1:] * training_gamma * pred_values[:, :, 1:] * (1 - training_lmbda)
    for i in range(pred_rewards[:, :, 1:].shape[2] - 1, -1, -1):
        lambda_values.append( interm[:, :, i] + pred_continues[:, :, 1:][:, :, i] * training_gamma * training_lmbda * lambda_values[-1] )
    lambda_values = torch.stack(lambda_values[:0:-1], dim=2)

    # Using previous values as baseline, compute advantage
    baseline = pred_values[:, :, :-1]
    norm_low, norm_range = utility_modules.lambda_normalizer(lambda_values)
    # NOTE: The jax implementation normalizes lambda values and advantage separately, but SheepRL normalizes together,
    #       https://github.com/danijar/dreamerv3/blob/b65cf81a6fb13625af8722127459283f899a35d9/dreamerv3/agent.py#L407-L410,
    #       which matches with Jax retnorm being the only enabled normalizer
    # normalized_lambda_values = (lambda_values - norm_low) / norm_range
    # normalized_baseline = (baseline - norm_low) / norm_range
    advantage = (lambda_values - baseline) / norm_range  # Changed April 11, 2026, 5:21 AM, to not normalize the baseline
    # Changed back April 11, 2026, 11:36 AM
    # Changed back again April 11, 2026, 12:28 PM, to not normalize baseline
    # Changed to apply scale to both April 11, 2026, 1:47 PM, matching Jax implementation

    # Compute discounts based on horizon, and disregard after non-continues
    horizon_discount = torch.cumprod(pred_continues[:, :, :-1] * training_gamma, dim=2) / training_gamma
    horizon_discount = horizon_discount.detach()

    # Compute objective by summing action log probabilities
    # NOTE: We do not need to bootstrap in the truncation case, as we can just imagine beyond that
    # TODO: Maybe this could be computed earlier? We could do it in the imagination loop, but it would propagate gradients throughout
    #       the horizon on the side of the distribution, which is not in the implementation.
    _, action_distributions = actor_critic_model.actor_model(imagined_trajectories[:, :, :-1].detach())
    objective = []
    for dist, action in zip(action_distributions, imagined_actions.split(
        [action.output_dim for action in actor_critic_model.actor_model._actions], dim=-1)):
        objective.append(dist.log_prob(action.detach()))
    objective = torch.stack(objective, dim=-1).sum(dim=-1)  # Vectorize, apparently faster than `sum`.
    # objective = advantage  # Continuous, TODO: Automatically check
    objective = advantage.detach() * objective

    # Compute entropy
    entropy = 3e-4 * torch.stack([dist.entropy() for dist in action_distributions], dim=-1).sum(dim=-1)

    # Compute actor loss
    actor_loss = - horizon_discount * (objective + entropy)
    actor_loss = actor_loss.mean()

    # Step actor model
    actor_critic_model.actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor_critic_model.actor_model.parameters(), max_norm=training_actor_model_grad_clip)
    actor_critic_model.actor_optimizer.step()

    # Log to tensorboard
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('Loss/Actor', actor_loss.item(), environment_step)
        tensorboard_writer.add_scalar('Actor/Objective', - objective.mean().item(), environment_step)
        tensorboard_writer.add_scalar('Actor/Entropy', - entropy.mean().item(), environment_step)

    # Compute predicted critic and target critic values
    # TODO: Could do this earlier, but the stored gradients may complicate backpropagation, time-wise.
    pred_values = actor_critic_model.critic_model(imagined_trajectories[:, :, :-1].detach())
    dist_pred_values = frl_distributions.TwoHot(pred_values)
    pred_target_values = frl_distributions.TwoHot(actor_critic_model.target_critic_model(imagined_trajectories[:, :, :-1].detach())).mean

    # Compute critic loss
    # April 11, 2026, 4:51 AM: Lambda values are optionally normalized here in Jax, but not in SheepRL
    # Further, the return norm has `{impl: perc, rate: 0.01, limit: 1.0, perclo: 5.0, perchi: 95.0, debias: False}`, while others have `{impl: none, rate: 0.01, limit: 1e-8}` and default matching lo and hi
    # NOTE: In the jax implementation, this uses a normalizer equivalent to the one used for the actor loss, but of a separate instance, so it is equivalent, https://github.com/danijar/dreamerv3/blob/b65cf81a6fb13625af8722127459283f899a35d9/dreamerv3/agent.py#L417-L422
    target_loss = - dist_pred_values.log_prob(lambda_values.detach())
    # Changed back April 11, 2026, 12:28 PM, to not normalize lambda values here
    divergence_loss = - dist_pred_values.log_prob(pred_target_values.detach())  # Don't stray too far from target critic values
    critic_loss = target_loss + divergence_loss
    critic_loss = horizon_discount * critic_loss  # Discount based on horizon
    critic_loss = critic_loss.mean()

    # Step critic model
    actor_critic_model.critic_optimizer.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(actor_critic_model.critic_model.parameters(), max_norm=training_critic_model_grad_clip)
    actor_critic_model.critic_optimizer.step()

    # Log to tensorboard
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('Loss/Critic', critic_loss.item(), environment_step)
        tensorboard_writer.add_scalar('Critic/Target', target_loss.mean().item(), environment_step)
        tensorboard_writer.add_scalar('Critic/Divergence', divergence_loss.mean().item(), environment_step)


@frl_utilities.optional_flatten_cfg(exceptions=[])
def compute_actions(
    world_model: frl_utilities.ContainerModule,
    actor_critic_model: frl_utilities.ContainerModule,
    obs: torch.Tensor = None,
    actions: torch.Tensor = None,
    posteriors: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    initializations: torch.Tensor = None,
    compute_prior: bool = False,
) -> dict[str, Union[np.ndarray, torch.Tensor]]:
    """Compute an action given the current observation.

    :param world_model: The world model for the Dreamer agent.
    :type world_model: frl_utilities.ContainerModule
    :param actor_critic_model: The actor-critic model for the Dreamer agent.
    :type actor_critic_model: frl_utilities.ContainerModule
    :param obs: The current observation from the environment, as a tensor of shape (batch_dim, obs_dim). (Default: ``None``)
    :type obs: torch.Tensor
    :param actions: The actions from the previous step. If not provided, will use a default. (Default: ``None``)
    :type actions: torch.Tensor
    :param posteriors: The posterior states from the previous step. If not provided, will use a default. (Default: ``None``)
    :type posteriors: torch.Tensor
    :param hidden_states: The hidden states from the previous step. If not provided, will use a default. (Default: ``None``)
    :type hidden_states: torch.Tensor
    :param initializations: The initializations (`terminations | truncations`) from the previous step. If not provided, will use a default. (Default: ``None``)
    :type initializations: torch.Tensor
    :param compute_prior: Whether to compute the prior state for the current step. If ``False``, will only compute the posterior. (Default: ``False``)
    :type compute_prior: bool
    :return: A dictionary containing the environment actions, actions, posteriors, hidden states, and world model output.
    :rtype: dict[str, Union[np.ndarray, torch.Tensor]]

    """
    # Make sure user is passing initializations
    if actions is not None and initializations is None:
        raise ValueError('Must provide initializations (`terminations | truncations`) based on previous step.')

    # Embed observation
    if obs is not None:
        obs = obs.to(world_model.encoder_model.parameters().__next__().dtype)  # Cast to same dtype as model parameters
        extracted_obs = frl_models.extract_representation(obs, world_model.encoder_model._encoder_specs)
        embedded_obs = world_model.encoder_model(extracted_obs)

    # Compute hidden states
    world_model_out = world_model.rssm_model(
        actions,
        posteriors,
        hidden_states,
        embedded_obs,
        initializations,
        batch_dim=obs.shape[0],
        compute_prior=compute_prior,
    )
    hidden_states = world_model_out['hidden_state']
    posteriors = world_model_out['posterior']

    # Compute actions
    actions, action_distributions = actor_critic_model.actor_model(torch.cat((posteriors, hidden_states), dim=-1))

    # Sample final actions
    env_actions = frl_actions.simplify_actions(actions, actor_critic_model.actor_model._actions).detach().cpu().numpy()
    if env_actions.shape[-1] == 1:
        # Remove final dim when using action spaces of 1 dim
        env_actions = env_actions.squeeze(-1)

    return {
        'env_actions': env_actions,
        'actions': actions,
        **world_model_out,
    }


@frl_utilities.optional_flatten_cfg(exceptions=[])
def train_loop(
    # Manually supplied parameters
    envs: frl_environments.VectorizedEnvironment,
    world_model: frl_utilities.ContainerModule,
    actor_critic_model: frl_utilities.ContainerModule,
    utility_modules: frl_utilities.Container,
    tensorboard_writer: tensorboard_SummaryWriter_cls = None,
    # Loaded model parameters
    start_environment_step: int = 0,
    # Checkpointing parameters
    checkpoint_dir: str = None,
    checkpoint_frequency: int = 5_000,
    # Evaluation parameters
    eval_frequency: int = 10**3,
    # Training parameters
    training_steps: int = 10**6,
    training_pretrain_steps: int = 1024,
    training_critic_target_update_freq: int = 1,
    training_batch_size: int = 16,
    training_sequence_length: int = 64,
    training_tau: float = 0.02,
    # Initialization parameters
    seed: int = None,
    eval_seed: int = 42,
    **kwargs: dict[str, Any],
) -> None:
    """Train the Dreamer agent in the given environment.

    :param envs: The vectorized environments to train in.
    :type envs: frl_environments.VectorizedEnvironment
    :param model_actions: A list of actions for the model to use. Should correspond to the environment action space.
    :type model_actions: list[frl_actions.Action]
    :param world_model: The world model for the Dreamer agent.
    :type world_model: frl_utilities.ContainerModule
    :param actor_critic_model: The actor-critic model for the Dreamer agent.
    :type actor_critic_model: frl_utilities.ContainerModule
    :param utility_modules: The utility modules for the Dreamer agent.
    :type utility_modules: frl_utilities.Container
    :param tensorboard_writer: The TensorBoard writer for logging, if desired. (Default: ``None``)
    :type tensorboard_writer: torch.utils.tensorboard.SummaryWriter
    :param start_environment_step: The initial environment step. Used for resuming training from a checkpoint. (Default: ``0``)
    :type start_environment_step: int
    :param checkpoint_dir: The directory to save checkpoints in. (Default: ``None``)
    :type checkpoint_dir: str
    :param checkpoint_frequency: The frequency (in environment steps) for saving checkpoints. (Default: ``5_000``)
    :type checkpoint_frequency: int
    :param env_name: The name of the environment for evaluation purposes. (Default: ``None``)
    :type env_name: str
    :param eval_frequency: The frequency (in environment steps) for performing evaluations. (Default: ``10**3``)
    :type eval_frequency: int
    :param training_steps: The total number of environment steps to train for. (Default: ``10**6``)
    :type training_steps: int
    :param training_pretrain_steps: The number of environment steps to pretrain the world model before starting training. (Default: ``1024``)
    :type training_pretrain_steps: int
    :param training_critic_target_update_freq: The interval for soft updates of the target critic. (Default: ``1``)
    :type training_critic_target_update_freq: int
    :param training_batch_size: The batch size for training. (Default: ``16``)
    :type training_batch_size: int
    :param training_sequence_length: The batch sample sequence length for training. (Default: ``64``)
    :type training_sequence_length: int
    :param training_tau: The tau parameter for soft updates of the target critic. (Default: ``0.02``)
    :type training_tau: float
    :param seed: The random seed for reproducibility. (Default: ``None``)
    :type seed: int
    :param eval_seed: The random seed for evaluation. (Default: ``42``)
    :type eval_seed: int
    :param kwargs: Catch keyword arguments for compatibility with ``utilities.optional_flatten_cfg``.
    :type kwargs: dict[str, Any]

    """
    # Detect device
    device = next(world_model.parameters()).device

    # Initialize variables
    actions = posteriors = hidden_states = initializations = None
    rewards, terminations, truncations = np.zeros(envs.num_envs), np.zeros(envs.num_envs, dtype=bool), np.zeros(envs.num_envs, dtype=bool)

    # Tracking
    cumulative_rewards = np.zeros(envs.num_envs)
    cumulative_gradient_steps = 0  # NOTE: These will reset on checkpoint load
    cumulative_episodes = 0

    # Loop for specified number of iterations
    obs, info = envs.reset(seed=seed)
    for environment_step in range(start_environment_step + envs.num_envs, training_steps + envs.num_envs, envs.num_envs):
        # Compute an action using the model
        # TODO: Make this easier for inference
        if environment_step > training_pretrain_steps:
            with torch.no_grad():
                ret = compute_actions(
                    obs=torch.from_numpy(obs).to(device),
                    world_model=world_model,
                    actor_critic_model=actor_critic_model,
                    actions=actions,
                    posteriors=posteriors,
                    hidden_states=hidden_states,
                    initializations=initializations,
                )
                env_actions = ret['env_actions']
                actions = ret['actions']
                posteriors = ret['posterior']
                hidden_states = ret['hidden_state']

        # Compute action randomly if not training yet
        else:
            # TODO: Revise this flow
            sampled_actions = envs.action_sample().reshape(envs.num_envs, -1)
            actions = frl_actions.construct_actions(torch.tensor(sampled_actions, dtype=torch.get_default_dtype()), actor_critic_model.actor_model._actions)
            env_actions = sampled_actions.squeeze(-1) if sampled_actions.shape[-1] == 1 else sampled_actions

        # Record to buffer
        utility_modules.buffer.add({
            # Environment-related experiences
            'obs': obs,
            'rewards': rewards,
            'terminations': terminations,
            'truncations': truncations,
            # Actions
            'actions': actions.detach().cpu().numpy(),
        })

        # Step environment
        obs, rewards, terminations, truncations, infos = envs.step(env_actions)
        initializations = torch.tensor(terminations | truncations, dtype=torch.bool, device=device)

        # Iterate and record rewards if done, and also track cumulative rewards by environment
        cumulative_rewards += rewards
        for i in range(envs.num_envs):
            # Track per-environment cumulative rewards
            # NOTE: The tracked step is multiplied by the number of environments, but we keep it that way for easier comparison with total environment steps
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(f'Reward/Environment_{i}', cumulative_rewards[i], environment_step)

            if terminations[i] or truncations[i]:
                # TODO: Maybe differentiate between terminations and truncations in logging
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar('Reward/Episode', cumulative_rewards[i], environment_step)
                cumulative_rewards[i] = 0
                cumulative_episodes += 1

        # Train
        if environment_step > training_pretrain_steps:
            gradient_steps = utility_modules.ratio(environment_step - training_pretrain_steps)
            for _ in range(gradient_steps):
                # Sample batch of experiences from buffer
                batch = utility_modules.buffer.sample(training_batch_size, sequence_length=training_sequence_length)
                batch = frl_buffers.convert_samples_to_tensors(batch, device=device)
                batch['obs'] = batch['obs'].to(world_model.encoder_model.parameters().__next__().dtype)  # TODO: Maybe a cleaner way?

                # Iterate target critic model towards critic model
                # NOTE: Assumes initialization took care of initial copy
                if cumulative_gradient_steps % training_critic_target_update_freq == 0:
                    for target_param, param in zip(actor_critic_model.target_critic_model.parameters(), actor_critic_model.critic_model.parameters()):
                        target_param.data.copy_(training_tau * param.data + (1 - training_tau) * target_param.data)

                # Train
                learning_step(
                    batch=batch,
                    world_model=world_model,
                    actor_critic_model=actor_critic_model,
                    utility_modules=utility_modules,
                    tensorboard_writer=tensorboard_writer,
                    environment_step=environment_step,
                    **kwargs)

                # Iterate
                cumulative_gradient_steps += 1

        # Evaluate
        if (
            eval_frequency
            and environment_step % eval_frequency < envs.num_envs
            and tensorboard_writer is not None
        ):
            # Run evaluation episode
            frames, fps = evaluate(
                env=envs.copy(num_envs=1, allow_rendering=True),
                world_model=world_model,
                actor_critic_model=actor_critic_model,
                seed=eval_seed)

            # Output to gif
            frl_utilities.export_frames(
                os.path.join(tensorboard_writer.get_logdir(), f'eval_{environment_step:07d}.gif'),
                frames,
                fps=fps)

            # Log video to tensorboard (For some reason, causes permission error and is broken)
            # tensorboard_writer.add_video(  # (batch, time, channels, height, width)
            #     'Evaluation/Video',
            #     np.expand_dims(video_out.transpose(0, 1, 4, 2, 3), 0),
            #     environment_step,
            #     fps=fps)

        # Checkpoint
        if checkpoint_dir is not None and environment_step % checkpoint_frequency < envs.num_envs:
            save_models(
                path=os.path.join(checkpoint_dir, f'checkpoint_{environment_step:07d}.pt'),
                world_model=world_model,
                actor_critic_model=actor_critic_model,
                utility_modules=utility_modules)


@frl_utilities.optional_flatten_cfg(exceptions=[])
def evaluate(
    env: frl_environments.VectorizedEnvironment,
    world_model: frl_utilities.ContainerModule,
    actor_critic_model: frl_utilities.ContainerModule,
    seed: int = None,
) -> tuple[np.ndarray, float]:
    """Run a single evaluation episode for the Dreamer agent.

    :param env: The name of the environment to evaluate in.
    :type  env: frl_environments.VectorizedEnvironment
    :param world_model: The world model for the Dreamer agent.
    :type world_model: frl_utilities.ContainerModule
    :param actor_critic_model: The actor-critic model for the Dreamer agent.
    :type actor_critic_model: frl_utilities.ContainerModule
    :param seed: The random seed for the environment. (Default: ``None``)
    :type seed: int
    :return: A tuple containing the video frames of the episode (frames, height, width, channels) and the frames per second
        (FPS) of the environment rendering.
    :rtype: tuple[np.ndarray, float]

    """
    # Check that there is only one environment
    if env.num_envs != 1:
        raise ValueError(f'Expected one environment provided to `evaluate`, got {env.num_envs}.')

    # Infer device
    device = next(world_model.parameters()).device

    # Create environment with rendering
    obs, info = env.reset(seed=seed)
    actions = posteriors = hidden_states = initializations = None
    frames = [env.render().squeeze(0)]

    # Loop until episode ends
    while True:
        # Compute action using model
        with torch.no_grad():
            ret = compute_actions(
                obs=torch.from_numpy(obs).to(device),  # TODO: Maybe auto-detect in `compute_actions`
                world_model=world_model,
                actor_critic_model=actor_critic_model,
                actions=actions,
                posteriors=posteriors,
                hidden_states=hidden_states,
                initializations=initializations,
            )
            env_actions = ret['env_actions']
            actions = ret['actions']
            posteriors = ret['posterior']
            hidden_states = ret['hidden_state']

        # Step environment
        obs, rewards, terminated, truncated, infos = env.step(env_actions)
        frames.append(env.render().squeeze(0))

        # Update initializations
        initializations = torch.from_numpy(terminated | truncated).to(dtype=bool, device=device)

        # Check for episode end
        if terminated or truncated:
            break

    # Stack video frames
    frames = np.stack(frames, axis=0)  # (time, height, width, channels)

    # Return video frames
    return frames, env.render_fps

