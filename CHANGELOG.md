# 0.18.0
- Add automatic resetting to `VectorizedRLGymEnvironment`
- Add arguments to `CNNEncoder` and `CNNDecoder` to accomodate differing image sizes
- Add `environments.CloseReward` for closeness of car and ball in `RLGym` environments, with additional `use_diff` configuration parameter
- Add `models.AttentionEncoder` and `models.AttentionDecoder` models
- Add `models.CompoundEncoder` and `models.CompoundDecoder` models for combining CNN, MLP, and attention observations with utility function `models.extract_representation`
- Add new losses from segmented encoding to tensorboard
- Add `output_dim` property to all encoders
- Begin training on `RLGym` environment
- Compatibility with new model architectures
- Fix `tensorboard` dependency typing issue when not installed
- Fix training and evaluation environment seeding
- Formalize `VectorizedRLGymEnvironment` implementation in `environments` module
- Implement `losses.hungarian_loss` and `losses.attention_reconstruction_loss` for attention decoder losses
- Implement CNN and novel attention loss in `dreamer.learning_step` function
- `README` updates for transformer architecture
- Replace single encoder with compound encoder throughout `dreamer` module and training loop

# 0.17.0
- Add `environments` module with `VectorizedEnvironment` subclasses and `ENVIRONMENT_IDENTIFIERS` enum, generalizing environments in the library
- Add `group` parameter to `env` config corresponding to `ENVIRONMENT_IDENTIFIERS`
- Add mask to buffer `add` method to allow for differing numbers of recorded memories per step
- Begin outlining `RLGym` integration
- Bugfix for `num` parameter on replicate action creation
- Separate environment and action constructors in `dreamer` module

# 0.16.2
- Bugfix for checkpoint saving on wrong offset

# 0.16.1
- Add presets based on parameter numbers from original implementation
- Add units for training step saving in `Dreamer` notebook
- Adjust environment step offset to be more correct
- Allow replicate action creation from configuration files using `num` parameter
- `Walker2D` preview in `README`

# 0.16.0
- Add configurations for `Pusher-v5`, `Walker2D-v5`, and `Humanoid-v5`
- Add in-notebook model evaluation and export to `examples/images` folder
- Add `utilities.export_gif` with automatic and adjustable frame skipping
- Add optional calculation of prior for `models.RSSM`
- Change model construction behavior to line up with original
- Fix citation version
- Reorganize `utilities` module
- Small optimizations to `dreamer` inference
- Update `Hopper` results in `README`

# 0.15.1
- Add more details to `README`
- Add `tensor_log_prob` argument to `distributions.TwoHot`
- Allow custom folder naming in `Dreamer` notebook
- Iterate implementation of upon `TwoHotDiscretizedContinuous`
- Rename library to `DreamerX`

# 0.15.0
- Add checkpoint resume to notebook
- Add environment start state
- Add MuJoCo `Reacher-v5` configuration
- Add preliminary `Hopper-v5` results
- Change preview GIFs to loop

# 0.14.1
- Add trained model for `Ant-v5` in `README`
- Begin training model for `Hopper-v5` with associated config
- Change coloring for results display in `README`

# 0.14.0
- Add checkpointing
- Add preview for `BipedalWalker-v3` model in `README`
- Add tensor casting in `MLPEncoder`
- Begin training for `Ant-v5` in `MuJoCo`
- Fix logging chronology for episode rewards
- Fix small performance bug with non-contiguous tensor bucketization
- Reorganize file structure
- Train new `LunarLander-v3` model

# 0.13.0
- Add `dreamer.compute_actions` function for easier use of model
- Add `dreamer.evaluate` function with ability to export video, and integration with `dreamer.train_loop` for upload to TensorBoard
- Add `dreamer.save_models` and `dreamer.load_models`
- Add `dreamer.train_loop` and `dreamer.learning_step` functions, migrating from the experimental notebook
- Add `exclusions` argument to `utilities.optional_flatten_cfg`
- Add `gym` optional dependencies to `make build` and `requirements-dev.txt`, also fixing autodeployment of documentation
- Add per-environment cumulative reward logging by step
- Add `use_symlog` argument to `models.MLPEncoder` for symlog normalization of observations
- Critical bugfixes, mainly for scaling, normalization, and discounting to match with the Jax implementation
- Fix soft critic target update logic
- Move rendering and logging requirements to `extras`
- Rename main notebook to `Dreamer`
- Separate config files for model and environment
- Train new models for `CartPole-v1` and `LunarLander-v3`
- Update `README` with benchmarking results and examples

# 0.12.0
- Add `actions.ACTION_IDENTIFIERS` enum for identifying action types from strings
- Add `DotDict` class and multiple config loading, with merging and priority
- Add `dreamer` module and contained method `construct_models`
- Add `utilities.CaseInsensitiveEnumMeta` for case insensitive enum lookup
- Add `utilities.ContainerModule` for combining multiple `nn.Module`s in one class instance
- Add `utilities.load_config` utility
- Add `utilities.optional_flatten_cfg` to feed function arguments from config files
- Explicitly define documentation templates
- Make buffers and `Ratio` modules to allow for easier containerization

# 0.11.0
- Add discretization strategy `actions.TwoHotDiscretizedContinuousAction` and `actions.DiscretizedContinuousAction`
- Add `distributions.identity` utility function
- Add novel `TwoHot` sampling and entropy
- Add preliminary model saving
- Add preliminary tuned configuration file
- Add `state_dict` and `load_state_dict` methods to buffers
- Add `utilities.Ratio` class for computing gradient updates
- Add `learnable_initial_state` argument to `models.RSSM`
- Documentation and typing corrections, default parameter adjustments
- Fix documentation formatting and minor issues
- Numerous bugfixes, including gradient detach in action generation and proper loss formulations
- Revise sampling strategy `buffers.IndependentVectorizedBuffer` to sample from all buffers randomly
- Test to optimality on `CartPole-v1`

# 0.10.0
- Add `actions.construct_actions` utility function
- Add `construct` method to `actions.Action` and all child actions
- Add initialization methods
- Add random action sampling before training
- Add `tensorboard` integration
- Add `utilities` module with `MovingMinMaxScaler` lambda value scaler
- Change parameter default values across most models
- Fix advantage utilization, observation loss clipping, `losses.mse_loss`, and actor loss formulation
- Fix continuous action `actions.ContinuousActions` sample distribution event dimension
- Fix `mean` output from `distributions.TwoHot`
- Implement actor and critic losses, target critic model, and missing hyperparameters

# 0.9.0
- Add `symexp` and `TwoHot` to `distributions` module
- Fix and add `dims` argument to `mse_loss`
- Implement full reconstruction loss and world optimizer
- Rename `MSELoss` to `mse_loss`

# 0.8.0
- Add `convert_samples_to_tensors` to `buffers` module for converting from `numpy`
- Add `sequence_length` argument to buffer sampling
- Develop initial training step function
- Fix initializations array in main loop and `models.RSSM` class
- Separate main loop and training step function

# 0.7.0
- Add `buffers` module with `Buffer`, `SequentialBuffer` and `IndependentVectorizedBufferGroup` and implement in main loop
- Incorporate main loop logic with training cutaway

# 0.6.0
- Add `actions` module and definitions for `actions.Action`, `actions.ContinuousActions`, and `actions.DiscreteAction`
- Construct complete inference loop
- Fix development dependencies and `make build` command
- General syntax and naming changes
- Implement actor (`models.Actor`) and critic models
- Implement reward and continue predictions
- Vectorize environments

# 0.5.0
- Add `output_dim` argument to `models.MLP` for unnormalized output
- Fix transition and representation model structures in example notebook
- Revise `models.MLPEncoder` and `models.MLPDecoder` to use `models.MLP` as constructor

# 0.4.0
- Add tentative mode calculation to RSSM stochastic sampling
- Add workflow to automatically generate docs and deploy to GitHub pages
- Fix incorrect `Makefile` library reference

# 0.3.0
- Add Dreamer wrapper for recurrent state space modeling, `models.RSSM`
- Add classic gymnasium environments
- Add general purpose models `models.MLP`, `models.MLPEncoder`, `models.MLPDecoder`, and `models.CNNDecoder`
- Add initial environment loop for testing
- Add recurrent model `models.RecurrentModel` with layer `models.LayerNormGRU` and corresponding unit test
- Add uniform mix `distributions.uniform_mix` and symlog `distributions.symlog`
- Changes to `models.CNNEncoder` final layer
- Display `__init__` methods in documentation
- Group ordering of classes and functions in documentation

# 0.2.0
- Add convolutional encoder `models.CNNEncoder` and channel layer normalization `models.ChannelNorm`
- Add tests for `models` module
- Further refine linting and add module docstrings
- `README.md` formatting fix

# 0.1.0
- Boilerplate, documentation with `Sphinx`, linting with `Ruff`, unit testing
- Initial commit
