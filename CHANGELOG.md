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
