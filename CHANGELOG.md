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
