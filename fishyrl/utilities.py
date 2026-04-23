"""Utilities for state management and common operations."""

import enum
import functools
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn


class MovingMinMaxScaler(nn.Module):
    """Moving percentile-based min-max scaler for normalizing inputs."""
    def __init__(
        self,
        beta: float = .99,
        frac_low: float = .05,
        frac_high: float = .95,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the ``MovingMinMaxScaler``.

        :param beta: The decay rate for the moving min and max. (Default: ``0.99``)
        :type beta: float
        :param eps: Minimal value for the computed high-low range. (Default: ``1e-8``)
        :type eps: float
        :param frac_low: The lower percentile for scaling. (Default: ``0.05``)
        :type frac_low: float
        :param frac_high: The upper percentile for scaling. (Default: ``0.95``)
        :type frac_high: float

        """
        super().__init__()

        # Parameters
        self._beta = beta
        self._frac_low = frac_low
        self._frac_high = frac_high
        self._eps = torch.tensor(eps, dtype=torch.get_default_dtype())

        # Initialize low and high buffers
        self.register_buffer('_low', torch.zeros((), dtype=torch.get_default_dtype()))
        self.register_buffer('_high', torch.zeros((), dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update and return the low and range estimates.

        :param x: The input tensor to use while updating the estimates.
        :type x: torch.Tensor
        :return: A tuple containing the low estimate and the range estimate.
        :rtype: Tuple[torch.Tensor, torch.Tensor]

        """
        # Detatch input to avoid memory leaks
        x = x.detach()

        # Update low and high estimates
        low = torch.quantile(x, self._frac_low)
        high = torch.quantile(x, self._frac_high)
        self._low = self._beta * self._low + (1 - self._beta) * low
        self._high = self._beta * self._high + (1 - self._beta) * high
        # self._low, self._high = self._low.detach(), self._high.detach()

        # Return low and range
        return self._low, torch.max(self._high - self._low, self._eps)


class Ratio(nn.Module):
    """Module for computing the number of gradient update steps."""
    def __init__(self, ratio: float = 1.) -> None:
        """Initialize ``Ratio``.

        :param ratio: The ratio of gradient update steps to environment steps. (Default: ``1.0``)
        :type ratio: float

        """
        super().__init__()

        # Parameters
        self._ratio = ratio
        self._step = 0

    def __call__(self, step: int) -> int:
        """Compute the number of gradient update steps for the given environment step.

        :param step: The current environment step.
        :type step: int
        :return: The number of gradient update steps to perform.
        :rtype: int

        """
        # Compute the number of gradient update steps
        num_updates = int((step - self._step) * self._ratio)
        self._step += num_updates / self._ratio
        return num_updates

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the module as a dictionary.

        :return: A dictionary containing the state of the module.
        :rtype: dict[str, Any]

        """
        return {'_ratio': self._ratio, '_step': self._step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state of the module from a dictionary.

        :param state_dict: The state dictionary to load from.
        :type state_dict: dict[str, Any]

        """
        self._ratio = state_dict['_ratio']
        self._step = state_dict['_step']


class DotDict(dict):
    """Allow accessing dictionary keys as attributes.

    Attribute access code from https://stackoverflow.com/a/23689767 and SheepRL

    """
    def __init__(self, *args: list, **kwargs: dict[str, Any]) -> None:
        """Initialize the ``DotDict``.

        :param args: Positional arguments to initialize the dictionary.
        :type args: list
        :param kwargs: Keyword arguments to initialize the dictionary.
        :type kwargs: dict[str, Any]

        """
        super().__init__(*args, **kwargs)

        # Crawl nested dictionaries and convert them to dotdicts
        self._crawl(self)

    def _crawl(self, lookup: dict | list) -> None:
        """Recursively convert nested dictionaries to ``DotDict``.

        Crawls through dictionaries and lists.

        :param lookup: The dictionary or list to crawl through.
        :type lookup: dict | list

        """
        if isinstance(lookup, dict):
            pack = lookup.items()
        elif isinstance(lookup, list):
            pack = enumerate(lookup)
        for k, v in pack:
            if isinstance(v, dict):
                lookup[k] = DotDict(v)
                self._crawl(lookup[k])
            elif isinstance(v, list):
                self._crawl(lookup[k])

    # Attribute access to dictionary keys
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ContainerModule(nn.Module):
    """Module for containing multiple submodules."""
    def __init__(self, **modules: nn.Module) -> None:
        """Initialize the ``ContainerModule``.

        :param modules: The submodules to contain.
        :type modules: dict[str, nn.Module]

        """
        super().__init__()

        # Register the submodules
        for name, module in modules.items():
            self.add_module(name, module)


class Container:
    """Container for containing multiple submodules and utilities, without torch integration."""
    def __init__(self, **modules: Any) -> None:  # noqa: ANN401
        """Initialize the ``Container``.

        :param modules: The submodules and utilities to contain.
        :type modules: dict[str, Any]

        """
        # Register the submodules and utilities
        for name, module in modules.items():
            setattr(self, name, module)

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the container as a dictionary.

        :return: A dictionary containing the state of the container.
        :rtype: dict[str, Any]

        """
        state = {}
        for name, module in self.__dict__.items():
            if not hasattr(module, 'state_dict'):
                raise ValueError(f'Module "{name}" does not have a state_dict method.')
            state[name] = module.state_dict()

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state of the container from a dictionary.

        :param state_dict: The state dictionary.
        :type state_dict: dict[str, Any]

        """
        for name, module in self.__dict__.items():
            if not hasattr(module, 'load_state_dict'):
                raise ValueError(f'Module "{name}" does not have a load_state_dict method.')
            if name not in state_dict:
                raise ValueError(f'Module "{name}" not found in state dictionary.')
            module.load_state_dict(state_dict[name])


class CaseInsensitiveEnumMeta(enum.EnumMeta):
    """Enum meta class for case-insensitive lookup."""
    def __getitem__(cls, value: str) -> Any:  # noqa: ANN401
        """Override the default item access to perform case-insensitive lookup.

        :param value: The value to look for.
        :type value: str
        :return: The matching enum member.
        :rtype: Any

        """
        return super().__getitem__(value.upper()).value


def load_config(*paths: list[str], list_behavior: str = 'replace') -> DotDict:
    """Load and merge YAML configuration files into a single ``DotDict``, with priority given to earlier files.

    :param paths: The paths to the YAML configuration files to load.
    :type paths: list[str]
    :param list_behavior: The behavior for merging lists. Can be either 'replace' or 'merge'. (Default: ``'replace'``)
    :type list_behavior: str

    :return: A ``DotDict`` containing the merged configuration.
    :rtype: DotDict

    """
    # Load and merge the YAML files
    cfg = DotDict()
    for path in paths[::-1]:
        with open(path, 'r') as f:
            new_cfg = yaml.load(f, Loader=yaml.SafeLoader)
            _merge_dotdicts(cfg, DotDict(new_cfg), list_behavior=list_behavior)

    return cfg


def _merge_dotdicts(base: DotDict, new: DotDict, list_behavior: str = 'replace') -> None:
    """Merge two ``DotDict`` objects, with priority given to the new one.

    :param base: The base ``DotDict`` to merge into.
    :type base: DotDict
    :param new: The new ``DotDict`` to merge from.
    :type new: DotDict
    :param list_behavior: The behavior for merging lists. Can be either 'replace' or 'merge'. (Default: ``'replace'``)
    :type list_behavior: str

    """
    # Loop through the new dictionary and merge it into the base dictionary
    for k, v in new.items():
        # If the key is in base, examine further
        if k in base:
            # If dictionary, recurse
            if isinstance(base[k], dict) and isinstance(v, dict):
                _merge_dotdicts(base[k], v, list_behavior=list_behavior)
            # If list, either replace or merge
            elif isinstance(base[k], list) and isinstance(v, list):
                if list_behavior == 'replace':
                    base[k] = v
                elif list_behavior == 'merge':
                    base[k].extend(v)
                else:
                    raise ValueError(f'Invalid list behavior strategy, "{list_behavior}".')
            # Otherwise, replace
            else:
                base[k] = v
        # Otherwise, add to base
        else:
            base[k] = v


def optional_flatten_cfg(func: callable = None, exceptions: list[str] = [], exclusions: list[str] = []) -> callable:
    """Decorator to optionallly flatten a config DotDict before passing it to the function.

    :param func: The function to decorate.
    :type func: Callable
    :param exceptions: A list of keys to exclude from flattening. (Default: ``[]``)
        As an example, the key ``'model_default'`` will tell the decorator to not flatten
        ``cfg.model.default.*`` and instead pass it to the function as a DotDict using the
        argument ``'model_default'``, where it would have otherwise passed
        ``'model_default_embedded'``, ``'model_default_blocks'``, etc.
    :type exceptions: list[str]
    :param exclusions: A list of keys to exclude from being passed to the function. (Default: ``[]``)
    :type exclusions: list[str]

    """
    @functools.wraps(func)
    def decorator(f: callable) -> callable:
        @functools.wraps(f)
        def wrapper(*args: list[Any], cfg: DotDict = None, **kwargs: dict[str, Any]) -> Any:  # noqa: ANN401
            # If cfg is provided, flatten and revise kwargs
            if cfg is not None:
                new_kwargs = _flatten_dict(cfg, exceptions=exceptions, exclusions=exclusions)
                new_kwargs.update(kwargs)  # Overwrite cfg kwargs with manual kwargs
                kwargs = new_kwargs
            # Otherwise, just use normally
            return f(*args, **kwargs)

        return wrapper

    # If directly used as a decorator, return the wrapper
    if func:
        return decorator(func)

    # Otherwise, return the decorator to generate the wrapper
    return decorator


def _flatten_dict(
    base: DotDict | dict,
    exceptions: list[str] = [],
    exclusions: list[str] = [],
    sep: str = '_',
    _key: str = '',
    _result: dict = {},
) -> dict:
    """Recursively flatten a nested ``DotDict`` into a single-level ``DotDict`` with concatenated keys.

    :param base: The base ``DotDict`` to flatten.
    :type base: DotDict
    :param exceptions: A list of keys to exclude from flattening. (Default: ``[]``)
        See the documentation for the ``optional_flatten_cfg`` decorator for more details.
    :type exceptions: list[str]
    :param exclusions: A list of keys to exclude from being passed to the function. (Default: ``[]``)
    :type exclusions: list[str]
    :param sep: The separator to use when concatenating keys. (Default: ``'_'``)
    :type sep: str
    :param _key: The current key prefix during recursion. (Default: ``''``)
    :type _key: str
    :param _result: The current result dictionary during recursion. (Default: ``{}``)
    :type _result: dict
    :return: A flattened dictionary with concatenated keys.
    :rtype: dict

    """
    # Loop through items
    for k, v in base.items():
        # Create new key
        new_k = f'{_key}{sep}{k}' if _key else k
        # If value is a dictionary and new key is not in exceptions, recurse
        if isinstance(v, dict) and new_k not in exceptions:
            _flatten_dict(v, exceptions=exceptions, exclusions=exclusions, sep=sep, _key=new_k, _result=_result)
        # Otherwise, add to result if new key is not in exclusions
        elif new_k not in exclusions:
            _result[new_k] = v
    return _result


def export_frames(path: str, frames: np.ndarray, fps: int = 30, max_fps: int = None, **kwargs: dict[str, Any]) -> None:
    """Export a sequence of frames to a GIF or video file.

    :param path: The path of the resultant GIF or video file.
    :type path: str
    :param frames: The sequence of frames to export, as a numpy array of shape (time, height, width, channels).
    :type frames: np.ndarray
    :param fps: Frames per second. (Default: ``30``)
    :type fps: int
    :param max_fps: The maximum frames per second to use when exporting. Will drop frames if needed. By default, will cut off at 30 if
        the file is a GIF. (Default: ``None``)
    :type max_fps: int
    :param kwargs: Additional keyword arguments to pass to the imageio.mimsave function.
    :type kwargs: dict[str, Any]

    """
    # Import if needed
    import imageio

    # Default kwargs
    new_kwargs = {}
    if path.split('.')[-1].upper() == 'GIF':
        max_fps = 30 if max_fps is None else max_fps
        new_kwargs.update({'loop': 0})  # Default GIF kwargs
    new_kwargs.update(kwargs)

    # Drop frames if fps is greater than max_fps
    if max_fps is not None and fps > max_fps:
        step = int(np.ceil(fps / max_fps))
        frames = frames[::step]
        fps = int(fps / step)

    # Save the frames as a GIF
    # TODO: Maybe add delay on loop? (https://github.com/imageio/imageio/issues/1073#issuecomment-2040188027)
    imageio.mimsave(path, frames, fps=fps, **new_kwargs)


# Taken from SheepRL
# https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/algos/dreamer_v3/utils.py#L143
def init_weights(m):  # noqa
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# Taken from SheepRL
# https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/algos/dreamer_v3/utils.py#L170
def uniform_init_weights(given_scale):  # noqa
    def f(m):  # noqa
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f
