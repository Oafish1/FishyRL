"""Buffer classes for storing and managing experiences."""

import warnings
from abc import abstractmethod
from typing import Any

import numpy as np
import torch


class Buffer:
    """Class template for buffers."""
    # TODO: Maybe add more properties here, pending use cases
    @property
    @abstractmethod
    def size(self) -> int:
        """The current number of experiences stored in the buffer.

        :type: int

        """
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a dictionary containing the state of the buffer for saving.

        :return: A dictionary containing the state of the buffer.
        :rtype: dict[str, Any]

        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state of the buffer from a dictionary.

        :param state_dict: A dictionary containing the state of the buffer, usually obtained from ``state_dict()``.
        :type state_dict: dict[str, Any]

        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset/initialize the buffer."""
        pass

    @abstractmethod
    def add(self, experience: dict[str, np.ndarray]) -> None:
        """Add an experience to the buffer.

        :param experience: The experience to add, represented as a dictionary of numpy arrays.
        :type experience: dict[str, np.ndarray]

        """
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of experiences from the buffer.

        :param batch_size: The number of experiences to sample.
        :type batch_size: int
        :return: A batch of experiences, represented as a dictionary of numpy arrays.
        :rtype: dict[str, np.ndarray]

        """
        pass


class SequentialBuffer(Buffer):
    """A buffer that stores experiences sequentially."""
    def __init__(self, capacity: int, validate_keys: bool = True, seed: int = None) -> None:
        """Initialize the buffer.

        :param capacity: The maximum number of experiences to store.
        :type capacity: int
        :param validate_keys: Whether to validate that all experiences have the same keys.
        :type validate_keys: bool

        """
        super().__init__()

        # Parameters
        self._capacity = capacity
        self._validate_keys = validate_keys

        # Initialize rng
        self._rng = np.random.default_rng(seed=seed)

        # Initialize buffer
        self.reset()

    @property
    def capacity(self) -> int:
        """The maximum number of experiences the buffer can store.

        :type: int

        """
        return self._capacity

    @property
    def is_full(self) -> bool:
        """Whether the buffer is full.

        :type: bool

        """
        return self._curptr >= self._capacity

    @property
    def size(self) -> int:
        """The current number of experiences stored in the buffer.

        :type: int

        """
        return self._capacity if self._full else self._curptr

    def state_dict(self) -> dict[str, Any]:
        """Return a dictionary containing the state of the buffer for saving.

        :return: A dictionary containing the state of the buffer.
        :rtype: dict[str, Any]

        """
        return {
            '_capacity': self._capacity,
            '_validate_keys': self._validate_keys,
            '_rng': self._rng.bit_generator.state,
            '_buffer': self._buffer,
            '_curptr': self._curptr,
            '_full': self._full,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state of the buffer from a dictionary.

        :param state_dict: A dictionary containing the state of the buffer, usually obtained from ``state_dict()``.
        :type state_dict: dict[str, Any]

        """
        self._capacity = state_dict['_capacity']
        self._validate_keys = state_dict['_validate_keys']
        self._rng.bit_generator.state = state_dict['_rng']
        self._buffer = state_dict['_buffer']
        self._curptr = state_dict['_curptr']
        self._full = state_dict['_full']

    def reset(self) -> None:
        """Reset/initialize the buffer."""
        self._buffer = {}
        self._curptr = 0
        self._full = False

    def add(self, experience: dict[str, np.ndarray]) -> None:
        """Add an experience to the buffer.

        :param experience: The experience to add, represented as a dictionary of numpy arrays.
        :type experience: dict[str, np.ndarray]

        """
        # Make sure keys match if buffer is not empty
        if self._validate_keys and self._buffer:
            for k in np.unique(list(experience.keys()) + list(self._buffer.keys())):
                if k not in self._buffer or k not in experience:
                    raise ValueError(f"Key '{k}' not common between experience and buffer.")

        # Add experience to buffer
        for k, v in experience.items():
            # Initialize whole buffer if key is not known
            if k not in self._buffer:
                # TODO: Add memory mapping
                self._buffer[k] = np.empty((self._capacity, *v.shape), dtype=v.dtype)

            # Add to existing buffer
            self._buffer[k][self._curptr] = v

        # Increment pointer and check if buffer is full
        self._curptr += 1
        if self._curptr >= self._capacity:
            self._full = True
            self._curptr = 0

    def sample(self, batch_size: int, sequence_length: int) -> dict[str, np.ndarray]:
        """Sample a batch of experiences from the buffer.

        :param batch_size: The number of experiences to sample.
        :type batch_size: int
        :param sequence_length: The length of each sampled experience.
        :type sequence_length: int
        :return: A batch of experiences, represented as a dictionary of numpy arrays.
        :rtype: dict[str, np.ndarray]

        """
        # Ensure batch size is not larger than buffer size, not needed with replacement
        # if batch_size > self.size:
        #     raise ValueError(f'Batch size {batch_size} larger than buffer size {self.size}.')

        # Sample start points
        idx = self._rng.choice(self.size - sequence_length + 1, size=batch_size, replace=True)  # TODO: Evaluate replace

        # Offset indices based on pointer if buffer is full
        if self._full:
            idx = (idx + self._curptr) % self._capacity

        # Add batch dimension as range
        idx = idx[:, None] + np.arange(sequence_length)[None, :]

        # Index into buffer and return batch
        return {k: np.take(v, idx, mode='wrap', axis=0) for k, v in self._buffer.items()}


class IndependentVectorizedBuffer(Buffer):
    """Class for group manipulation of independent buffers for vectorized environments."""
    def __init__(self, num_buffers: int, *buffer_args: list, buffer_class: Buffer = SequentialBuffer, seed: int = None, **buffer_kwargs: dict) -> None:
        """Initialize the buffer group.

        :param num_buffers: The number of buffers to create.
        :type num_buffers: int
        :param capacity: The maximum number of experiences each buffer can store.
        :type capacity: int
        :param validate_keys: Whether to validate that all experiences have the same keys across buffers.
        :type validate_keys: bool
        :param buffer_class: The class of the buffers to create.
        :type buffer_class: Buffer
        :param seed: The seed for the random number generator used for sampling.
        :type seed: int
        :param buffer_args: Positional arguments to pass to the buffer class.
        :type buffer_args: list
        :param buffer_kwargs: Keyword arguments to pass to the buffer class.
        :type buffer_kwargs: dict

        """
        super().__init__()

        # Initialize buffers
        self._buffers = [
            buffer_class(
                *buffer_args,
                seed=(seed + i) if seed is not None else None,
                **buffer_kwargs)
            for i in range(num_buffers)]

        # Initialize rng
        self._rng = np.random.default_rng(seed=seed)

    @property
    def size(self) -> int:
        """The current number of experiences stored in all buffers.

        :type: int

        """
        return sum([buffer.size for buffer in self._buffers])

    def state_dict(self) -> dict[str, Any]:
        """Return a dictionary containing the state of all buffers for saving.

        :return: A dictionary containing the state of all buffers.
        :rtype: dict[str, Any]

        """
        return {
            '_buffers': [buffer.state_dict() for buffer in self._buffers],
            '_rng': self._rng.bit_generator.state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state of all buffers from a dictionary.

        :param state_dict: A dictionary containing the state of all buffers, usually obtained from ``state_dict()``.
        :type state_dict: dict[str, Any]

        """
        # Load buffers state
        for buffer, buffer_state in zip(self._buffers, state_dict['_buffers']):
            buffer.load_state_dict(buffer_state)

        # Load rng state
        self._rng.bit_generator.state = state_dict['_rng']

    def reset(self) -> None:
        """Reset/initialize all buffers."""
        for buffer in self._buffers:
            buffer.reset()

    def add(self, experience: dict[str, np.ndarray]) -> None:
        """Add experience to all buffers. Note that buffers will desync if an error occurs while adding and experience to any buffer.

        :param experience: Vectorized experiences to add, represented as a dictionary of 2-dimensional numpy arrays.
        :type experience: dict[str, np.ndarray]

        """
        # Add each experience to buffers
        for i, buffer in enumerate(self._buffers):
            buffer.add({k: v[i] for k, v in experience.items()})

    def sample(self, batch_size: int, **sample_kwargs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Sample a batch of experiences from each buffer and concatenate them.

        :param batch_size: The number of experiences to sample, divided randomly among buffers.
        :type batch_size: int
        :param sample_kwargs: Additional keyword arguments to pass to each buffer's sample method.
        :type sample_kwargs: dict[str, Any]
        :return: A batch of experiences, represented as a dictionary of numpy arrays.
        :rtype: dict[str, np.ndarray]

        """
        # Ensure batch size is divisible by number of buffers
        if batch_size % len(self._buffers) != 0:
            warnings.warn(f'Batch size {batch_size} not divisible by number of buffers {len(self._buffers)}, rounding down.')

        # Compute per-buffer batch size
        per_buffer_batch_size = np.bincount(self._rng.integers(0, len(self._buffers), size=batch_size), minlength=len(self._buffers))

        # Sample from each buffer and concatenate
        batches = [buffer.sample(bs, **sample_kwargs) for buffer, bs in zip(self._buffers, per_buffer_batch_size) if bs > 0]
        return {k: np.concatenate([batch[k] for batch in batches], axis=0) for k in batches[0].keys()}


def convert_samples_to_tensors(samples: dict[str, np.ndarray], **tensor_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert sampled experiences from numpy arrays to tensors.

    :param samples: A batch of experiences, represented as a dictionary of numpy arrays.
    :type samples: dict[str, np.ndarray]
    :param tensor_kwargs: Keyword arguments to pass to the torch.tensor constructor, usually for specifying device.
    :type tensor_kwargs: dict[str, Any]
    :return: A batch of experiences, represented as a dictionary of tensors.
    :rtype: dict[str, torch.Tensor]

    """
    return {k: torch.from_numpy(v).to(**tensor_kwargs) for k, v in samples.items()}
