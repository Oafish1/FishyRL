"""Buffer classes for storing and managing experiences."""

import warnings
from abc import abstractmethod

import numpy as np


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
    def __init__(self, capacity: int, validate_keys: bool = True) -> None:
        """Initialize the buffer.

        :param capacity: The maximum number of experiences to store.
        :type capacity: int
        :param validate_keys: Whether to validate that all experiences have the same keys.
        :type validate_keys: bool

        """
        # Parameters
        self._capacity = capacity
        self._validate_keys = validate_keys

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

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of experiences from the buffer.

        :param batch_size: The number of experiences to sample.
        :type batch_size: int
        :return: A batch of experiences, represented as a dictionary of numpy arrays.
        :rtype: dict[str, np.ndarray]

        """
        # Ensure batch size is not larger than buffer size
        if batch_size > self.size:
            raise ValueError(f'Batch size {batch_size} larger than buffer size {self.size}.')

        # Sample experiences
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        return {k: v[idx] for k, v in self._buffer.items()}


class IndependentVectorizedBufferGroup(Buffer):
    """Class for grouping independent buffers for vectorized environments."""
    def __init__(self, num_buffers: int, *buffer_args: list, buffer_class: Buffer = SequentialBuffer, **buffer_kwargs: dict) -> None:
        """Initialize the buffer group.

        :param num_buffers: The number of buffers to create.
        :type num_buffers: int
        :param capacity: The maximum number of experiences each buffer can store.
        :type capacity: int
        :param validate_keys: Whether to validate that all experiences have the same keys across buffers.
        :type validate_keys: bool
        :param buffer_class: The class of the buffers to create.
        :type buffer_class: Buffer
        :param buffer_args: Positional arguments to pass to the buffer class.
        :type buffer_args: list
        :param buffer_kwargs: Keyword arguments to pass to the buffer class.
        :type buffer_kwargs: dict

        """
        # Initialize buffers
        self._buffers = [buffer_class(*buffer_args, **buffer_kwargs) for _ in range(num_buffers)]

    @property
    def size(self) -> int:
        """The current number of experiences stored in all buffers.

        :type: int

        """
        return sum([buffer.size for buffer in self._buffers])

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

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of experiences from each buffer and concatenate them.

        :param batch_size: The number of experiences to sample, divided equally among buffers.
        :type batch_size: int
        :return: A batch of experiences, represented as a dictionary of numpy arrays.
        :rtype: dict[str, np.ndarray]

        """
        # Ensure batch size is divisible by number of buffers
        if batch_size % len(self._buffers) != 0:
            warnings.warn(f'Batch size {batch_size} not divisible by number of buffers {len(self._buffers)}, rounding down.')

        # Compute per-buffer batch size
        per_buffer_batch_size = batch_size // len(self._buffers)

        # Sample from each buffer and concatenate
        batches = [buffer.sample(per_buffer_batch_size) for buffer in self._buffers]
        return {k: np.concatenate([batch[k] for batch in batches], axis=0) for k in batches[0].keys()}
