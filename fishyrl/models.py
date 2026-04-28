"""Utility models and layers for construction of reinforcement learning agents."""

import copy
import math
from typing import Any

import einops
import torch
import torch.nn as nn

from . import actions as frl_actions
from . import distributions as frl_distributions
from . import utilities as frl_utilities


class MLP(nn.Module):
    """MLP for processing vector inputs."""
    def __init__(self, input_dim: int, output_dim: int | None = None, hidden_dims: list[int] = []) -> None:
        """Initialize the MLP.

        :param input_dim: The dimension of the input vector.
        :type input_dim: int
        :param output_dim: The dimension of the output vector. If provided,
            a final hidden layer will be added with no normalization or activation
        :type output_dim: int | None
        :param hidden_dims: The dimensions of the hidden layers.
        :type hidden_dims: list[int]

        """
        super().__init__()

        # Create model in blocks
        layers = []
        hidden_dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            layers += [
                # Layer, dropout, norm, activation
                nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False),
                nn.LayerNorm(hidden_dims[i + 1], eps=1e-3),
                nn.SiLU(),
            ]

        # Add final hidden layer
        if output_dim is not None:
            layers += [nn.Linear(hidden_dims[-1], output_dim)]

        # Assemble model
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the MLP.

        :param x: The input tensor of shape (batch_dim, input_dim).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_dim, hidden_dims[-1]).
        :rtype: torch.Tensor

        """
        return self._model(x)


class MLPEncoder(nn.Module):
    """MLP encoder for processing vector observations."""
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 3, hidden_dim: int = 512, use_symlog: bool = True) -> None:
        """Initialize the MLP encoder.

        :param input_dim: The dimension of the input vector observation.
        :type input_dim: int
        :param output_dim: The dimension of the output vector.
        :type output_dim: int
        :param num_blocks: The number of blocks in the MLP. (Default: ``3``)
        :type num_blocks: int
        :param hidden_dim: The dimension of the hidden layers. (Default: ``512``)
        :type hidden_dim: int
        :param use_symlog: Whether to apply symlog transformation to the input. (Default: ``True``)
        :type use_symlog: bool

        """
        super().__init__()

        # Parameters
        self._output_dim = output_dim
        self._use_symlog = use_symlog

        # Create model
        self._model = MLP(
            input_dim=input_dim,
            hidden_dims=num_layers * [hidden_dim] + [output_dim],
        )

    @property
    def output_dim(self) -> int:
        """Output dimension of the MLP decoder.

        :type: int

        """
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the MLP encoder.

        :param x: The input tensor of shape (batch_dim, input_dim).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_dim, hidden_dim).
        :rtype: torch.Tensor

        """
        # Cast to proper dtype
        x = x.to(dtype=next(self.parameters()).dtype)

        # Apply symlog transformation if specified
        x = frl_distributions.symlog(x) if self._use_symlog else x

        return self._model(x)

class MLPDecoder(nn.Module):
    """MLP decoder for reconstructing vector observations from latent representations."""
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 3, hidden_dim: int = 512) -> None:
        """Initialize the MLP decoder.

        :param input_dim: The dimension of the input tensor.
        :type input_dim: int
        :param output_dim: The dimension of the output vector observations.
        :type output_dim: int
        :param num_blocks: The number of blocks in the MLP. (Default: ``3``)
        :type num_blocks: int
        :param hidden_dim: The dimension of the hidden layers. (Default: ``512``)
        :type hidden_dim: int

        """
        super().__init__()

        # Create model
        self._model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=num_layers * [hidden_dim],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the MLP decoder.

        :param x: The input tensor of shape (batch_dim, hidden_dim).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_dim, output_dim).
        :rtype: torch.Tensor

        """
        return self._model(x)


class ChannelNorm(nn.LayerNorm):
    """Layer normalization across the channel dimension."""
    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        """Initialize the ChannelNorm layer.

        :param args: Positional arguments for the LayerNorm.
        :type args: list[Any]
        :param kwargs: Keyword arguments for the LayerNorm.
        :type kwargs: dict[str, Any]

        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization across the channel dimension.

        Swaps the channel dimension to the end, applies layer normalization, then swaps it back.

        :param x: The input tensor of shape (batch_dim, channels, height, width).
        :type x: torch.Tensor
        :return: The normalized tensor of the same shape.
        :rtype: torch.Tensor

        """
        # Permute channels to last dimension
        x = x.permute(0, 2, 3, 1)
        # Apply layer normalization
        x = super().forward(x)
        # Reverse permutation
        return x.permute(0, 3, 1, 2)


class CNNEncoder(nn.Module):
    """CNN encoder for processing image observations.

    Processes a (generally) 64x64 image to 4x4 using [2, 4, 8, 16] channels and interleaved
    layer normalization and SiLU activations.

    """
    def __init__(self, input_channels: int, image_dim: tuple[int, int] = (64, 64), num_blocks: int = 4) -> None:
        """Initialize the CNN encoder.

        :param input_channels: The number of input channels in the image.
        :type input_channels: int
        :param image_dim: The size of the input image. (Default: ``(64, 64)``)
        :type image_dim: tuple[int, int]
        :param num_blocks: The number of blocks in the CNN. (Default: ``4``)
        :type num_blocks: int

        """
        super().__init__()

        # Create model in blocks
        layers = []
        hidden_channels = [input_channels] + [2 * 2 ** i for i in range(num_blocks)]
        for i in range(num_blocks):
            layers += [
                # Layer, dropout, norm, activation
                nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
                ChannelNorm(hidden_channels[i + 1], eps=1e-3),
                nn.SiLU(),
            ]
        layers += [nn.Flatten()]
        self._output_dim = hidden_channels[-1] * math.prod(image_dim) // 4 ** num_blocks
        self._model = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        """Output dimension of the CNN encoder.

        :type: int

        """
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the CNN encoder.

        :param x: The input tensor of shape (batch_dim, channels, height, width).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_dim, output_dim, height, width).
        :rtype: torch.Tensor

        """
        # TODO: Add casting and conversion
        # NOTE: Difference here with SheepRL, where batch dims are flattened during forward
        return self._model(x)


class CNNDecoder(nn.Module):
    """CNN decoder for reconstructing image observations from latent representations.

    Uses transposed convolutions to upsample to original resolution, with interleaved layer normalization
    and SiLU activations.

    """
    def __init__(
        self,
        output_channels: int,
        input_dim: int,
        image_dim: tuple[int, int] = (64, 64),
        num_blocks: int = 4,
    ) -> None:
        """Initialize the CNN decoder.

        :param output_channels: The number of output channels in the reconstructed image.
        :type output_channels: int
        :param input_dim: The size of the latent representation.
        :type input_dim: int
        :param image_dim: The size of the output image. (Default: ``(64, 64)``)
        :type image_dim: tuple[int, int]
        :param num_blocks: The number of blocks in the CNN. (Default: ``4``)
        :type num_blocks: int

        """
        super().__init__()

        # Get encoded dimension and downsampled image dimensions
        encoder_dim = 2 ** num_blocks * math.prod(image_dim) // 4 ** num_blocks
        encoded_image_dim = (image_dim[0] // 2 ** num_blocks, image_dim[1] // 2 ** num_blocks)

        # Create model in blocks
        layers = [
            nn.Linear(input_dim, encoder_dim),
            nn.Unflatten(-1, (2 ** num_blocks, *encoded_image_dim)),  # Format into CxHxW
        ]
        hidden_channels = [2 ** (num_blocks - i) for i in range(num_blocks)] + [output_channels]
        for i in range(num_blocks):
            # Add blocks if not last
            if i != num_blocks - 1:
                layers += [
                    # Layer, dropout, norm, activation
                    nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
                    ChannelNorm(hidden_channels[i + 1], eps=1e-3),
                    nn.SiLU(),
                ]
            # Otherwise, add final block with bias and no norm/activation
            else:
                layers += [
                    nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=True),
                ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the CNN decoder.

        :param x: The input tensor of shape (batch_dim, latent_dim).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_dim, channels, height, width).
        :rtype: torch.Tensor

        """
        return self._model(x)


class PositionalEncoding(nn.Module):
    """Module for adding positional encoding to input tensors.

    From https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html.

    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """Initialize the positional encoding.

        :param d_model: The dimension of the model and positional encoding.
        :type d_model: int
        :param dropout: The dropout rate. (Default: ``0.1``)
        :type dropout: float
        :param max_len: The maximum length of the input sequences. (Default: ``5000``)
        :type max_len: int

        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor.

        :param x: The input tensor
        :type x: torch.Tensor
        :return: The output tensor with positional encoding added.
        :rtype: torch.Tensor

        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AttentionEncoder(nn.Module):
    """Attention encoder for processing sequential observations with attention mechanisms."""
    def __init__(self, input_dims: list[int], hidden_dim: int = 512, num_layers: int = 2, num_heads: int = 8, num_queries: int = 1) -> None:
        """Initialize the attention encoder.

        :param input_dims: A list of input dimensions for each sequential observation to be processed with attention.
        :type input_dims: list[int]
        :param hidden_dim: The dimension of the hidden layers and output per query. (Default: ``512``)
        :type hidden_dim: int
        :param num_layers: The number of layers in the attention encoder. (Default: ``2``)
        :type num_layers: int
        :param num_heads: The number of attention heads to use. If None, defaults to the number of layers. (Default: ``None``)
        :type num_heads: int | None
        :param num_queries: The number of learned queries to use in the cross attention layer. (Default: ``1``)
        :type num_queries: int

        """
        super().__init__()

        # Parameters
        self._hidden_dim = hidden_dim

        # Activation
        self._activation = nn.SiLU()

        # Initialize input heads
        self._input_heads = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for input_dim in input_dims])

        # Positional encoding
        self._positional_encoding = PositionalEncoding(hidden_dim)  # TODO: Include max_len parameter

        # Initialize self attention blocks
        self._self_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)])
        self._self_attention_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim, eps=1e-3)
            for _ in range(num_layers)])
        self._self_attention_fcl = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)])

        # Transformer
        # self._self_attention = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         hidden_dim,
        #         num_heads,
        #         dim_feedforward=hidden_dim,
        #         activation='gelu',
        #         batch_first=True),  # NOTE: We normally use SiLU everywhere else
        #     num_layers=num_layers)
        # TODO: Use encoder src_key_mask and reshape to (batch, padded_cars, feats)

        # Initialize cross attention layer
        self._queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self._cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    @property
    def output_dim(self) -> int:
        """Output dimension of the attention encoder.

        :type: int

        """
        return self._hidden_dim

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the attention encoder.

        :param x: A list of input tensors for each sequential observation, where each tensor has shape (batch_dims, seq_len, input_dim).
        :type x: list[torch.Tensor]
        :return: The output tensor of shape (batch_dims, num_queries * hidden_dim).
        :rtype: torch.Tensor

        """
        # Store batch dimensions and flatten
        batch_dims = x[0].shape[:-2]
        x = [xi.view(-1, *xi.shape[-2:]) for xi in x]

        # Process each input through its head
        x = [input_head(xi) for input_head, xi in zip(self._input_heads, x)]

        # Apply positional encoding to each head individually
        x = [self._positional_encoding(xi) for xi in x]

        # Concatenate inputs along sequence dimension
        x = torch.cat(x, dim=-2)
        # x = self._self_attention(x)

        # Apply self attention layers with pre-normalization and residual connections
        for self_attn, norm, fcl in zip(self._self_attention, self._self_attention_norm, self._self_attention_fcl):
            # Apply self attention on normalized input
            x_norm = norm(x)
            x_attn, _ = self_attn(x_norm, x_norm, x_norm, need_weights=False)
            x_attn = self._activation(x_attn)
            x = x + x_attn

            # Apply fcl with residual
            x_fcl = fcl(norm(x))
            x_fcl = self._activation(x_fcl)
            x = x + x_fcl

        # Apply cross attention with learned queries
        queries = self._queries.view(1, *self._queries.shape).expand(*x.shape[:-2], -1, -1)
        out, _ = self._cross_attention(queries, x, x, need_weights=False)

        return out.view(*batch_dims, -1)


class AttentionDecoder(nn.Module):
    """Attention decoder for reconstructing sequential observations from latent representations."""
    def __init__(self, input_dim: int, output_dims: list[int], num_queries: list[int] | int = 1, hidden_dim: int = 512, num_layers: int = 2, num_heads: int = 8) -> None:
        """Initialize the attention decoder.

        :param input_dim: The dimension of the input latent representation.
        :type input_dim: int
        :param output_dims: A list of output dimensions for each sequential observation to be reconstructed with attention.
        :type output_dims: list[int]
        :param num_queries: A list of the number of queries to use for each output sequential observation, which determines the maximum
            number of segments that can be reconstructed for each output. If a single integer is provided, the same number of queries
            will be used for each output. (Default: ``1``)
        :type num_queries: list[int] | int
        :param hidden_dim: The dimension of the hidden layers and output per query. (Default: ``512``)
        :type hidden_dim: int
        :param num_layers: The number of layers in the attention decoder. (Default: ``2``)
        :type num_layers: int
        :param num_heads: The number of attention heads to use. If None, defaults to the number of layers. (Default: ``None``)
        :type num_heads: int | None

        """
        super().__init__()

        # Parameters
        self._output_dims = output_dims

        # Default parameters
        if isinstance(num_queries, int):
            num_queries = [num_queries] * len(output_dims)
        self._num_queries = num_queries

        # Activation
        self._activation = nn.SiLU()

        # Initialize input head
        self._input_head = nn.Linear(input_dim, hidden_dim)

        # Initialize cross attention layer
        self._queries = nn.Parameter(torch.randn(sum(num_queries), hidden_dim))
        self._cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Initialize self attention blocks
        self._self_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)])
        self._self_attention_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim, eps=1e-3)
            for _ in range(num_layers)])
        self._self_attention_fcl = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)])

        # Initialize self attention blocks
        # self._self_attention = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         hidden_dim,
        #         num_heads,
        #         dim_feedforward=hidden_dim,
        #         activation='gelu',
        #         batch_first=True),
        #     num_layers=num_layers)

        # Initialize output heads and existence predictors
        self._output_heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for output_dim in output_dims])
        self._existence_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in output_dims])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Perform a forward pass through the attention decoder.

        :param x: The input tensor of shape (batch_dim, input_dim).
        :type x: torch.Tensor
        :return: A tuple containing a list of output tensors for each sequential observation in the original dimension
            and a list of existence logits for each sequential observation.
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor]]

        """
        # Store batch dimensions and flatten
        batch_dims = x.shape[:-1]
        x = x.view(-1, x.shape[-1])

        # Process input through input head and add sequence dimension for attention
        x = self._input_head(x).unsqueeze(-2)

        # Apply cross attention with learned queries
        queries = self._queries.view(1, *self._queries.shape).expand(*x.shape[:-2], -1, -1)
        x, _ = self._cross_attention(queries, x, x, need_weights=False)

        # Apply self attention
        # x = self._self_attention(x)
        for self_attn, norm, fcl in zip(self._self_attention, self._self_attention_norm, self._self_attention_fcl):
            # Apply self attention on normalized input
            x_norm = norm(x)
            x_attn, _ = self_attn(x_norm, x_norm, x_norm, need_weights=False)
            x_attn = self._activation(x_attn)
            x = x + x_attn

            # Apply fcl with residual
            x_fcl = fcl(norm(x))
            x_fcl = self._activation(x_fcl)
            x = x + x_fcl

        # Unflatten batch dimensions
        x = x.view(*batch_dims, *x.shape[-2:])

        # Get output and existence predictions for each sequential observation
        outputs, existence_logits = [], []
        start_idx = 0
        for output_head, existence_head, num_queries in zip(self._output_heads, self._existence_heads, self._num_queries):
            # Subset input to the number of queries for this output
            x_sub = x[..., start_idx : start_idx + num_queries, :]
            start_idx += num_queries

            # Record
            outputs.append(output_head(x_sub))
            existence_logits.append(existence_head(x_sub).squeeze(-1))

        return outputs, existence_logits


def extract_representation(x: torch.Tensor, specs: list[dict[str, Any]]) -> list[torch.Tensor | list[torch.Tensor]]:
    """Extract representations from the input tensor based on the provided specifications.

    :param x: The input tensor of shape (batch_dim, input_dim).
    :type x: torch.Tensor
    :param specs: A list of specifications for each representation to extract. More details can be found in the documentation
        for the ``CompoundEncoder`` class.
    :type specs: list[dict[str, Any]]
    :return: A list of extracted representations, where each representation is either a tensor or a list of tensors depending
        on the specification.
    :rtype: list[torch.Tensor | list[torch.Tensor]]

    """
    # Loop through specifications and extract representations based on type
    representation = []
    for spec in specs:
        # Get type and segments
        spec_type = spec.type
        spec_segments = spec.segments

        # Extract representation based on type
        if spec_type.upper() == 'MLP':
            # Extract and record representations in specified ranges
            extracted_segments = []
            for spec_segment in spec_segments:
                extracted_segment = x[..., spec_segment.range[0] : spec_segment.range[1]]
                extracted_segments.append(extracted_segment)
            # Concatenate extracted segments along the last dimension
            extracted_segments = torch.cat(extracted_segments, dim=-1)

        elif spec_type.upper() == 'CNN':
            # Extract, reshape to image dimensions
            input_len = math.prod(spec_segments.image_dim)
            offset = spec_segments.get('offset', 0)
            extracted_segments = x[..., offset : offset + input_len]
            extracted_segments = extracted_segments.view(*extracted_segments.shape[:-1], *spec_segments.image_dim)

        elif spec_type.upper() == 'ATTENTION':
            # Extract and reshape if needed
            extracted_segments = []
            for spec_segment in spec_segments:
                # Filter to input range and reshape to (seq_len, feature_dim)
                segment_len = spec_segment.segment_len if 'segment_len' in spec_segment else spec_segment.range[1] - spec_segment.range[0]
                extracted_segment = x[..., spec_segment.range[0] : spec_segment.range[1]]
                extracted_segment = extracted_segment.view(*extracted_segment.shape[:-1], -1, segment_len)
                extracted_segments.append(extracted_segment)

        # Record
        representation.append(extracted_segments)

    return representation


class CompoundEncoder(nn.Module):
    """Compound encoder for processing MLP, CNN, and attention observations."""
    def __init__(
        self,
        *encoder_specs: list[dict[str, Any]],
        # Encoder parameters
        output_dim: int = 512,  # Output dimension of each encoder
        num_blocks: int = 4,
        num_layers: int = 3,
        num_att_layers: int = 2,
        num_heads: int = 8,
        hidden_dim: int = 512,
    ) -> None:
        """Initialize the compound encoder.

        :param encoder_specs: A list of specifications for each encoder, where each specification is a dictionary containing the
            type and segment specifications for the encoder. These options include ``type`` and ``segments``, where ``type`` is one of
            ``MLP``, ``CNN``, or ``ATTENTION``, and ``segments`` is a list of segment specifications for the encoder. Specifically,
            for ``MLP``, ``segments`` should be a list of dictionaries containing the key ``range`` with a list of (start, end) indices
            for the input range. For ``CNN``, ``segments`` should be a list of dictionaries containing the key ``image_dim`` with a tuple
            of (channels, height, width) for the input image and an optional offset containing the index of the first value for the
            flattened image. For ``ATTENTION``, ``segments`` should be a list of dictionaries containing the key ``range`` with a list
            of (start, end) indices for the input range, and an optional key ``segment_len`` with the length of each segment to use as
            an input to the attention encoder.
        :type encoder_specs: list[dict[str, Any]]
        :param output_dim: The output dimension of each encoder. (Default: ``512``)
        :type output_dim: int
        :param num_blocks: The number of blocks in each CNN encoder. (Default: ``4``)
        :type num_blocks: int
        :param num_layers: The number of layers in each block of the MLP encoders. (Default: ``3``)
        :type num_layers: int
        :param num_att_layers: The number of layers in the attention encoders. (Default: ``2``)
        :type num_att_layers: int
        :param num_heads: The number of attention heads in each attention encoder. (Default: ``8``)
        :type num_heads: int
        :param hidden_dim: The hidden dimension of each encoder. (Default: ``512``)
        :type hidden_dim: int

        """
        super().__init__()

        # Parameters
        self._output_dim = output_dim
        self._encoder_specs = encoder_specs

        # Create encoders for each option
        self._encoders, self._encoder_specs = [], []
        for encoder_spec in encoder_specs:
            # Copy encoder spec
            encoder_spec = frl_utilities.DotDict(copy.deepcopy(encoder_spec))

            # Get type
            encoder_type = encoder_spec.type

            # Get specifications
            if 'segments' in encoder_spec:
                encoder_segments = encoder_spec.segments
            # Use defaults if not provided
            else:
                raise ValueError('Must specify input ranges for encoders.')
                # # MLP and attention defaults
                # if encoder_type.upper() in ('MLP', 'ATTENTION'):
                #     encoder_segments = [{'range': (0, None)}]
                # # CNN defaults
                # elif encoder_type.upper() == 'CNN':
                #     raise NotImplementedError('Must specify input dimensions for CNN encoder.')

            # Initialize encoder based on type
            # MLP encoder
            if encoder_type.upper() == 'MLP':
                # Infer input dimension from adding encoder ranges
                input_dim = sum(
                    encoder_segment.range[1] - encoder_segment.range[0]
                    for encoder_segment in encoder_segments)

                # Initialize
                encoder = MLPEncoder(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim)

            # CNN encoder
            elif encoder_type.upper() == 'CNN':
                # Infer input channels from image dim
                if len(encoder_segments.image_dim) != 3:
                    raise ValueError('CNN encoder requires image input with 3 dimensions (channels, height, width).')
                input_channels, image_dim = encoder_segments.image_dim[0], encoder_segments.image_dim[1:]

                # Initialize
                encoder = CNNEncoder(
                    input_channels=input_channels,
                    image_dim=image_dim,
                    num_blocks=num_blocks)

            # Attention encoder
            elif encoder_type.upper() == 'ATTENTION':
                # Infer input dimensions from encoder ranges
                input_dims = []
                for encoder_segment in encoder_segments:
                    # Get range, using each separate specification as another input MLP
                    encoder_range = encoder_segment.segment_len if 'segment_len' in encoder_segment else encoder_segment.range[1] - encoder_segment.range[0]
                    input_dims.append(encoder_range)

                # Initialize
                encoder = AttentionEncoder(
                    input_dims=input_dims,
                    # output_dim=output_dim,  # TODO: Maybe add a final linear transformation here
                    hidden_dim=hidden_dim,
                    num_layers=num_att_layers,
                    num_heads=num_heads)

            # Otherwise, raise error
            else:
                raise ValueError(f'Unsupported encoder type "{encoder_type}", must be one of "MLP", "CNN", or "Attention".')

            # Record
            self._encoders.append(encoder)
            self._encoder_specs.append(encoder_spec)

        # Wrap in a module list
        self._encoders = nn.ModuleList(self._encoders)

    @property
    def output_dim(self) -> int:
        """Output dimension of the compound encoder.

        :type: int

        """
        return sum(encoder.output_dim for encoder in self._encoders)

    def forward(self, representation: list[torch.Tensor | list[torch.Tensor]]) -> torch.Tensor:
        """Perform a forward pass through the compound encoder, combining the outputs of each individual encoder.

        :param representation: A list of input representations for each encoder, normally produced using
            the ``extract_representation`` function.
        :type representation: list[torch.Tensor | list[torch.Tensor]]
        :return: The output tensor of shape (batch_dim, output_dim).
        :rtype: torch.Tensor

        """
        # Process each encoder and return concatenated outputs
        out = [encoder(x) for encoder, x in zip(self._encoders, representation)]
        return torch.cat(out, dim=-1)


class CompoundDecoder(nn.Module):
    """Compound decoder for reconstructing MLP, CNN, and attention observations from latent representations."""
    def __init__(
        self,
        *decoder_specs: list[dict[str, Any]],
        # Decoder parameters
        input_dim: int = 512,  # Input dimension of each decoder
        num_blocks: int = 4,
        num_layers: int = 3,
        num_att_layers: int = 2,
        num_heads: int = 8,
        hidden_dim: int = 512,
    ) -> None:
        """Initialize the compound decoder.

        :param decoder_specs: A list of specifications for each decoder, where each specification is a dictionary containing the
            type and output specifications for the decoder. These options include ``type`` and ``outputs``, where ``type`` is one of
            ``MLP``, ``CNN``, or ``ATTENTION``, and ``outputs`` is a list of output specifications for the decoder. Specifically,
            for ``MLP``, ``outputs`` should be a list of dictionaries containing the key ``range`` with a list of (start, end) indices
            for the output range. For ``CNN``, ``outputs`` should be a list of dictionaries containing the key ``image_dim`` with a tuple
            of (channels, height, width) for the output image and an optional offset containing the index of the first value for the
            flattened image. For ``ATTENTION``, ``outputs`` should be a list of dictionaries containing the key ``range`` with a list
            of (start, end) indices for the output range, and an optional key ``segment_len`` with the length of each segment to use as
            an output from the attention decoder.
        :type decoder_specs: list[dict[str, Any]]
        :param input_dim: The input dimension of each decoder. (Default: ``512``)
        :type input_dim: int
        :param num_blocks: The number of blocks in each CNN decoder. (Default: ``4``)
        :type num_blocks: int
        :param num_layers: The number of layers in each block of the MLP decoders. (Default: ``3``)
        :type num_layers: int
        :param num_att_layers: The number of layers in the attention decoders. (Default: ``2``)
        :type num_att_layers: int
        :param num_heads: The number of attention heads in each attention decoder. (Default: ``8``)
        :type num_heads: int
        :param hidden_dim: The hidden dimension of each decoder. (Default: ``512``)
        :type hidden_dim: int

        """
        super().__init__()

        # Parameters
        self._input_dim = input_dim
        self._decoder_specs = decoder_specs

        # Create decoders for each option
        self._decoders, self._decoder_specs = [], []
        for decoder_spec in decoder_specs:
            # Copy decoder spec
            decoder_spec = frl_utilities.DotDict(copy.deepcopy(decoder_spec))

            # Get type
            decoder_type = decoder_spec.type

            # Get specifications
            if 'segments' in decoder_spec:
                decoder_segments = decoder_spec.segments
            # Use defaults if not provided
            else:
                raise ValueError('Must specify output ranges for decoders.')

            # Initialize decoder based on type
            # MLP decoder
            if decoder_type.upper() == 'MLP':
                # Infer output dimension from adding decoder ranges
                output_dim = sum(
                    decoder_segment.range[1] - decoder_segment.range[0]
                    for decoder_segment in decoder_segments)

                # Initialize
                decoder = MLPDecoder(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim)

            # CNN decoder
            elif decoder_type.upper() == 'CNN':
                # Infer output channels from image dim
                if len(decoder_segments.image_dim) != 3:
                    raise ValueError('CNN decoder requires image output with 3 dimensions (channels, height, width).')
                output_channels, image_dim = decoder_segments.image_dim[0], decoder_segments.image_dim[1:]

                # Initialize
                decoder = CNNDecoder(
                    output_channels=output_channels,
                    input_dim=input_dim,
                    image_dim=image_dim,
                    num_blocks=num_blocks)

            # Attention decoder
            elif decoder_type.upper() == 'ATTENTION':
                # Infer output dimensions from decoder ranges
                output_dims, num_queries = [], []
                for decoder_segment in decoder_segments:
                    # Get range, using each separate specification as another output MLP
                    # Use segment length if provided, otherwise infer from range
                    segment_len = decoder_segment.segment_len if 'segment_len' in decoder_segment else decoder_segment.range[1] - decoder_segment.range[0]
                    max_segments = decoder_segment.get('max_segments', 1)
                    output_dims.append(segment_len)
                    num_queries.append(max_segments)

                # Initialize
                decoder = AttentionDecoder(
                    input_dim=input_dim,
                    output_dims=output_dims,
                    num_queries=num_queries,
                    num_layers=num_att_layers,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim)

            # Otherwise, raise error
            else:
                raise ValueError(f'Unsupported decoder type "{decoder_type}", must be one of "MLP", "CNN", or "Attention".')

            # Record
            self._decoders.append(decoder)
            self._decoder_specs.append(decoder_spec)

        # Wrap in a module list
        self._decoders = nn.ModuleList(self._decoders)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor | tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Perform a forward pass through the compound decoder, combining the outputs of each individual decoder.

        :param x: The input tensor of shape (batch_dim, input_dim).
        :type x: torch.Tensor
        :param reconstruct: Whether to reconstruct the output in its original format.
        :type reconstruct: bool
        :return: A list of all reconstructed outputs, where MLP and CNN decoders return tensors, while attention decoders return a
            tuple containing a list of output tensors and a list of existence logits for each sequential observation.
        :rtype: list[torch.Tensor | tuple[list[torch.Tensor], list[torch.Tensor]]]

        """
        # Process each decoder and concatenate outputs
        return [decoder(x) for decoder in self._decoders]


class LayerNormGRU(nn.Module):
    """GRU layer with internal layer normalization."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initialize the LayerNormGRU.

        :param input_dim: The size of the input tensor.
        :type input_dim: int
        :param hidden_dim: The size of the hidden state.
        :type hidden_dim: int

        """
        super().__init__()

        # Record parameters
        self._hidden_dim = hidden_dim

        # Initialize layers
        self._mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 3 * hidden_dim),
            nn.LayerNorm(3 * hidden_dim, eps=1e-3),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """Perform a forward pass through the LayerNormGRU.

        Applies layer normalization to the hidden state after the GRU computation.

        :param x: The input tensor of shape (batch_dim, input_dim).
        :type x: torch.Tensor
        :param h: The initial hidden state of shape (batch_dim, hidden_dim), or None to use zeros. (Default: ``None``)
        :type h: torch.Tensor | None
        :return: The final hidden state of shape (batch_dim, hidden_dim).
        :rtype: torch.Tensor

        """
        # If h is None, initialize to zeros
        if h is None:
            h = torch.zeros(*x.shape[:-1], self._hidden_dim, device=x.device, dtype=x.dtype)

        # Process input and hidden state through MLP
        x = torch.cat((x, h), dim=-1)
        x = self._mlp(x)

        # Get reset, update, and candidate gates
        reset, update, candidate = x.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update - 1)  # Bias update gate towards copying
        candidate = torch.tanh(reset * candidate)
        # NOTE: This deviates from the standard GRU formulation, but is a common variant that works well in practice

        # Update hidden state
        return (1 - update) * h + update * candidate


class SingleRecurrentModel(nn.Module):
    """Recurrent model for processing sequences of latent representations.

    Uses a GRU to process sequences of latent representations, with layer normalization and SiLU
    activations on the output.

    """
    def __init__(self, input_dim: int, hidden_dim: int, deter_dim: int) -> None:
        """Initialize the recurrent model.

        :param input_dim: The size of the input tensor.
        :type input_dim: int
        :param hidden_dim: The size of the hidden dimensions.
        :type hidden_dim: int
        :param deter_dim: The size of the deterministic state output by the recurrent model.
        :type deter_dim: int

        """
        super().__init__()

        # Store variables
        self._deter_dim = deter_dim

        # Create MLP and GRU layers
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU(),
        )
        self._gru = LayerNormGRU(hidden_dim, deter_dim)

    @property
    def deter_dim(self) -> int:
        """Deterministic size of the recurrent model.

        :type: int

        """
        return self._deter_dim

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """Perform a forward pass through the recurrent model.

        :param x: The input tensor of shape (batch_dim, seq_len, latent_dim).
        :type x: torch.Tensor
        :param h: The initial deterministic state of shape (1, batch_dim, deter_dim), or None to use zeros. (Default: ``None``)
        :type h: torch.Tensor | None
        :return: The final deterministic state of shape (batch_dim, deter_dim).
        :rtype: torch.Tensor

        """
        # Pass through the MLP and GRU
        x = self._mlp(x)
        h = self._gru(x, h)
        return h


class BlockLinear(nn.Module):
    """Linear layer that splits the input into blocks and processes each block separately."""
    def __init__(self, input_dim: int, output_dim: int, num_blocks: int = 8, bias: bool = True) -> None:
        """Initialize the BlockLinear layer.

        :param input_dim: The size of the input tensor.
        :type input_dim: int
        :param output_dim: The size of the output tensor.
        :type output_dim: int
        :param num_blocks: The number of blocks for splitting the input and output. (Default: ``8``)
        :type num_blocks: int
        :param bias: Whether to include a bias term. (Default: ``True``)
        :type bias: bool

        """
        super().__init__()

        # Check feasibility
        if input_dim % num_blocks != 0:
            raise ValueError(f'Input dimension ({input_dim}) must be divisible by number of blocks ({num_blocks}).')
        if output_dim % num_blocks != 0:
            raise ValueError(f'Output dimension ({output_dim}) must be divisible by number of blocks ({num_blocks}).')

        # Store parameters
        self._num_blocks = num_blocks
        self._use_bias = bias

        # Compute block sizes
        self._block_input_dim = input_dim // num_blocks
        self._block_output_dim = output_dim // num_blocks

        # Initialize block linear weights
        self._weight = nn.Parameter(
            torch.randn(num_blocks, self._block_input_dim, self._block_output_dim) / math.sqrt(self._block_input_dim))
        self._bias = nn.Parameter(torch.zeros(num_blocks, self._block_output_dim)) if bias else None  # NOTE: Is not present if `bias` is False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the BlockLinear layer.

        :param x: The input tensor of shape (batch_dim, input_dim).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_dim, output_dim).
        :rtype: torch.Tensor

        """
        # Split into blocks, process with weight tensor, and flatten
        x = einops.rearrange(x, 'b (g i) -> b g i', g=self._num_blocks)
        x = torch.einsum('bgi,gio->bgo', x, self._weight)
        if self._use_bias:
            x = x + self._bias
        return einops.rearrange(x, 'b g o -> b (g o)')


class BlockRecurrentModel(nn.Module):
    """Block recurrent model for processing sequences of latent representations.

    Uses a block GRU to process sequences of latent representations, with layer normalization and SiLU
    activations on the output.

    """
    def __init__(self, stoch_dim: int, act_dim: int, hidden_dim: int, deter_dim: int, num_blocks: int = 8) -> None:
        """Initialize the recurrent model.

        :param stoch_dim: The size of the stochastic state.
        :type stoch_dim: int
        :param act_dim: The size of the action tensor.
        :type act_dim: int
        :param hidden_dim: The size of the hidden dimension.
        :type hidden_dim: int
        :param deter_dim: The size of the deterministic state.
        :type deter_dim: int
        :param num_blocks: The number of blocks to split the deterministic state. Must divide ``deter_dim``. (Default: ``8``)
        :type num_blocks: int

        """
        super().__init__()

        # Check parameter feasibility
        if deter_dim % num_blocks != 0:
            raise ValueError(f'Number of blocks ({num_blocks}) must divide deterministic dimension ({deter_dim}).')

        # Store parameters
        self._deter_dim = deter_dim
        self._num_blocks = num_blocks

        # Initialize projection layers
        # TODO: Anything special with bias here?
        self._stoch_mlp = nn.Sequential(
            nn.Linear(stoch_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU())
        self._act_mlp = nn.Sequential(
            nn.Linear(act_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU())
        self._deter_mlp = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU())

        # Initialize block linear layers
        block_hidden_input = 3 * hidden_dim + deter_dim // num_blocks
        self._block_linear_hidden = nn.Sequential(
            BlockLinear(num_blocks * block_hidden_input, deter_dim, num_blocks=num_blocks, bias=False),
            nn.LayerNorm(deter_dim, eps=1e-3),  # Norm on flat representation, as in original Jax implementation
            nn.SiLU())
        self._block_linear_gates = BlockLinear(deter_dim, 3 * deter_dim, num_blocks=num_blocks)

    @property
    def deter_dim(self) -> int:
        """Deterministic size of the recurrent model.

        :type: int

        """
        return self._deter_dim

    def forward(self, s: torch.Tensor, a: torch.Tensor, d: torch.Tensor | None = None) -> torch.Tensor:
        """Perform a forward pass through the recurrent model.

        :param s: The stochastic tensor of shape (batch_dim, seq_len, latent_dim).
        :type s: torch.Tensor
        :param a: The action tensor of shape (batch_dim, seq_len, action_dim).
        :type a: torch.Tensor
        :param d: The initial deterministic state of shape (1, batch_dim, deter_dim), or None to use zeros. (Default: ``None``)
        :type d: torch.Tensor | None
        :return: The final deterministic state of shape (batch_dim, deter_dim).
        :rtype: torch.Tensor

        """
        # If d is None, initialize to zeros
        if d is None:
            d = torch.zeros(*s.shape[:-1], self._deter_dim, device=s.device, dtype=s.dtype)

        # Flatten batch dimensions
        batch_dims = s.shape[:-1]
        s, a, d = s.view(-1, s.shape[-1]), a.view(-1, a.shape[-1]), d.view(-1, d.shape[-1])

        # Clip action
        # NOTE: This is in the original, but I don't think it is good for continuous actions and does nothing for discrete
        # a = a / torch.clamp(a.abs(), min=1).detach()

        # Project to hidden dimensions
        # TODO: Could technically be parallelized using an uneven block weight matrix
        s_h = self._stoch_mlp(s)  # Includes layer norm and activation
        a_h = self._act_mlp(a)
        d_h = self._deter_mlp(d)

        # Concatenate and split by blocks
        x = torch.cat((s_h, a_h, d_h), dim=-1)
        x = x.unsqueeze(-2).expand(*x.shape[:-1], self._num_blocks, x.shape[-1])

        # Split deterministic state into blocks and concatenate
        d_g = einops.rearrange(d, 'b (g h) -> b g h', g=self._num_blocks)
        x = torch.cat((d_g, x), dim=-1)

        # Feed through block linear layer
        x = einops.rearrange(x, 'b g h -> b (g h)')
        x = self._block_linear_hidden(x)  # Includes layer norm and activation

        # Compute gates
        x = self._block_linear_gates(x)  # No layer norm or activation
        x = einops.rearrange(x, 'b (g h) -> b g h', g=self._num_blocks)  # Technically 3h
        reset, update, candidate = x.chunk(3, dim=-1)

        # Resize and process gates
        # TODO: Maybe implement `flat2group` and `group2flat` functions like in original
        reset = einops.rearrange(reset, 'b g h -> b (g h)')
        reset = torch.sigmoid(reset)
        update = einops.rearrange(update, 'b g h -> b (g h)')
        update = torch.sigmoid(update - 1)
        candidate = einops.rearrange(candidate, 'b g h -> b (g h)')
        candidate = torch.tanh(reset * candidate)
        # NOTE: This deviates from the standard GRU formulation, but is a common variant that works well in practice

        # Update deterministic state
        d = update * candidate + (1 - update) * d

        # Unflatten batch dimensions
        return d.view(*batch_dims, d.shape[-1])


class RSSM(nn.Module):
    """Recurrent state-space model for modeling environment dynamics.

    The RSSM consists of a recurrent model for tracking the hidden state, a representation model for inferring the stochastic
    state from the hidden state and observation, and a transition model for predicting the stochastic state from the
    hidden state.

    """
    def __init__(
        self,
        recurrent_model: nn.Module,
        representation_model: nn.Module,
        transition_model: nn.Module,
        bins: int = 32,
        learnable_initial_state: bool = False,
    ) -> None:
        """Initialize the RSSM.

        :param recurrent_model: The recurrent model for keeping track of the environment dynamics.
        :type recurrent_model: nn.Module
        :param representation_model: The model for inferring the stochastic state from the hidden state and observation.
        :type representation_model: nn.Module
        :param transition_model: The model for predicting the stochastic state from the hidden state.
        :type transition_model: nn.Module
        :param bins: The number of bins for the stochastic state. (Default: ``32``)
        :type bins: int

        """
        super().__init__()

        # Record variables
        self._recurrent_model = recurrent_model
        self._representation_model = representation_model
        self._transition_model = transition_model
        self._bins = bins

        # Initialize recurrent model hidden state
        # NOTE: Making this initial state learnable allows the model to learn an initial state
        #       rather than always starting from zeros
        if learnable_initial_state:
            self._initial_hidden_state = nn.Parameter(torch.zeros(recurrent_model.deter_dim))
        else:
            self.register_buffer('_initial_hidden_state', torch.zeros(recurrent_model.deter_dim, dtype=torch.get_default_dtype()))

    @property
    def initial_hidden_state(self) -> torch.Tensor:
        """Trainable initial hidden state of the recurrent model.

        :type: torch.Tensor

        """
        return self._initial_hidden_state

    def infer_stochastic(self, hidden_state: torch.Tensor, embedded_obs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer the stochastic state.

        Infer the stochastic state from the hidden state (using the transition model) or from the hidden state and the
        embedded observation (using the representation model).

        :param hidden_state: The hidden state from the recurrent model, of shape (batch_dim, hidden_dim).
        :type hidden_state: torch.Tensor
        :param embedded_obs: The embedded observation, of shape (batch_dim, obs_dim). (Default: ``None``)
        :type embedded_obs: torch.Tensor | None
        :return: The logits and sampled stochastic state.
        :rtype: tuple[torch.Tensor, torch.Tensor]

        """
        # Get the prior/posterior distribution
        # Use the transition model if not using an observation
        if embedded_obs is None:
            logits = self._transition_model(hidden_state)
        # Otherwise, use the representation model
        else:
            logits = self._representation_model(torch.cat((hidden_state, embedded_obs), dim=-1))

        # Reshape logits into bins
        logits = logits.view(*logits.shape[:-1], -1, self._bins)

        # Mix with uniform distribution
        logits, probs = frl_distributions.uniform_mix(logits)

        # Sample
        # NOTE: SheepRL uses `torch.distributions.Independent`, but this is unneeded since we never
        #       compute log probabilities
        dist = torch.distributions.OneHotCategoricalStraightThrough(probs=probs)
        sample = dist.rsample()

        # Don't sample, and instead take the mode - Might be useful for inference
        # maxprobs = probs.argmax(dim=-1)
        # mode = torch.nn.functional.one_hot(maxprobs, num_classes=probs.size(-1))

        # Flatten both logits and sample
        # NOTE: This assumes no continuity between bins. It may be more faithful to theory if these were
        #       made into continuous values after categorical sampling. However, this is what is done in
        #       the actual implementation.
        #       https://github.com/danijar/dreamerv3/blob/b65cf81a6fb13625af8722127459283f899a35d9/dreamerv3/rssm.py#L136
        #       https://github.com/danijar/embodied/blob/f460cf117bb6ca30e16d22876601845a3fc41082/embodied/jax/outs.py#L243
        #       https://github.com/danijar/embodied/blob/f460cf117bb6ca30e16d22876601845a3fc41082/embodied/jax/outs.py#L40
        logits = logits.view(*logits.shape[:-2], -1)
        sample = sample.view(*sample.shape[:-2], -1)

        return logits, sample

    def forward(
        self,
        action: torch.Tensor | None = None,
        posterior: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        embedded_obs: torch.Tensor | None = None,
        initialize: torch.Tensor | None = None,
        batch_dim: int | None = None,
        compute_prior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Perform one step of the RSSM.

        Will compute the hidden state using action, posterior, and previous hidden state. If not available,
        will instead use the initial hidden state. If ``embedded_obs`` is not provided, imagines the next
        hidden state.

        :param action: The action taken, of shape (batch_dim, action_dim).
        :type action: torch.Tensor | None
        :param posterior: The posterior stochastic state from the previous step, of shape (batch_dim, latent_dim).
        :type posterior: torch.Tensor | None
        :param hidden_state: The initial hidden state of the recurrent model, of shape (batch_dim, hidden_dim).
        :type hidden_state: torch.Tensor | None
        :param embedded_obs: The embedded observation, of shape (batch_dim, obs_dim), or None if imagining.
        :type embedded_obs: torch.Tensor | None
        :param initialize: Boolean tensor of hidden states to initialize of shape (batch_dim). (Default: ``None``)
        :type initialize: torch.Tensor | None
        :param batch_dim: The batch dimension, inferred if not provided. (Default: ``None``)
        :type batch_dim: int | None
        :param compute_prior: Whether to compute the prior distribution. (Default: ``True``)
        :type compute_prior: bool
        :return: A dictionary containing the prior and posterior logits and samples, and the updated hidden state.
        :rtype: dict[str, torch.Tensor]

        """
        # Get initial hidden state if not provided
        # NOTE: SheepRL performs one step of the recurrent model with zero action to get to the initial hidden state.
        #       We do not perform this, as it appears to be unnecessary and might be harmful to the action formulation.
        # if hidden_state is None:
        #     hidden_state = torch.tanh(self._initial_hidden_state).expand(action.shape[:-1], -1)

        # If not providing a posterior, infer the prior from the hidden state
        # if posterior is None:
        #     posterior = self.infer_stochastic(hidden_state)[1]
        #     posterior = self.infer_stochastic(hidden_state, embedded_obs)[1]

        # Infer batch dim
        if batch_dim is None:
            if action is not None:
                batch_dim = action.shape[:-1]
            elif posterior is not None:
                batch_dim = posterior.shape[:-1]
            elif hidden_state is not None:
                batch_dim = hidden_state.shape[:-1]
            elif embedded_obs is not None:
                batch_dim = embedded_obs.shape[:-1]
            else:
                raise ValueError(
                    'At least one of `action`, `posterior`, `hidden_state`, or `embedded_obs`'
                    ' must be provided if `batch_dim` is not defined.')
        # Format if `batch_dim` is an int
        elif isinstance(batch_dim, int):
            batch_dim = (batch_dim,)

        # Initialize returns
        return_dict = {}

        # Update the hidden state using the recurrent model
        # NOTE: Redundant calculations when overwritten by initialize, but might be more
        #       efficient to avoid indexing
        if posterior is not None and action is not None and hidden_state is not None:
            hidden_state = self._recurrent_model(posterior, action, hidden_state)  # NOTE: SheepRL concatenates action and posterior to a single-cell GRU, but this is not faithful to the block GRN
            if initialize is not None:
                initialize = initialize.unsqueeze(-1) if hidden_state.ndim > 1 else initialize
                hidden_state = (
                    ~initialize * hidden_state
                    + initialize * torch.tanh(self._initial_hidden_state).expand(*batch_dim, -1)
                )
        # Initialize if anything is missing
        else:
            hidden_state = torch.tanh(self._initial_hidden_state).expand(*batch_dim, -1)
        return_dict['hidden_state'] = hidden_state

        # Get the prior distribution from the transition model
        if compute_prior:
            prior_logits, prior = self.infer_stochastic(hidden_state)
            return_dict['prior_logits'] = prior_logits
            return_dict['prior'] = prior

        # Get the posterior distribution from the representation model if not imagining
        if embedded_obs is not None:
            posterior_logits, posterior = self.infer_stochastic(hidden_state, embedded_obs)
            return_dict['posterior_logits'] = posterior_logits
            return_dict['posterior'] = posterior

        return return_dict


class Actor(nn.Module):
    """Actor network.

    Pulls action heads from a latent representation and samples.

    """
    def __init__(
        self,
        input_dim: int,
        actions: list[frl_actions.Action],
        num_layers: int = 3,
        hidden_dim: int = 512,
    ) -> None:
        """Initialize the actor network.

        :param input_dim: The dimension of the input vector.
        :type input_dim: int
        :param actions: A list of action definitions, can be continuous or discrete.
        :type actions: list[fishyrl.actions.Action]
        :param num_layers: The number of blocks in the MLP. (Default: ``3``)
        :type num_layers: int
        :param hidden_dim: The dimension of the hidden layers. (Default: ``512``)
        :type hidden_dim: int

        """
        super().__init__()

        # Parameters
        self._actions = actions

        # Extract input and output sizes
        self._input_dims = [action.input_dim for action in actions]
        self._output_dims = [action.output_dim for action in actions]

        # Create base model
        self._model = MLP(
            input_dim,
            output_dim=sum(self._input_dims),
            hidden_dims=num_layers * [hidden_dim],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the actor network.

        :param x: The input tensor of shape (batch_dim, input_dim).
        :type x: torch.Tensor
        :return: A tuple containing the sampled actions and their distributions.
        :rtype: tuple[torch.Tensor, torch.Tensor]

        """
        # Compute logits
        x = self._model(x)

        # Sample each action
        actions, distributions = [], []
        for action, logits in zip(self._actions, x.split(self._input_dims, dim=-1)):
            # Sample each action
            head_actions, head_dist = action.sample(logits)
            actions.append(head_actions)
            distributions.append(head_dist)

        return torch.cat(actions, dim=-1), distributions
