"""PyTorch ANN architecture for the inference stage.

This module defines the **embedding network** that maps a video tensor
`[batch, time, height, width]` to a latent embedding `[batch, embed_dim]`.
The embedding is then consumed by a downstream density estimator (a masked
autoregressive flow, MAF) that learns the posterior `p(theta | video)`.

Forward pipeline:

    video:  [B, T, H, W]
      |
      v   Complex3DCNN (3D convolutional encoder)
      |   - 3D convs capture local spatio-temporal patterns (a particle's
      |     PSF + its motion across a few neighboring frames).
      |   - Spatial dims are halved at each layer via MaxPool3d.
      |   - Temporal dim T is preserved through the conv stack.
      |
      v   Spatial averaging  ->  [B, C, T]
      |
      v   TemporalTransformer (attention over the temporal sequence)
      |   - Captures long-range temporal dependencies that 3D convs
      |     cannot reach with small kernels.
      |   - Returns a single summary vector (CLS-token embedding).
      |
      v   embedding:  [B, embed_dim]
      |
      v   [downstream: MAF density estimator]

Module contents:
    PositionalEncoding   -- sinusoidal positional encoding (Vaswani et al., 2017,
                            "Attention Is All You Need",
                            https://arxiv.org/abs/1706.03762).
    AttentionBlock       -- single transformer-style block: multi-head self-attention
                            + Mish-MLP, both residual with LayerNorm.
    TemporalTransformer  -- stack of AttentionBlocks with a learnable CLS token
                            and sinusoidal positional encoding (BERT-style CLS:
                            Devlin et al., 2018, https://arxiv.org/abs/1810.04805;
                            extended to vision in Dosovitskiy et al., 2020,
                            https://arxiv.org/abs/2010.11929).
    Complex3DCNN         -- end-to-end encoder: 3D CNN backbone + optional
                            TemporalTransformer.
"""

import math

import torch
from torch import nn


# =============================================================================
# Positional encoding
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from Vaswani et al. (2017),
    "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Adds a fixed (non-learnable) position-dependent vector to each token in an
    input sequence so that an otherwise permutation-invariant attention block
    can use order information. Sinusoidal because:
        - it generalizes to sequence lengths longer than seen at training time;
        - it provides a continuous notion of relative position via the
          sin/cos identities.

    The encoding is precomputed up to `max_len` positions and registered as a
    non-parametric buffer (saved with state_dict, but not optimized).

    Args:
        d_model: Embedding dimension. Each position gets a `d_model`-vector.
        max_len: Maximum sequence length supported. Inputs longer than this
            will raise an indexing error in `forward`.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape [1, max_len, d_model] for broadcast addition.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to an input sequence.

        Args:
            x: Tensor of shape `[batch, seq_len, d_model]`. `seq_len` must be
                <= `max_len` set at construction.

        Returns:
            Tensor of same shape as `x`, with the positional code added.
        """
        return x + self.pe[:, : x.size(1)]


# =============================================================================
# Attention block (self-attention + Mish-MLP, both residual)
# =============================================================================

class AttentionBlock(nn.Module):
    """Transformer-style block: multi-head self-attention + feed-forward MLP.

    The architecture is "post-norm" (LayerNorm is applied AFTER the residual
    addition), matching the original Vaswani et al. (2017) formulation,
    https://arxiv.org/abs/1706.03762. Each sub-layer is followed by a residual
    connection and LayerNorm:

        x = LayerNorm(x + MultiHeadAttention(x, x, x))
        x = LayerNorm(x + MLP(x))

    The MLP expansion factor is 4 (Linear from `embed_dim` to `4*embed_dim`
    and back), which is the standard Transformer choice.

    Activation is Mish (Misra, 2019, "Mish: A Self Regularized Non-Monotonic
    Activation Function", https://arxiv.org/abs/1908.08681), a smooth
    non-monotonic alternative to GELU/Swish; empirically slightly better
    on some image-recognition benchmarks.

    Args:
        embed_dim: Token embedding dimension. Must be divisible by `num_heads`
            (`nn.MultiheadAttention` constraint).
        num_heads: Number of attention heads. Each head sees `embed_dim/num_heads`
            channels.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention then MLP, each with a residual connection.

        Args:
            x: Tensor `[batch, seq_len, embed_dim]`.

        Returns:
            Tensor of same shape as input.
        """
        # Self-attention with residual + LayerNorm
        attn_out, _attn_weights = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        # MLP with residual + LayerNorm
        mlp_out = self.mlp(x)
        x = self.layer_norm2(x + mlp_out)
        return x


# =============================================================================
# Temporal transformer (CLS token + positional encoding + stacked AttentionBlocks)
# =============================================================================

class TemporalTransformer(nn.Module):
    """Transformer encoder that summarizes a temporal sequence into a single embedding.

    A learnable "CLS" (class) token is prepended to each input sequence; after
    several attention blocks, the CLS token's final embedding serves as a
    summary of the whole sequence. This is the same pattern used in BERT
    (Devlin et al., 2018, https://arxiv.org/abs/1810.04805) and extended to
    image patches in Vision Transformer / ViT (Dosovitskiy et al., 2020,
    https://arxiv.org/abs/2010.11929).

    Why a CLS token? Self-attention is permutation-invariant on the input
    set, so any "summary" pooling would lose order information. The CLS
    token, combined with positional encoding, gives the model a designated
    "output" position whose embedding can attend to all other positions and
    is supervised through downstream loss.

    Args:
        embed_dim: Token embedding dimension.
        n_layers: Number of stacked AttentionBlocks.
        num_heads: Number of attention heads per block.
    """

    def __init__(self, embed_dim: int, n_layers: int = 2, num_heads: int = 8):
        super().__init__()
        # Learnable token prepended at position 0 of every sequence.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Summarize a temporal sequence to a single embedding via the CLS token.

        Args:
            x: Tensor `[batch, seq_len, embed_dim]` (one embedding per time step).

        Returns:
            Tensor `[batch, embed_dim]` — the CLS-token output after `n_layers`
            attention blocks. This is the summary embedding of the whole sequence.
        """
        batch_size = x.shape[0]
        # Broadcast the learnable CLS token across the batch and prepend.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)         # [B, T+1, embed_dim]
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x[:, 0]                                # CLS-token output: [B, embed_dim]


# =============================================================================
# Complex 3D CNN encoder (+ Temporal transformer)
# =============================================================================

class Complex3DCNN(nn.Module):
    """End-to-end video-to-embedding network for simulation-based inference.

    Architecture:

        input video:     [B, T, H, W]   or   [B, 1, T, H, W]
              |
              |   1. 3D CNN backbone
              |      ----------------
              |      Stack of `n_conv_layers` blocks of:
              |        Conv3d(kernel=(3,3,3), padding=1)  -- preserves T, H, W
              |        BatchNorm3d
              |        Mish activation
              |        MaxPool3d(kernel=(1,2,2))           -- halves H, W; T preserved
              |      Channels double at each layer:
              |        in_channels -> start_channels -> ... -> start_channels * 2^(n-1)
              v
        features:        [B, C, T, H', W']
              |
              |   2. Spatial averaging
              |      -----------------
              |      Mean over (H', W') axes.
              v
        features:        [B, C, T]
              |
              |   3. Temporal summarization
              |      ----------------------
              |      EITHER the TemporalTransformer (if `use_temporal_attention=True`),
              |      which returns the CLS-token embedding of the sequence,
              |      OR a simple temporal mean over T.
              v
        embedding:       [B, C]

    The CLS-token embedding IS the output, suitable as a conditioning input for
    a downstream MAF density estimator that learns the full posterior
    p(theta | video).

    Why this architecture?
        - 3D convolutions capture LOCAL spatio-temporal features (a single
          particle's appearance and its motion across a few frames).
        - The temporal transformer captures LONG-RANGE temporal dependencies
          across many frames, which small 3D conv kernels cannot reach.
        - Spatial pooling reduces dimensionality before the attention stage,
          where each step is O(T^2) in memory and compute.

    Args:
        n_frames: Number of temporal frames in the input video. The network
            uses a dummy forward pass at construction to infer the output
            shape after the 3D conv stack, so `n_frames` is required (cannot
            be inferred at construction without an actual input). Compute as
            `total_time_seconds / frame_time_seconds` for the configured run.
        input_channels: Channels in the input video (1 for grayscale).
        n_conv_layers: Number of 3D conv blocks in the backbone.
        n_attn_layers: Number of AttentionBlocks in the TemporalTransformer
            (only used if `use_temporal_attention=True`).
        start_channels: Output channels of the first conv block; doubles each
            block. With `start_channels=8` and `n_conv_layers=4`, the final
            channel count is 128.
        use_temporal_attention: If True, summarize the temporal sequence via
            TemporalTransformer (CLS token output). If False, average over T.
        attention_heads: Number of attention heads in each AttentionBlock.
            Must divide `start_channels * 2^(n_conv_layers-1)`.
        verbose: If True, print the inferred output shape and feature dim at
            construction time.
    """

    def __init__(self,
                 n_frames: int,
                 input_channels: int = 1,
                 n_conv_layers: int = 4,
                 n_attn_layers: int = 1,
                 start_channels: int = 8,
                 use_temporal_attention: bool = True,
                 attention_heads: int = 2,
                 verbose: bool = False):
        super().__init__()
        self.use_temporal_attention = use_temporal_attention

        # ---- 3D CNN backbone ------------------------------------------------
        layers = []
        in_channels = input_channels
        out_channels = start_channels
        for _ in range(n_conv_layers):
            layers.extend([
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=(3, 3, 3),
                          stride=(1, 1, 1),
                          padding=(1, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.Mish(),
                nn.MaxPool3d(kernel_size=(1, 2, 2)),  # spatial /2 each block; T preserved
            ])
            in_channels = out_channels
            out_channels *= 2
        self.features = nn.Sequential(*layers)

        # ---- Infer output shape via dummy forward pass ----------------------
        # The post-conv channel count drives the embedding dimension of the
        # temporal transformer. Spatial dims are hardcoded to 256x256 to match
        # the simulation grid; if the simulation grid size changes, update here.
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, n_frames, 256, 256)
            dummy_output = self.features(dummy_input)
            pre_pool_shape = dummy_output.shape
            _, channels, _, _, _ = dummy_output.shape
            self.feature_dim = channels

        # ---- Optional temporal transformer ----------------------------------
        if use_temporal_attention:
            self.temporal_transformer = TemporalTransformer(
                embed_dim=self.feature_dim,
                n_layers=n_attn_layers,
                num_heads=attention_heads,
            )

        if verbose:
            attn_part = f", {n_attn_layers} attention layers" if use_temporal_attention else ""
            print(
                f"Complex3DCNN initialized: {n_conv_layers} CNN layers{attn_part}; "
                f"pre-pool shape {pre_pool_shape}; feature dim {self.feature_dim}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a video into a summary embedding.

        Args:
            x: Input tensor. Either:
                - `[B, T, H, W]` (4D; channel dim will be added automatically), or
                - `[B, 1, T, H, W]` (5D; channel dim already present).

        Returns:
            `[B, feature_dim]` -- the summary embedding (the CLS-token output
            when temporal attention is used, otherwise the temporal mean).
        """
        # Add channel dim if missing.
        if x.dim() == 4:
            x = x.unsqueeze(1)                # [B, 1, T, H, W]
        x = self.features(x)                  # [B, C, T, H', W']
        x = torch.mean(x, dim=(3, 4))         # spatial mean -> [B, C, T]
        if self.use_temporal_attention:
            x = x.transpose(1, 2)              # [B, T, C]
            x = self.temporal_transformer(x)   # CLS token -> [B, C]
        else:
            x = torch.mean(x, dim=2)           # temporal mean -> [B, C]
        return x
