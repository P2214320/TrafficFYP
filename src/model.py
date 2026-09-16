"""A compact PyTorch Transformer encoder for multi-station flow forecasting."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding for ``[batch, time, feature]`` tensors."""

    def __init__(self, max_length: int, d_model: int) -> None:
        super().__init__()
        if max_length <= 0 or d_model <= 0:
            raise ValueError("max_length and d_model must be positive")

        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        even_dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
        angle_rates = torch.exp(-math.log(10_000.0) * even_dimensions / d_model)

        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * angle_rates)
        encoding[:, 1::2] = torch.cos(positions * angle_rates[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, time, feature]")
        if inputs.size(1) > self.encoding.size(1):
            raise ValueError("sequence is longer than this positional encoding")
        return inputs + self.encoding[:, : inputs.size(1)].to(dtype=inputs.dtype)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm self-attention, FFN, and residual connections."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")

        self.norm_attention = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.norm_attention(inputs)
        attention, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        x = inputs + self.attention_dropout(attention)
        return x + self.ffn_dropout(self.ffn(self.norm_ffn(x)))


class TrafficTransformer(nn.Module):
    """Encoder-only Transformer mapping ``[B, 288, 1913]`` to ``[B, 144, 1913]``."""

    def __init__(
        self,
        *,
        num_stations: int = 1_913,
        input_steps: int = 288,
        forecast_steps: int = 144,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(num_stations, input_steps, forecast_steps, num_encoder_layers) <= 0:
            raise ValueError("model dimensions and num_encoder_layers must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")

        self.num_stations = num_stations
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.input_projection = nn.Linear(num_stations, d_model)
        self.input_positional_encoding = PositionalEncoding(input_steps, d_model)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        # This shared head uses d_model * num_stations weights, avoiding a large
        # d_model -> (forecast_steps * num_stations) dense projection.
        self.forecast_positional_encoding = PositionalEncoding(forecast_steps, d_model)
        self.forecast_ffn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.output_projection = nn.Linear(d_model, num_stations)

    def forward(self, inputs: Tensor) -> Tensor:
        expected_shape = (self.input_steps, self.num_stations)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected_shape:
            raise ValueError(
                f"inputs must have shape [batch, {self.input_steps}, {self.num_stations}], "
                f"got {tuple(inputs.shape)}"
            )

        encoded = self.input_projection(inputs)
        encoded = self.input_positional_encoding(encoded)
        encoded = self.encoder(encoded)

        context = encoded.mean(dim=1)
        future = context.unsqueeze(1).expand(-1, self.forecast_steps, -1)
        future = self.forecast_positional_encoding(future)
        future = self.forecast_ffn(future)
        return self.output_projection(future)


def build_traffic_transformer(**kwargs: int | float) -> TrafficTransformer:
    """Construct the traffic Transformer using the recommended compact defaults."""
    return TrafficTransformer(**kwargs)
