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


class CrossAttentionForecastBlock(nn.Module):
    """Let each future query read the relevant encoded history."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        attention, _ = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        x = queries + self.attention_dropout(attention)
        return x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))


class _NeighborResidualMixer(nn.Module):
    """Base class for spatial residual mixers with an exact-safe zero start."""

    def __init__(self, num_stations: int) -> None:
        super().__init__()
        # A station-wise linear gate is deliberately zero-initialised.  At the
        # first update ``tanh(0)=0``, so the spatial model exactly matches the
        # validated time-only baseline rather than perturbing every station.
        self.gate_weight = nn.Parameter(torch.zeros(num_stations))
        self.gate_bias = nn.Parameter(torch.zeros(num_stations))

    def _apply_gate(self, inputs: Tensor, neighbor_aggregate: Tensor) -> Tensor:
        gate = torch.tanh(neighbor_aggregate * self.gate_weight + self.gate_bias)
        return inputs + gate * neighbor_aggregate


class GatedKNNResidualMixer(_NeighborResidualMixer):
    """Add a zero-gated mean of each station's fixed KNN road neighbors."""

    def __init__(self, neighbor_indices: Tensor) -> None:
        indices = torch.as_tensor(neighbor_indices, dtype=torch.long)
        if indices.ndim != 2 or indices.size(1) < 2:
            raise ValueError("neighbor_indices must be [station, self_plus_neighbors]")
        super().__init__(indices.size(0))
        # Dataset graphs place self at column zero.  It must not leak into the
        # neighbour mean, otherwise this is partly an identity transformation.
        self.register_buffer("neighbor_indices", indices[:, 1:])

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.size(-1) != self.neighbor_indices.size(0):
            raise ValueError("KNN graph station count does not match model input")
        neighbor_mean = inputs[:, :, self.neighbor_indices].mean(dim=-1)
        return self._apply_gate(inputs, neighbor_mean)


class GATLiteResidualMixer(_NeighborResidualMixer):
    """Static per-station attention over KNN neighbours, protected by a gate."""

    def __init__(self, neighbor_indices: Tensor) -> None:
        indices = torch.as_tensor(neighbor_indices, dtype=torch.long)
        if indices.ndim != 2 or indices.size(1) < 2:
            raise ValueError("neighbor_indices must be [station, self_plus_neighbors]")
        super().__init__(indices.size(0))
        self.register_buffer("neighbor_indices", indices[:, 1:])
        # Equal logits yield a uniform KNN average at start; the zero gate keeps
        # its contribution exactly off until optimisation finds useful signal.
        self.attention_logits = nn.Parameter(torch.zeros_like(indices[:, 1:], dtype=torch.float32))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.size(-1) != self.neighbor_indices.size(0):
            raise ValueError("KNN graph station count does not match model input")
        neighbor_values = inputs[:, :, self.neighbor_indices]
        weights = torch.softmax(self.attention_logits, dim=-1)
        neighbor_aggregate = (neighbor_values * weights).sum(dim=-1)
        return self._apply_gate(inputs, neighbor_aggregate)


class GCNLiteResidualMixer(_NeighborResidualMixer):
    """Sparse normalized graph aggregation followed by the same safe gate."""

    def __init__(
        self, num_stations: int, adjacency_indices: Tensor, adjacency_values: Tensor
    ) -> None:
        super().__init__(num_stations)
        indices = torch.as_tensor(adjacency_indices, dtype=torch.long)
        values = torch.as_tensor(adjacency_values, dtype=torch.float32)
        if indices.ndim != 2 or indices.shape[0] != 2 or indices.shape[1] != values.numel():
            raise ValueError("Sparse adjacency must have indices [2, edges] and matching values")
        self.register_buffer("adjacency_indices", indices)
        self.register_buffer("adjacency_values", values)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.size(-1) != self.gate_weight.numel():
            raise ValueError("GCN graph station count does not match model input")
        # CUDA sparse.mm has no float16 kernel.  Keep this small graph-only
        # operation in float32 while the attention/FFN remains AMP-enabled.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            adjacency = torch.sparse_coo_tensor(
                self.adjacency_indices,
                self.adjacency_values,
                (inputs.size(-1), inputs.size(-1)),
                device=inputs.device,
            ).coalesce()
            # sparse.mm aggregates every station for all B*T observations at once.
            values = inputs.float().permute(2, 0, 1).reshape(inputs.size(-1), -1)
            aggregate = torch.sparse.mm(adjacency, values)
        neighbor_aggregate = aggregate.reshape(inputs.size(-1), inputs.size(0), inputs.size(1)).permute(1, 2, 0)
        return self._apply_gate(inputs, neighbor_aggregate)


class TrafficTransformer(nn.Module):
    """Encoder-only Transformer mapping ``[B, 288, 1913]`` to ``[B, 144, 1913]``."""

    def __init__(
        self,
        *,
        num_stations: int = 1_913,
        input_steps: int = 288,
        forecast_steps: int = 144,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        spatial_mode: str = "none",
        neighbor_indices: Tensor | None = None,
        adjacency_indices: Tensor | None = None,
        adjacency_values: Tensor | None = None,
    ) -> None:
        super().__init__()
        if min(num_stations, input_steps, forecast_steps, num_encoder_layers) <= 0:
            raise ValueError("model dimensions and num_encoder_layers must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")

        self.num_stations = num_stations
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        valid_spatial_modes = {"none", "gated_knn", "gat_lite", "gcn_lite"}
        if spatial_mode not in valid_spatial_modes:
            raise ValueError(f"spatial_mode must be one of {sorted(valid_spatial_modes)}")
        self.spatial_mode = spatial_mode
        if spatial_mode == "none":
            self.spatial_mixer: nn.Module | None = None
        elif spatial_mode == "gated_knn":
            if neighbor_indices is None:
                raise ValueError("gated_knn requires neighbor_indices")
            self.spatial_mixer = GatedKNNResidualMixer(neighbor_indices)
        elif spatial_mode == "gat_lite":
            if neighbor_indices is None:
                raise ValueError("gat_lite requires neighbor_indices")
            self.spatial_mixer = GATLiteResidualMixer(neighbor_indices)
        else:
            if adjacency_indices is None or adjacency_values is None:
                raise ValueError("gcn_lite requires sparse normalized adjacency")
            self.spatial_mixer = GCNLiteResidualMixer(
                num_stations, adjacency_indices, adjacency_values
            )
        self.input_projection = nn.Linear(num_stations, d_model)
        self.hour_embedding = nn.Embedding(24, 8)
        self.day_embedding = nn.Embedding(7, 4)
        self.time_projection = nn.Linear(12, d_model)
        self.input_positional_encoding = PositionalEncoding(input_steps, d_model)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        # Learnable horizon queries avoid collapsing all encoded timestamps into
        # one vector.  Each future step cross-attends to the full history.
        self.future_queries = nn.Parameter(torch.empty(forecast_steps, d_model))
        nn.init.normal_(self.future_queries, mean=0.0, std=0.02)
        self.forecast_positional_encoding = PositionalEncoding(forecast_steps, d_model)
        self.forecast_decoder = CrossAttentionForecastBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.output_projection = nn.Linear(d_model, num_stations)
        # Begin from the strong 24-hour seasonal baseline.  The network learns
        # residual corrections rather than reconstructing every station flow.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _time_embedding(self, time_codes: Tensor, expected_steps: int) -> Tensor:
        if time_codes.ndim != 2 or time_codes.size(1) != expected_steps:
            raise ValueError(f"time_codes must have shape [batch, {expected_steps}]")
        if time_codes.dtype != torch.long:
            time_codes = time_codes.long()
        if torch.any(time_codes < 0) or torch.any(time_codes >= 168):
            raise ValueError("time_codes must contain hour-of-week values in [0, 167]")

        hours = torch.remainder(time_codes, 24)
        days = torch.div(time_codes, 24, rounding_mode="floor")
        return self.time_projection(
            torch.cat((self.hour_embedding(hours), self.day_embedding(days)), dim=-1)
        )

    def forward(self, inputs: Tensor, past_time: Tensor, future_time: Tensor) -> Tensor:
        expected_shape = (self.input_steps, self.num_stations)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected_shape:
            raise ValueError(
                f"inputs must have shape [batch, {self.input_steps}, {self.num_stations}], "
                f"got {tuple(inputs.shape)}"
            )

        spatial_features = self.spatial_mixer(inputs) if self.spatial_mixer is not None else inputs
        encoded = self.input_projection(spatial_features) + self._time_embedding(
            past_time, self.input_steps
        )
        encoded = self.input_positional_encoding(encoded)
        encoded = self.encoder(encoded)

        future = self.future_queries.unsqueeze(0).expand(inputs.size(0), -1, -1)
        future = future + self._time_embedding(future_time, self.forecast_steps)
        future = self.forecast_positional_encoding(future)
        future = self.forecast_decoder(future, encoded)
        predicted_residual = self.output_projection(future)

        # Target point k is exactly 24 hours after input point k at a 5-minute
        # cadence: X[:, :144, :] is the aligned seasonal baseline for y.
        seasonal_baseline = inputs[:, : self.forecast_steps, :]
        return seasonal_baseline + predicted_residual


def build_traffic_transformer(**kwargs: object) -> TrafficTransformer:
    """Construct the traffic Transformer using the recommended compact defaults."""
    return TrafficTransformer(**kwargs)
