"""Smoke test for the PeMS traffic Transformer shape contract."""

from __future__ import annotations

import torch

try:
    from .model import build_traffic_transformer
except ImportError:  # Supports `python src/test_model.py`.
    from model import build_traffic_transformer


def main() -> None:
    torch.manual_seed(42)
    station_count = 1_913
    offsets = torch.arange(9, dtype=torch.long)
    neighbor_indices = (torch.arange(station_count).unsqueeze(1) + offsets) % station_count
    inputs = torch.randn(2, 288, station_count, dtype=torch.float32)
    past_time = torch.randint(0, 168, (2, 288), dtype=torch.long)
    future_time = torch.randint(0, 168, (2, 144), dtype=torch.long)
    centers = torch.arange(station_count)
    source = centers.repeat_interleave(2)
    target = torch.stack(((centers - 1) % station_count, (centers + 1) % station_count), dim=1).flatten()
    adjacency_indices = torch.stack((source, target))
    adjacency_values = torch.full((source.numel(),), 0.5)
    expected_shape = (2, 144, station_count)
    seasonal_baseline = inputs[:, :144, :]

    for spatial_mode in ("none", "gated_knn", "gat_lite", "gcn_lite"):
        model = build_traffic_transformer(
            spatial_mode=spatial_mode,
            neighbor_indices=neighbor_indices if spatial_mode in {"gated_knn", "gat_lite"} else None,
            adjacency_indices=adjacency_indices if spatial_mode == "gcn_lite" else None,
            adjacency_values=adjacency_values if spatial_mode == "gcn_lite" else None,
        ).eval()
        with torch.no_grad():
            outputs = model(inputs, past_time, future_time)
        if tuple(outputs.shape) != expected_shape:
            raise AssertionError(f"{spatial_mode}: expected {expected_shape}, got {tuple(outputs.shape)}")
        if not torch.allclose(outputs, seasonal_baseline):
            raise AssertionError(f"{spatial_mode}: zero start must reproduce the baseline")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{spatial_mode}: {parameter_count:,} parameters; shape verified")

    print(f"Input shape:  {tuple(inputs.shape)}")
    print(f"Output shape: {expected_shape}")
    print("All spatial residual paths: zero-gated baseline verified")


if __name__ == "__main__":
    main()
