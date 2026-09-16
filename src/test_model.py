"""Smoke test for the PeMS traffic Transformer shape contract."""

from __future__ import annotations

import torch

try:
    from .model import build_traffic_transformer
except ImportError:  # Supports `python src/test_model.py`.
    from model import build_traffic_transformer


def main() -> None:
    torch.manual_seed(42)
    model = build_traffic_transformer().eval()
    inputs = torch.randn(2, 288, 1_913, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)

    expected_shape = (2, 144, 1_913)
    if tuple(outputs.shape) != expected_shape:
        raise AssertionError(f"Expected {expected_shape}, got {tuple(outputs.shape)}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Input shape:  {tuple(inputs.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")
    print(f"Parameters: {parameter_count:,}")


if __name__ == "__main__":
    main()
