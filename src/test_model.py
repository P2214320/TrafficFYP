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
    past_time = torch.randint(0, 168, (2, 288), dtype=torch.long)
    future_time = torch.randint(0, 168, (2, 144), dtype=torch.long)
    with torch.no_grad():
        outputs = model(inputs, past_time, future_time)

    expected_shape = (2, 144, 1_913)
    if tuple(outputs.shape) != expected_shape:
        raise AssertionError(f"Expected {expected_shape}, got {tuple(outputs.shape)}")
    seasonal_baseline = inputs[:, :144, :]
    if not torch.allclose(outputs, seasonal_baseline):
        raise AssertionError("Zero-initialized residual head must initially reproduce the baseline")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Input shape:  {tuple(inputs.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")
    print("Seasonal residual path: verified")
    print(f"Parameters: {parameter_count:,}")


if __name__ == "__main__":
    main()
