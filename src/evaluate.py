"""Evaluate a trained joint multi-station traffic Transformer on the test split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

try:
    from .dataloader import NumpyDataLoader
    from .dataset import load_pems_datasets
    from .model import build_traffic_transformer
    from .train import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, resolve_device
except ImportError:  # Supports `python src/evaluate.py`.
    from dataloader import NumpyDataLoader
    from dataset import load_pems_datasets
    from model import build_traffic_transformer
    from train import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=4, help="Use 4 on a 6 GB RTX 3060.")
    parser.add_argument(
        "--station-limit",
        type=int,
        default=None,
        help="Override the station limit stored in the checkpoint.",
    )
    parser.add_argument("--knn-k", type=int, default=None, help="Override KNN neighbor count.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    station_limit = (
        args.station_limit
        if args.station_limit is not None
        else checkpoint["data_config"]["station_limit"]
    )
    knn_k = (
        args.knn_k
        if args.knn_k is not None
        else checkpoint["data_config"].get("knn_k")
    )
    datasets = load_pems_datasets(
        args.data_path, station_limit=station_limit, knn_k=knn_k
    )
    if len(datasets.station_ids) != checkpoint["model_config"]["num_stations"]:
        raise ValueError("Dataset station count does not match the checkpoint")
    if not len(datasets.test):
        raise ValueError("The test split produced no windows")

    model = build_traffic_transformer(
        **checkpoint["model_config"], neighbor_indices=datasets.neighbor_indices
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_loader = NumpyDataLoader(datasets.test, batch_size=args.batch_size)
    scaler_mean = torch.as_tensor(
        checkpoint["scaler_mean"], dtype=torch.float32, device=device
    ).view(1, 1, -1)
    scaler_std = torch.as_tensor(
        checkpoint["scaler_std"], dtype=torch.float32, device=device
    ).view(1, 1, -1)

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    value_count = 0
    with torch.no_grad():
        for features, targets, past_time, future_time in test_loader:
            features = torch.from_numpy(features).to(device, non_blocking=True)
            targets = torch.from_numpy(targets).to(device, non_blocking=True)
            past_time = torch.from_numpy(past_time).to(device, non_blocking=True)
            future_time = torch.from_numpy(future_time).to(device, non_blocking=True)
            predictions = model(features, past_time, future_time)
            # Dataset targets are normalized; report traffic-flow metrics in the
            # original units by applying the scaler fitted during training.
            errors = (predictions * scaler_std + scaler_mean) - (
                targets * scaler_std + scaler_mean
            )
            absolute_error_sum += errors.abs().sum().item()
            squared_error_sum += errors.square().sum().item()
            value_count += targets.numel()

    mae = absolute_error_sum / value_count
    rmse = math.sqrt(squared_error_sum / value_count)
    print(f"Test MAE (flow):  {mae:.6f}")
    print(f"Test RMSE (flow): {rmse:.6f}")


if __name__ == "__main__":
    main()
