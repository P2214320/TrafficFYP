"""Evaluate a trained joint multi-station traffic Transformer on the test split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--station-limit",
        type=int,
        default=None,
        help="Override the station limit stored in the checkpoint.",
    )
    parser.add_argument("--knn-k", type=int, default=None, help="Override KNN neighbor count.")
    parser.add_argument(
        "--val-ratio", type=float, default=None,
        help="Override the validation ratio saved in the checkpoint.",
    )
    parser.add_argument(
        "--experiment-name", default=None,
        help="Append this evaluation to experiments/ablation_log.md.",
    )
    parser.add_argument(
        "--ablation-log", type=Path, default=PROJECT_ROOT / "experiments" / "ablation_log.md"
    )
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
    val_ratio = (
        args.val_ratio
        if args.val_ratio is not None
        else checkpoint["data_config"].get("val_ratio", 0.10)
    )
    if not 0.0 < val_ratio < 0.8:
        raise ValueError("val_ratio must lie between 0 and 0.8")
    spatial_mode = checkpoint["model_config"].get("spatial_mode", "none")
    # Historic time-only checkpoints need no graph even if their data config
    # contains a stale K value from an earlier experiment.
    if spatial_mode == "none":
        knn_k = None
    datasets = load_pems_datasets(
        args.data_path,
        station_limit=station_limit,
        knn_k=knn_k,
        val_ratio=val_ratio,
    )
    if len(datasets.station_ids) != checkpoint["model_config"]["num_stations"]:
        raise ValueError("Dataset station count does not match the checkpoint")
    if not len(datasets.test):
        raise ValueError("The test split produced no windows")

    model = build_traffic_transformer(
        **checkpoint["model_config"],
        neighbor_indices=datasets.neighbor_indices,
        adjacency_indices=(
            datasets.normalized_adjacency[0]
            if datasets.normalized_adjacency is not None
            else None
        ),
        adjacency_values=(
            datasets.normalized_adjacency[1]
            if datasets.normalized_adjacency is not None
            else None
        ),
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
    baseline_absolute_error_sum = 0.0
    baseline_squared_error_sum = 0.0
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
            baseline_errors = (features[:, : targets.size(1), :] * scaler_std + scaler_mean) - (
                targets * scaler_std + scaler_mean
            )
            absolute_error_sum += errors.abs().sum().item()
            squared_error_sum += errors.square().sum().item()
            baseline_absolute_error_sum += baseline_errors.abs().sum().item()
            baseline_squared_error_sum += baseline_errors.square().sum().item()
            value_count += targets.numel()

    mae = absolute_error_sum / value_count
    rmse = math.sqrt(squared_error_sum / value_count)
    baseline_mae = baseline_absolute_error_sum / value_count
    baseline_rmse = math.sqrt(baseline_squared_error_sum / value_count)
    print(f"Test MAE (flow):  {mae:.6f}")
    print(f"Test RMSE (flow): {rmse:.6f}")
    print(f"Baseline MAE (flow):  {baseline_mae:.6f}")
    print(f"Baseline RMSE (flow): {baseline_rmse:.6f}")

    if args.experiment_name:
        args.ablation_log.parent.mkdir(parents=True, exist_ok=True)
        if not args.ablation_log.exists():
            args.ablation_log.write_text(
                "# Spatial ablation log\n\n"
                "All model and seasonal-baseline metrics are calculated on the same test windows.\n\n"
                "| Experiment | Spatial mode | Stations | K | Train/Val/Test | Best val MSE | "
                "Model MAE | Model RMSE | Baseline MAE | Baseline RMSE | Checkpoint |\n"
                "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|\n",
                encoding="utf-8",
            )
        split = f"{0.80 - val_ratio:.0%}/{val_ratio:.0%}/20%"
        with args.ablation_log.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"| {args.experiment_name} | {spatial_mode} | {len(datasets.station_ids)} | "
                f"{knn_k or '-'} | {split} | {checkpoint.get('best_val_mse', float('nan')):.6f} | "
                f"{mae:.6f} | {rmse:.6f} | {baseline_mae:.6f} | {baseline_rmse:.6f} | "
                f"{args.model_path.as_posix()} |\n"
            )
        print(f"Ablation entry appended: {args.ablation_log}")


if __name__ == "__main__":
    main()
