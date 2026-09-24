"""Train the joint multi-station traffic Transformer."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .dataloader import NumpyDataLoader
    from .dataset import load_pems_datasets
    from .model import build_traffic_transformer
except ImportError:  # Supports `python src/train.py`.
    from dataloader import NumpyDataLoader
    from dataset import load_pems_datasets
    from model import build_traffic_transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "data_processed" / "pems_la.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
PROTECTED_TIME_ONLY_PATH = PROJECT_ROOT / "models" / "best_model_time_only.pth"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_loss(
    model: nn.Module,
    loader: NumpyDataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate MSE over a complete validation loader."""
    if not len(loader):
        raise ValueError("The validation split produced no windows")

    model.eval()
    total_loss = 0.0
    total_elements = 0
    with torch.no_grad():
        for features, targets, past_time, future_time in loader:
            features = torch.from_numpy(features).to(device, non_blocking=True)
            targets = torch.from_numpy(targets).to(device, non_blocking=True)
            past_time = torch.from_numpy(past_time).to(device, non_blocking=True)
            future_time = torch.from_numpy(future_time).to(device, non_blocking=True)
            predictions = model(features, past_time, future_time)
            total_loss += criterion(predictions, targets).item() * targets.numel()
            total_elements += targets.numel()
    return total_loss / total_elements


def mean_seasonal_baseline_loss(
    loader: NumpyDataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """MSE of the 24-hour aligned baseline used by the residual model."""
    if not len(loader):
        raise ValueError("The validation split produced no windows")

    total_loss = 0.0
    total_elements = 0
    with torch.no_grad():
        for features, targets, _, _ in loader:
            features = torch.from_numpy(features).to(device, non_blocking=True)
            targets = torch.from_numpy(targets).to(device, non_blocking=True)
            baseline = features[:, : targets.size(1), :]
            total_loss += criterion(baseline, targets).item() * targets.numel()
            total_elements += targets.numel()
    return total_loss / total_elements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--station-limit", type=int, default=None, help="Use 5 for a mini run.")
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Validation proportion; train is 0.80 - val-ratio and test is 0.20.",
    )
    parser.add_argument(
        "--spatial-mode", choices=("none", "gated_knn", "gat_lite", "gcn_lite"), default="none"
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=8,
        help="Road-neighbor count for a spatial experiment (default: 8).",
    )
    parser.add_argument(
        "--experiment-name", default=None,
        help="Optional name used for a JSON epoch history under experiments/logs/.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.patience) <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if not 0.0 < args.val_ratio < 0.8:
        raise ValueError("val_ratio must lie between 0 and 0.8")
    if args.spatial_mode != "none" and args.knn_k <= 0:
        raise ValueError("A spatial experiment requires --knn-k > 0")
    if args.model_path.resolve() == PROTECTED_TIME_ONLY_PATH.resolve():
        raise ValueError(
            "Refusing to overwrite models/best_model_time_only.pth; choose a separate model path."
        )

    set_seed(args.seed)
    device = resolve_device(args.device)
    knn_k = args.knn_k if args.spatial_mode != "none" else None
    datasets = load_pems_datasets(
        args.data_path,
        station_limit=args.station_limit,
        knn_k=knn_k,
        val_ratio=args.val_ratio,
    )
    if not len(datasets.train) or not len(datasets.val):
        raise ValueError(
            "Train/validation windows are empty. Keep the full time range; a "
            "three-day subset is too short after the chronological split."
        )

    train_loader = NumpyDataLoader(
        datasets.train, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = NumpyDataLoader(datasets.val, batch_size=args.batch_size)

    model_config = {
        "num_stations": len(datasets.station_ids),
        "input_steps": datasets.train.input_steps,
        "forecast_steps": datasets.train.forecast_steps,
        "d_model": 128,
        "nhead": 8,
        "num_encoder_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "spatial_mode": args.spatial_mode,
    }
    model = build_traffic_transformer(
        **model_config,
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = (
        PROJECT_ROOT / "experiments" / "logs" / f"{args.experiment_name}_train.json"
        if args.experiment_name
        else None
    )
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0
    amp_enabled = device.type == "cuda"
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    baseline_val_loss = mean_seasonal_baseline_loss(val_loader, criterion, device)
    # The zero-initialized residual head exactly reproduces the seasonal baseline.
    # Preserve that safe checkpoint unless training genuinely improves on it.
    best_val_loss = baseline_val_loss
    best_state: dict[str, torch.Tensor] | None = deepcopy(model.state_dict())
    def checkpoint_payload(state: dict[str, torch.Tensor], val_mse: float) -> dict[str, object]:
        return {
            "model_state_dict": state,
            "model_config": model_config,
            "data_config": {
                "station_limit": args.station_limit,
                "knn_k": knn_k,
                "val_ratio": args.val_ratio,
            },
            "station_ids": datasets.station_ids,
            "scaler_mean": datasets.scaler.mean,
            "scaler_std": datasets.scaler.std,
            "best_val_mse": val_mse,
        }

    torch.save(checkpoint_payload(best_state, best_val_loss), args.model_path)

    print(
        f"Device: {device}; stations: {len(datasets.station_ids)}; "
        f"train/val/test windows: {len(datasets.train)}/{len(datasets.val)}/{len(datasets.test)}"
    )
    print(f"Validation seasonal-baseline MSE: {baseline_val_loss:.6f}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_elements = 0
        for features, targets, past_time, future_time in train_loader:
            features = torch.from_numpy(features).to(device, non_blocking=True)
            targets = torch.from_numpy(targets).to(device, non_blocking=True)
            past_time = torch.from_numpy(past_time).to(device, non_blocking=True)
            future_time = torch.from_numpy(future_time).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(features, past_time, future_time)
                loss = criterion(predictions, targets)
            gradient_scaler.scale(loss).backward()
            gradient_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            total_train_loss += loss.item() * targets.numel()
            total_train_elements += targets.numel()

        train_loss = total_train_loss / total_train_elements
        val_loss = mean_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}: train_mse={train_loss:.6f}, "
            f"val_mse={val_loss:.6f}, baseline_mse={baseline_val_loss:.6f}, "
            f"lr={learning_rate:.2e}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "baseline_val_mse": baseline_val_loss,
                "learning_rate": learning_rate,
            }
        )
        if history_path is not None:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(checkpoint_payload(best_state, best_val_loss), args.model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    print(f"Best validation MSE: {best_val_loss:.6f}")
    print(f"Saved checkpoint: {args.model_path}")
    if history_path is not None:
        print(f"Epoch history: {history_path}")


if __name__ == "__main__":
    main()
