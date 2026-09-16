"""Train the joint multi-station traffic Transformer."""

from __future__ import annotations

import argparse
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
    with torch.no_grad():
        for features, targets in loader:
            features = torch.from_numpy(features).to(device, non_blocking=True)
            targets = torch.from_numpy(targets).to(device, non_blocking=True)
            predictions = model(features)
            total_loss += criterion(predictions, targets).item()
    return total_loss / len(loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4, help="Use 4 on a 6 GB RTX 3060.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--station-limit", type=int, default=None, help="Use 5 for a mini run.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.patience) <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    datasets = load_pems_datasets(args.data_path, station_limit=args.station_limit)
    if not len(datasets.train) or not len(datasets.val):
        raise ValueError(
            "Train/validation windows are empty. Keep the full time range; a "
            "three-day subset is too short after a 70%/10%/20% split."
        )

    train_loader = NumpyDataLoader(
        datasets.train, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = NumpyDataLoader(datasets.val, batch_size=args.batch_size)

    model_config = {
        "num_stations": len(datasets.station_ids),
        "input_steps": datasets.train.input_steps,
        "forecast_steps": datasets.train.forecast_steps,
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
    }
    model = build_traffic_transformer(**model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    amp_enabled = device.type == "cuda"
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(
        f"Device: {device}; stations: {len(datasets.station_ids)}; "
        f"train/val/test windows: {len(datasets.train)}/{len(datasets.val)}/{len(datasets.test)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for features, targets in train_loader:
            features = torch.from_numpy(features).to(device, non_blocking=True)
            targets = torch.from_numpy(targets).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(features)
                loss = criterion(predictions, targets)
            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_loss = mean_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d}: train_mse={train_loss:.6f}, val_mse={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_config": model_config,
                    "data_config": {"station_limit": args.station_limit},
                    "station_ids": datasets.station_ids,
                    "scaler_mean": datasets.scaler.mean,
                    "scaler_std": datasets.scaler.std,
                    "best_val_mse": best_val_loss,
                },
                args.model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was saved")
    print(f"Best validation MSE: {best_val_loss:.6f}")
    print(f"Saved checkpoint: {args.model_path}")


if __name__ == "__main__":
    main()
