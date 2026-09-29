"""Export a compact, front-end-ready traffic-flow forecast and paper figure.

The script exports one complete 24-hour input / 12-hour forecast window rather
than the full test tensor.  This keeps the CSV small enough for the Vue client
while retaining the actual values needed for a defensible demonstration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from ..dataset import load_pems_datasets
    from ..model import build_traffic_transformer
    from ..train import DEFAULT_DATA_PATH, PROTECTED_TIME_ONLY_PATH, resolve_device
except ImportError:  # Supports `python src/visualization/export_flow_predictions.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import load_pems_datasets
    from model import build_traffic_transformer
    from train import DEFAULT_DATA_PATH, PROTECTED_TIME_ONLY_PATH, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=PROTECTED_TIME_ONLY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--station-ids",
        nargs="+",
        default=None,
        help="One to five station IDs. Defaults to three evenly spaced station columns.",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=-1,
        help="Test-window index; -1 selects the final available test window.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    return parser.parse_args()


def resolve_window_index(index: int, length: int) -> int:
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise IndexError(f"window-index {index} is outside the test dataset of length {length}")
    return index


def select_station_indices(
    station_ids: tuple[str, ...], requested_station_ids: list[str] | None
) -> tuple[list[int], list[str]]:
    if requested_station_ids is None:
        positions = sorted(set(np.linspace(0, len(station_ids) - 1, num=3, dtype=int).tolist()))
        return positions, [station_ids[position] for position in positions]
    if not 1 <= len(requested_station_ids) <= 5:
        raise ValueError("Select between one and five station IDs for an export")
    positions_by_id = {station_id: position for position, station_id in enumerate(station_ids)}
    missing = [station_id for station_id in requested_station_ids if station_id not in positions_by_id]
    if missing:
        raise ValueError(f"Unknown station IDs: {', '.join(missing)}")
    return [positions_by_id[station_id] for station_id in requested_station_ids], requested_station_ids


def load_model_and_data(args: argparse.Namespace) -> tuple[torch.nn.Module, object, dict, torch.device]:
    device = resolve_device(args.device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    data_config = checkpoint["data_config"]
    model_config = checkpoint["model_config"]
    spatial_mode = model_config.get("spatial_mode", "none")
    knn_k = data_config.get("knn_k") if spatial_mode != "none" else None
    datasets = load_pems_datasets(
        args.data_path,
        station_limit=data_config.get("station_limit"),
        knn_k=knn_k,
        val_ratio=data_config.get("val_ratio", 0.10),
    )
    if len(datasets.station_ids) != model_config["num_stations"]:
        raise ValueError("Dataset station count does not match the checkpoint")
    adjacency = datasets.normalized_adjacency
    model = build_traffic_transformer(
        **model_config,
        neighbor_indices=datasets.neighbor_indices,
        adjacency_indices=adjacency[0] if adjacency is not None else None,
        adjacency_values=adjacency[1] if adjacency is not None else None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, datasets, checkpoint, device


def main() -> None:
    args = parse_args()
    model, datasets, checkpoint, device = load_model_and_data(args)
    test_dataset = datasets.test
    index = resolve_window_index(args.window_index, len(test_dataset))
    positions, selected_ids = select_station_indices(datasets.station_ids, args.station_ids)
    features, targets, past_time, future_time = test_dataset[index]

    with torch.no_grad():
        prediction = model(
            torch.from_numpy(features).unsqueeze(0).to(device),
            torch.from_numpy(past_time).unsqueeze(0).to(device),
            torch.from_numpy(future_time).unsqueeze(0).to(device),
        ).squeeze(0).cpu().numpy()

    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["scaler_std"], dtype=np.float32)
    history_flow = features * std + mean
    actual_flow = targets * std + mean
    prediction_flow = prediction * std + mean
    baseline_flow = features[: targets.shape[0]] * std + mean

    start = index * test_dataset.stride
    split = start + test_dataset.input_steps
    end = split + test_dataset.forecast_steps
    historical_timestamps = datasets.test_timestamps[start:split]
    forecast_timestamps = datasets.test_timestamps[split:end]
    rows: list[dict[str, object]] = []
    for station_position, station_id in zip(positions, selected_ids, strict=True):
        metadata = datasets.station_metadata.loc[station_id]
        common = {
            "station_id": station_id,
            "freeway": metadata["freeway"],
            "direction": metadata["dir"],
            "abs_pm": metadata["abs_pm"],
            "latitude": metadata["latitude"],
            "longitude": metadata["longitude"],
        }
        for timestamp, flow in zip(historical_timestamps, history_flow[:, station_position], strict=True):
            rows.append({**common, "timestamp": timestamp, "phase": "history", "actual_flow": flow,
                         "model_prediction": np.nan, "seasonal_baseline": np.nan})
        for timestamp, actual, predicted, baseline in zip(
            forecast_timestamps,
            actual_flow[:, station_position],
            prediction_flow[:, station_position],
            baseline_flow[:, station_position],
            strict=True,
        ):
            rows.append({**common, "timestamp": timestamp, "phase": "forecast", "actual_flow": actual,
                         "model_prediction": predicted, "seasonal_baseline": baseline})

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    export = pd.DataFrame(rows)
    csv_path = output_dir / "forecast_window.csv"
    export.to_csv(csv_path, index=False, float_format="%.4f")

    selected_actual = actual_flow[:, positions]
    selected_prediction = prediction_flow[:, positions]
    selected_baseline = baseline_flow[:, positions]
    metrics = {
        "checkpoint": str(args.model_path),
        "window_index": index,
        "selected_station_ids": selected_ids,
        "forecast_steps": int(targets.shape[0]),
        "model_mae": float(np.abs(selected_prediction - selected_actual).mean()),
        "model_rmse": float(np.sqrt(np.square(selected_prediction - selected_actual).mean())),
        "baseline_mae": float(np.abs(selected_baseline - selected_actual).mean()),
        "baseline_rmse": float(np.sqrt(np.square(selected_baseline - selected_actual).mean())),
    }
    metrics_path = output_dir / "forecast_window_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(len(positions), 1, figsize=(14, 4 * len(positions)), sharex=True)
    if len(positions) == 1:
        axes = [axes]
    for axis, station_position, station_id in zip(axes, positions, selected_ids, strict=True):
        axis.plot(historical_timestamps, history_flow[:, station_position], color="#68737d", label="History")
        axis.plot(forecast_timestamps, actual_flow[:, station_position], color="#1f77b4", label="Actual")
        axis.plot(forecast_timestamps, prediction_flow[:, station_position], color="#e76f51", label="Transformer")
        axis.plot(forecast_timestamps, baseline_flow[:, station_position], color="#f4a261", linestyle="--", label="24-hour baseline")
        axis.axvline(forecast_timestamps[0], color="#4b5563", linestyle=":", linewidth=1)
        axis.set_title(f"Station {station_id}")
        axis.set_ylabel("Flow")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Timestamp")
    figure.suptitle("PeMS LA: 24-hour history and 12-hour forecast", y=0.995)
    figure.tight_layout()
    figure_path = output_dir / "forecast_window.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Exported CSV: {csv_path}")
    print(f"Exported metrics: {metrics_path}")
    print(f"Exported figure: {figure_path}")
    print(
        f"Selected-window MAE/RMSE: model={metrics['model_mae']:.4f}/{metrics['model_rmse']:.4f}; "
        f"baseline={metrics['baseline_mae']:.4f}/{metrics['baseline_rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
