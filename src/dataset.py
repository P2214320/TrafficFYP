"""Memory-conscious PeMS LA loading, preprocessing, and windowed datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


TIME_FORMAT = "%m/%d/%Y %H:%M:%S"
INPUT_STEPS = 288
FORECAST_STEPS = 144
STRIDE = 12
RAW_COLUMNS = ["timestamp", "station_id", "flow"]


def _station_sort_key(station_id: str) -> tuple[int, int | str]:
    """Give numeric station identifiers a stable numerical ordering."""
    return (0, int(station_id)) if station_id.isdigit() else (1, station_id)


def _read_chunks(
    csv_path: Path, chunksize: int, max_raw_rows: int | None
) -> Iterator[pd.DataFrame]:
    """Yield only the three columns needed to construct the flow matrix."""
    yield from pd.read_csv(
        csv_path,
        usecols=RAW_COLUMNS,
        dtype={"station_id": "string", "flow": "float32"},
        chunksize=chunksize,
        nrows=max_raw_rows,
    )


def read_pems_wide(
    csv_path: str | Path,
    *,
    chunksize: int = 500_000,
    max_raw_rows: int | None = None,
) -> pd.DataFrame:
    """Read the long CSV in two streaming passes and return ``[T, N]`` flow data.

    The first pass collects the small timestamp/station vocabularies.  The second
    pass fills a preallocated ``float32`` matrix, so the complete 1+ GB long-form
    table is never held in memory.  Columns are station IDs in deterministic
    numerical order and the index is chronological.

    ``max_raw_rows`` is intended for smoke tests.  It limits both passes to the
    same leading records and is not used in normal training.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"PeMS CSV not found: {path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if max_raw_rows is not None and max_raw_rows <= 0:
        raise ValueError("max_raw_rows must be positive when provided")

    timestamps: set[str] = set()
    station_ids: set[str] = set()
    for chunk in _read_chunks(path, chunksize, max_raw_rows):
        timestamps.update(chunk["timestamp"].dropna().astype(str).unique())
        station_ids.update(chunk["station_id"].dropna().astype(str).unique())

    if not timestamps or not station_ids:
        raise ValueError("No timestamp/station records were found in the CSV")

    ordered_timestamps = sorted(
        timestamps,
        key=lambda value: pd.to_datetime(value, format=TIME_FORMAT, errors="raise"),
    )
    ordered_station_ids = sorted(station_ids, key=_station_sort_key)
    time_positions = {timestamp: index for index, timestamp in enumerate(ordered_timestamps)}
    station_positions = {
        station_id: index for index, station_id in enumerate(ordered_station_ids)
    }
    values = np.full(
        (len(ordered_timestamps), len(ordered_station_ids)), np.nan, dtype=np.float32
    )

    for chunk in _read_chunks(path, chunksize, max_raw_rows):
        time_indices = chunk["timestamp"].map(time_positions)
        station_indices = chunk["station_id"].astype(str).map(station_positions)
        valid = time_indices.notna() & station_indices.notna()
        if not valid.any():
            continue

        values[
            time_indices.loc[valid].to_numpy(dtype=np.intp),
            station_indices.loc[valid].to_numpy(dtype=np.intp),
        ] = chunk.loc[valid, "flow"].to_numpy(dtype=np.float32, copy=False)

    datetime_index = pd.DatetimeIndex(
        pd.to_datetime(ordered_timestamps, format=TIME_FORMAT, errors="raise"),
        name="timestamp",
    )
    return pd.DataFrame(values, index=datetime_index, columns=ordered_station_ids)


def _interpolate_within_split(values: np.ndarray) -> np.ndarray:
    """Linearly fill gaps without accessing another train/validation/test split."""
    filled = pd.DataFrame(values).interpolate(axis=0, limit_direction="both")
    if filled.isna().any(axis=None):
        raise ValueError("A station has no observed flow values in this split")
    return filled.to_numpy(dtype=np.float32, copy=False)


@dataclass(frozen=True)
class FlowScaler:
    """Per-station standardization parameters fitted on the training split only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, train_values: np.ndarray) -> "FlowScaler":
        mean = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < np.finfo(np.float32).eps] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray((values - self.mean) / self.std, dtype=np.float32)


class PemsFlowDataset:
    """A strided, multi-station forecasting dataset backed by a ``float32`` array."""

    def __init__(
        self,
        values: np.ndarray,
        *,
        input_steps: int = INPUT_STEPS,
        forecast_steps: int = FORECAST_STEPS,
        stride: int = STRIDE,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [time, station]")
        if min(input_steps, forecast_steps, stride) <= 0:
            raise ValueError("input_steps, forecast_steps, and stride must be positive")

        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.stride = stride
        available = len(self.values) - input_steps - forecast_steps
        self._length = 0 if available < 0 else available // stride + 1

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index * self.stride
        split = start + self.input_steps
        end = split + self.forecast_steps
        return self.values[start:split], self.values[split:end]


@dataclass(frozen=True)
class PemsDatasetSplits:
    train: PemsFlowDataset
    val: PemsFlowDataset
    test: PemsFlowDataset
    scaler: FlowScaler
    station_ids: tuple[str, ...]
    train_timestamps: pd.DatetimeIndex
    val_timestamps: pd.DatetimeIndex
    test_timestamps: pd.DatetimeIndex


def load_pems_datasets(
    csv_path: str | Path,
    *,
    chunksize: int = 500_000,
    max_raw_rows: int | None = None,
    input_steps: int = INPUT_STEPS,
    forecast_steps: int = FORECAST_STEPS,
    stride: int = STRIDE,
) -> PemsDatasetSplits:
    """Load, split by time, interpolate, and normalize PeMS data safely.

    The split occurs *before* interpolation and normalization.  The scaler is
    fitted solely on the imputed training partition, then reused unchanged for
    validation and test data.
    """
    wide = read_pems_wide(
        csv_path, chunksize=chunksize, max_raw_rows=max_raw_rows
    )
    total_steps = len(wide)
    train_end = int(total_steps * 0.70)
    val_end = train_end + int(total_steps * 0.10)
    if train_end == 0 or val_end == train_end or val_end == total_steps:
        raise ValueError("Not enough timestamps for a 70%/10%/20% split")

    train_values = _interpolate_within_split(wide.iloc[:train_end].to_numpy())
    val_values = _interpolate_within_split(wide.iloc[train_end:val_end].to_numpy())
    test_values = _interpolate_within_split(wide.iloc[val_end:].to_numpy())

    scaler = FlowScaler.fit(train_values)
    return PemsDatasetSplits(
        train=PemsFlowDataset(
            scaler.transform(train_values),
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            stride=stride,
        ),
        val=PemsFlowDataset(
            scaler.transform(val_values),
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            stride=stride,
        ),
        test=PemsFlowDataset(
            scaler.transform(test_values),
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            stride=stride,
        ),
        scaler=scaler,
        station_ids=tuple(wide.columns.astype(str)),
        train_timestamps=wide.index[:train_end],
        val_timestamps=wide.index[train_end:val_end],
        test_timestamps=wide.index[val_end:],
    )
