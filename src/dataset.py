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
STATION_METADATA_COLUMNS = ["station_id", "freeway", "dir", "abs_pm", "latitude", "longitude"]


def _station_sort_key(station_id: str) -> tuple[int, int | str]:
    """Give numeric station identifiers a stable numerical ordering."""
    return (0, int(station_id)) if station_id.isdigit() else (1, station_id)


def _read_chunks(
    csv_path: Path,
    chunksize: int,
    max_raw_rows: int | None,
    *,
    include_metadata: bool = False,
) -> Iterator[pd.DataFrame]:
    """Yield only the three columns needed to construct the flow matrix."""
    usecols = RAW_COLUMNS if not include_metadata else list(
        dict.fromkeys([*RAW_COLUMNS, *STATION_METADATA_COLUMNS])
    )
    dtypes: dict[str, str] = {"station_id": "string", "flow": "float32"}
    if include_metadata:
        dtypes.update(
            {
                "freeway": "string",
                "dir": "string",
                "abs_pm": "float32",
                "latitude": "float32",
                "longitude": "float32",
            }
        )
    yield from pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
        nrows=max_raw_rows,
    )


def _read_pems_wide_and_metadata(
    csv_path: str | Path,
    *,
    chunksize: int = 500_000,
    max_raw_rows: int | None = None,
    station_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the long CSV in two streaming passes and return ``[T, N]`` flow data.

    The first pass collects the small timestamp/station vocabularies.  The second
    pass fills a preallocated ``float32`` matrix, so the complete 1+ GB long-form
    table is never held in memory.  Columns are station IDs in deterministic
    numerical order and the index is chronological.

    ``max_raw_rows`` is intended for smoke tests.  It limits both passes to the
    same leading records and is not used in normal training.  ``station_limit``
    selects the lowest numeric station IDs, while retaining all timestamps; it
    is useful for an end-to-end mini-training run.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"PeMS CSV not found: {path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if max_raw_rows is not None and max_raw_rows <= 0:
        raise ValueError("max_raw_rows must be positive when provided")
    if station_limit is not None and station_limit <= 0:
        raise ValueError("station_limit must be positive when provided")

    timestamps: set[str] = set()
    station_ids: set[str] = set()
    metadata_by_station: dict[str, tuple[str, str, float, float, float]] = {}
    for chunk in _read_chunks(path, chunksize, max_raw_rows, include_metadata=True):
        timestamps.update(chunk["timestamp"].dropna().astype(str).unique())
        station_ids.update(chunk["station_id"].dropna().astype(str).unique())
        static_rows = chunk[STATION_METADATA_COLUMNS].drop_duplicates("station_id")
        for row in static_rows.itertuples(index=False):
            station_id = str(row.station_id)
            if station_id not in metadata_by_station:
                metadata_by_station[station_id] = (
                    str(row.freeway),
                    str(row.dir),
                    float(row.abs_pm),
                    float(row.latitude),
                    float(row.longitude),
                )

    if not timestamps or not station_ids:
        raise ValueError("No timestamp/station records were found in the CSV")

    ordered_timestamps = sorted(
        timestamps,
        key=lambda value: pd.to_datetime(value, format=TIME_FORMAT, errors="raise"),
    )
    ordered_station_ids = sorted(station_ids, key=_station_sort_key)
    if station_limit is not None:
        ordered_station_ids = ordered_station_ids[:station_limit]
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
    metadata = pd.DataFrame(
        [metadata_by_station.get(station_id, (np.nan,) * 5) for station_id in ordered_station_ids],
        index=pd.Index(ordered_station_ids, name="station_id"),
        columns=STATION_METADATA_COLUMNS[1:],
    )
    return pd.DataFrame(values, index=datetime_index, columns=ordered_station_ids), metadata


def read_pems_wide(
    csv_path: str | Path,
    *,
    chunksize: int = 500_000,
    max_raw_rows: int | None = None,
    station_limit: int | None = None,
) -> pd.DataFrame:
    """Read the long CSV in two streaming passes and return ``[T, N]`` flow data."""
    return _read_pems_wide_and_metadata(
        csv_path,
        chunksize=chunksize,
        max_raw_rows=max_raw_rows,
        station_limit=station_limit,
    )[0]


def build_station_knn(metadata: pd.DataFrame, k: int = 8) -> np.ndarray:
    """Build a fixed, sparse road-aware KNN graph with a self-loop in column 0."""
    if k <= 0:
        raise ValueError("k must be positive")
    required = {"freeway", "dir", "abs_pm", "latitude", "longitude"}
    if missing := required.difference(metadata.columns):
        raise ValueError(f"Station metadata is missing columns: {sorted(missing)}")

    station_count = len(metadata)
    if station_count < 2:
        raise ValueError("At least two stations are required to build KNN neighbors")
    neighbor_count = min(k, station_count - 1)
    freeway = metadata["freeway"].astype(str).to_numpy()
    direction = metadata["dir"].astype(str).to_numpy()
    abs_pm = metadata["abs_pm"].to_numpy(dtype=np.float32)
    latitude = metadata["latitude"].to_numpy(dtype=np.float32)
    longitude = metadata["longitude"].to_numpy(dtype=np.float32)
    all_indices = np.arange(station_count)
    neighbors = np.empty((station_count, neighbor_count + 1), dtype=np.int64)

    for index in range(station_count):
        same_road = (freeway == freeway[index]) & (direction == direction[index])
        same_road[index] = False
        candidates = all_indices[same_road]
        if len(candidates):
            pm_distance = np.abs(abs_pm[candidates] - abs_pm[index])
            candidates = candidates[np.argsort(np.nan_to_num(pm_distance, nan=np.inf))]

        selected = list(candidates[:neighbor_count])
        if len(selected) < neighbor_count:
            remaining = np.setdiff1d(all_indices, np.array([index, *selected]), assume_unique=False)
            geo_distance = (latitude[remaining] - latitude[index]) ** 2 + (
                longitude[remaining] - longitude[index]
            ) ** 2
            nearest = remaining[np.argsort(np.nan_to_num(geo_distance, nan=np.inf))]
            selected.extend(nearest[: neighbor_count - len(selected)].tolist())

        neighbors[index] = np.asarray([index, *selected], dtype=np.int64)
    return neighbors


def build_normalized_adjacency(neighbor_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create symmetric sparse D^-1/2 (A + I) D^-1/2 from a KNN graph."""
    station_count = neighbor_indices.shape[0]
    rows = np.repeat(np.arange(station_count, dtype=np.int64), neighbor_indices.shape[1])
    cols = neighbor_indices.reshape(-1)
    directed = np.stack((rows, cols), axis=0)
    undirected = np.concatenate((directed, directed[::-1]), axis=1)
    edges = np.unique(undirected, axis=1)
    degree = np.bincount(edges[0], minlength=station_count).astype(np.float32)
    values = 1.0 / np.sqrt(degree[edges[0]] * degree[edges[1]])
    return edges.astype(np.int64, copy=False), values.astype(np.float32, copy=False)


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
    """Windowed flows plus hour-of-week codes backed by contiguous arrays."""

    def __init__(
        self,
        values: np.ndarray,
        timestamps: pd.DatetimeIndex,
        *,
        input_steps: int = INPUT_STEPS,
        forecast_steps: int = FORECAST_STEPS,
        stride: int = STRIDE,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [time, station]")
        if len(values) != len(timestamps):
            raise ValueError("timestamps must align one-to-one with values")
        if min(input_steps, forecast_steps, stride) <= 0:
            raise ValueError("input_steps, forecast_steps, and stride must be positive")

        self.values = np.ascontiguousarray(values, dtype=np.float32)
        # One integer preserves the [time] contract while encoding both features:
        # 0..23 = Monday hours, 24..47 = Tuesday hours, ..., 144..167 = Sunday.
        self.time_codes = np.ascontiguousarray(
            timestamps.dayofweek.to_numpy(dtype=np.int64) * 24
            + timestamps.hour.to_numpy(dtype=np.int64),
            dtype=np.int64,
        )
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.stride = stride
        available = len(self.values) - input_steps - forecast_steps
        self._length = 0 if available < 0 else available // stride + 1

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index * self.stride
        split = start + self.input_steps
        end = split + self.forecast_steps
        return (
            self.values[start:split],
            self.values[split:end],
            self.time_codes[start:split],
            self.time_codes[split:end],
        )


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
    station_metadata: pd.DataFrame
    neighbor_indices: np.ndarray | None
    normalized_adjacency: tuple[np.ndarray, np.ndarray] | None


def load_pems_datasets(
    csv_path: str | Path,
    *,
    chunksize: int = 500_000,
    max_raw_rows: int | None = None,
    station_limit: int | None = None,
    knn_k: int | None = None,
    val_ratio: float = 0.15,
    input_steps: int = INPUT_STEPS,
    forecast_steps: int = FORECAST_STEPS,
    stride: int = STRIDE,
) -> PemsDatasetSplits:
    """Load, split by time, interpolate, and normalize PeMS data safely.

    The split occurs *before* interpolation and normalization.  The scaler is
    fitted solely on the imputed training partition, then reused unchanged for
    validation and test data.
    """
    wide, station_metadata = _read_pems_wide_and_metadata(
        csv_path,
        chunksize=chunksize,
        max_raw_rows=max_raw_rows,
        station_limit=station_limit,
    )
    if not 0.0 < val_ratio < 0.80:
        raise ValueError("val_ratio must be between 0 and 0.80")
    total_steps = len(wide)
    train_end = int(total_steps * (0.80 - val_ratio))
    val_end = int(total_steps * 0.80)
    if train_end == 0 or val_end == train_end or val_end == total_steps:
        raise ValueError("Not enough timestamps for a 70%/10%/20% split")

    train_values = _interpolate_within_split(wide.iloc[:train_end].to_numpy())
    val_values = _interpolate_within_split(wide.iloc[train_end:val_end].to_numpy())
    test_values = _interpolate_within_split(wide.iloc[val_end:].to_numpy())

    scaler = FlowScaler.fit(train_values)
    neighbor_indices = build_station_knn(station_metadata, knn_k) if knn_k else None
    normalized_adjacency = (
        build_normalized_adjacency(neighbor_indices) if neighbor_indices is not None else None
    )
    return PemsDatasetSplits(
        train=PemsFlowDataset(
            scaler.transform(train_values),
            wide.index[:train_end],
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            stride=stride,
        ),
        val=PemsFlowDataset(
            scaler.transform(val_values),
            wide.index[train_end:val_end],
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            stride=stride,
        ),
        test=PemsFlowDataset(
            scaler.transform(test_values),
            wide.index[val_end:],
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            stride=stride,
        ),
        scaler=scaler,
        station_ids=tuple(wide.columns.astype(str)),
        train_timestamps=wide.index[:train_end],
        val_timestamps=wide.index[train_end:val_end],
        test_timestamps=wide.index[val_end:],
        station_metadata=station_metadata,
        neighbor_indices=neighbor_indices,
        normalized_adjacency=normalized_adjacency,
    )
