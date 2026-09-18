"""Simple NumPy batch loaders for :mod:`dataset` without a PyTorch dependency."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from .dataset import PemsDatasetSplits, PemsFlowDataset, load_pems_datasets
except ImportError:  # Supports `python src/dataloader.py` style imports.
    from dataset import PemsDatasetSplits, PemsFlowDataset, load_pems_datasets


class NumpyDataLoader:
    """Yield flow and hour-of-week batches with a shared leading batch axis."""

    def __init__(
        self,
        dataset: PemsFlowDataset,
        *,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            epoch_seed = None if self.seed is None else self.seed + self._epoch
            np.random.default_rng(epoch_seed).shuffle(indices)
            self._epoch += 1

        stop = len(indices)
        if self.drop_last:
            stop -= stop % self.batch_size
        for start in range(0, stop, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            samples = [self.dataset[int(index)] for index in batch_indices]
            yield tuple(
                np.stack([sample[position] for sample in samples]).astype(dtype, copy=False)
                for position, dtype in enumerate((np.float32, np.float32, np.int64, np.int64))
            )


@dataclass(frozen=True)
class PemsDataLoaders:
    train: NumpyDataLoader
    val: NumpyDataLoader
    test: NumpyDataLoader
    datasets: PemsDatasetSplits


def create_pems_dataloaders(
    csv_path: str | Path,
    *,
    batch_size: int = 32,
    chunksize: int = 500_000,
    max_raw_rows: int | None = None,
    station_limit: int | None = None,
    seed: int | None = None,
) -> PemsDataLoaders:
    """Create chronological validation/test and shuffled training batch loaders."""
    datasets = load_pems_datasets(
        csv_path,
        chunksize=chunksize,
        max_raw_rows=max_raw_rows,
        station_limit=station_limit,
    )
    return PemsDataLoaders(
        train=NumpyDataLoader(datasets.train, batch_size=batch_size, shuffle=True, seed=seed),
        val=NumpyDataLoader(datasets.val, batch_size=batch_size),
        test=NumpyDataLoader(datasets.test, batch_size=batch_size),
        datasets=datasets,
    )
