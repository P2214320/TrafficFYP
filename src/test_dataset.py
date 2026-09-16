"""Smoke-test the PeMS dataset without reading the complete 1.17 GB CSV."""

from __future__ import annotations

from math import ceil
from pathlib import Path

try:
    from .dataset import FORECAST_STEPS, INPUT_STEPS, load_pems_datasets
except ImportError:  # Supports `python src/test_dataset.py`.
    from dataset import FORECAST_STEPS, INPUT_STEPS, load_pems_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "data_processed" / "pems_la.csv"
STATION_COUNT = 1_913


def main() -> None:
    # 5,000 raw CSV rows contain only about three 5-minute time steps, whereas
    # one training sample needs 432 steps.  618 time steps make the 70% training
    # split just large enough for one [288, 1913] -> [144, 1913] sample.
    minimum_time_steps = ceil((INPUT_STEPS + FORECAST_STEPS) / 0.70)
    raw_rows_for_smoke_test = minimum_time_steps * STATION_COUNT

    datasets = load_pems_datasets(
        CSV_PATH,
        max_raw_rows=raw_rows_for_smoke_test,
    )
    if not len(datasets.train):
        raise RuntimeError("Smoke-test subset did not produce a training window")

    x, y = datasets.train[0]
    print(f"Read at most {raw_rows_for_smoke_test:,} raw rows.")
    print(f"Train samples: {len(datasets.train)}")
    print(f"X shape: {x.shape}, dtype: {x.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")


if __name__ == "__main__":
    main()
