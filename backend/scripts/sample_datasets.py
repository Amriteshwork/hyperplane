"""Build the sample datasets in datasets/<id>/data.csv.

Vendored datasets (clear licence) are committed; run with `--vendored` only to regenerate them.
Fetched datasets (licence unverified) are git-ignored and downloaded at build time.

Usage:
    uv run python -m scripts.sample_datasets            # fetched datasets only (default)
    uv run python -m scripts.sample_datasets --vendored # regenerate committed CSVs
    uv run python -m scripts.sample_datasets --all
"""

import argparse
import io
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skd

from app.core.config import get_settings

PENGUINS_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
)
SYNTHETIC_SEED = 42


@dataclass(frozen=True, slots=True)
class Builder:
    vendored: bool
    build: Callable[[], pd.DataFrame]


def _from_sklearn(loader: Callable[..., object], target: str) -> pd.DataFrame:
    bunch = loader(as_frame=True)
    frame: pd.DataFrame = bunch.frame  # type: ignore[attr-defined]
    return frame.rename(columns={"target": target})


def _iris() -> pd.DataFrame:
    df = _from_sklearn(skd.load_iris, "species")
    names = np.array(["setosa", "versicolor", "virginica"])
    df["species"] = names[df["species"].to_numpy()]
    return df


def _wine() -> pd.DataFrame:
    df = _from_sklearn(skd.load_wine, "cultivar")
    df["cultivar"] = "class_" + df["cultivar"].astype(str)
    return df


def _breast_cancer() -> pd.DataFrame:
    df = _from_sklearn(skd.load_breast_cancer, "diagnosis")
    # sklearn encodes 0 = malignant, 1 = benign.
    df["diagnosis"] = np.where(df["diagnosis"] == 1, "benign", "malignant")
    return df


def _penguins() -> pd.DataFrame:
    with urllib.request.urlopen(PENGUINS_URL, timeout=30) as response:
        return pd.read_csv(io.BytesIO(response.read()), na_values=["NA"])


def _synthetic_linear() -> pd.DataFrame:
    """y = 3 + 2·x1 − 1.5·x2 + 0.5·x3 + 0·x4 + segment offset + N(0, 1) noise."""
    rng = np.random.default_rng(SYNTHETIC_SEED)
    n = 500
    x = rng.normal(size=(n, 4))
    segment = rng.choice(["A", "B", "C"], size=n)
    offset = pd.Series(segment).map({"A": 0.0, "B": 1.0, "C": -1.0}).to_numpy()
    y = 3 + 2 * x[:, 0] - 1.5 * x[:, 1] + 0.5 * x[:, 2] + offset + rng.normal(size=n)
    df = pd.DataFrame(x.round(4), columns=["x1", "x2", "x3", "x4"])
    df["segment"] = segment
    df["y"] = y.round(4)
    return df


def _diabetes() -> pd.DataFrame:
    return _from_sklearn(skd.load_diabetes, "disease_progression")


def _california_housing() -> pd.DataFrame:
    return _from_sklearn(skd.fetch_california_housing, "MedHouseVal")


BUILDERS: dict[str, Builder] = {
    "iris": Builder(vendored=True, build=_iris),
    "wine": Builder(vendored=True, build=_wine),
    "breast-cancer": Builder(vendored=True, build=_breast_cancer),
    "penguins": Builder(vendored=True, build=_penguins),
    "synthetic-linear": Builder(vendored=True, build=_synthetic_linear),
    "diabetes": Builder(vendored=False, build=_diabetes),
    "california-housing": Builder(vendored=False, build=_california_housing),
}


def main(argv: list[str] | None = None) -> int:
    """Write the selected datasets; return the exit code."""
    parser = argparse.ArgumentParser(description="Build sample datasets.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--vendored", action="store_true", help="only the committed datasets")
    group.add_argument("--all", action="store_true", help="every dataset")
    parser.add_argument("--datasets-dir", type=Path, default=get_settings().datasets_dir)
    args = parser.parse_args(argv)

    for dataset_id, builder in BUILDERS.items():
        if not args.all and builder.vendored != args.vendored:
            continue
        folder: Path = args.datasets_dir / dataset_id
        folder.mkdir(parents=True, exist_ok=True)
        df = builder.build()
        df.to_csv(folder / "data.csv", index=False)
        print(f"{dataset_id}: {len(df)} rows × {df.shape[1]} columns → {folder / 'data.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
