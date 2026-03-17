"""Load parsed xplt simulation data from CSV files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import validate
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _add_insertion_depth(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Convert time_step → insertion_depth if not already present."""
    if "insertion_depth" in df.columns:
        return df
    if "time_step" not in df.columns:
        log.warning("Neither 'insertion_depth' nor 'time_step' found — skipping conversion.")
        return df

    method = cfg["data"]["time_to_depth"]["method"]
    scale = float(cfg["data"]["time_to_depth"]["scale"])

    df = df.copy()
    if method == "linear":
        df["insertion_depth"] = df["time_step"] * scale
    else:
        raise ValueError(f"Unknown time_to_depth method: '{method}'")

    log.info("Derived insertion_depth from time_step (method=%s, scale=%.4f)", method, scale)
    return df


def load_simulation_data(
    cfg: dict,
    path: str | None = None,
) -> pd.DataFrame:
    """
    Load simulation CSV(s) and return a validated DataFrame.

    Args:
        cfg:  Pipeline config dict.
        path: Optional path to a single CSV file. If None, loads all files
              matching cfg['data']['file_pattern'] in cfg['data']['source'].
    """
    if path is not None:
        files = [Path(path)]
    else:
        source = Path(cfg["data"]["source"])
        pattern = cfg["data"]["file_pattern"]
        files = sorted(source.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {source.resolve()}"
            )

    frames: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df = _add_insertion_depth(df, cfg)
        frames.append(df)
        log.info("Loaded %-40s  rows=%d", f.name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    validate(combined, cfg)
    log.info("Dataset ready — %d rows, %d columns", *combined.shape)
    return combined
