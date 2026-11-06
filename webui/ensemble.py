"""Summaries of independent Kronos sample paths.

The median and the quantile band describe the scatter of the model's own
draws. They are not a calibrated confidence interval and they are not a
probability that price will move.
"""

from __future__ import annotations

import math

import numpy as np


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Sample values must be finite numbers.")
    return array


def empirical_quantile(values: list[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("Quantile must be between 0 and 1.")
    return float(np.quantile(_finite(values), q, method="linear"))


def repair_ohlc(open_, high, low, close) -> tuple[float, float, float, float, bool]:
    """Make one bar drawable. Returns whether high or low had to be expanded."""

    repaired_high = max(open_, high, low, close)
    repaired_low = min(open_, high, low, close)
    changed = repaired_high != high or repaired_low != low
    return open_, repaired_high, repaired_low, close, changed


def _component(paths: list[list[dict]], key: str, index: int) -> list[float]:
    return [path[index][key] for path in paths]


def summarize_paths(paths: list[list[dict]], last_observed_close: float, trace_limit: int = 8) -> dict:
    if not paths or not paths[0]:
        raise ValueError("At least one sampled path is required.")
    width = len(paths[0])
    if any(len(path) != width for path in paths):
        raise ValueError("Every sample path must have the same horizon.")
    repairs = 0
    median_path = []
    mean_path = []
    band = []
    for index in range(width):
        row = {"time": paths[0][index]["time"]}
        mean_row = {"time": row["time"]}
        for key in ("open", "high", "low", "close", "volume", "amount"):
            present = [path[index].get(key) for path in paths]
            if any(value is None for value in present):
                continue
            values = _finite(present)
            row[key] = empirical_quantile(values.tolist(), 0.5)
            mean_row[key] = float(np.mean(values))
        open_, high, low, close, changed = repair_ohlc(row["open"], row["high"], row["low"], row["close"])
        row["open"], row["high"], row["low"], row["close"] = open_, high, low, close
        if changed:
            repairs += 1
        mean_open, mean_high, mean_low, mean_close, mean_changed = repair_ohlc(
            mean_row["open"], mean_row["high"], mean_row["low"], mean_row["close"]
        )
        mean_row["open"], mean_row["high"], mean_row["low"], mean_row["close"] = mean_open, mean_high, mean_low, mean_close
        if mean_changed:
            repairs += 1
        closes = _finite(_component(paths, "close", index)).tolist()
        band.append({
            "time": row["time"],
            "q10": empirical_quantile(closes, 0.10),
            "q25": empirical_quantile(closes, 0.25),
            "q50": empirical_quantile(closes, 0.50),
            "q75": empirical_quantile(closes, 0.75),
            "q90": empirical_quantile(closes, 0.90),
            "mean": float(np.mean(closes)),
        })
        median_path.append(row)
        mean_path.append(mean_row)
    terminals = _finite([path[-1]["close"] for path in paths]).tolist()
    above = sum(1 for value in terminals if value > last_observed_close)
    below = sum(1 for value in terminals if value < last_observed_close)
    equal = len(terminals) - above - below
    count = len(paths)
    return {
        "median_path": median_path,
        "mean_path": mean_path if count > 1 else None,
        "band": band if count > 1 else None,
        "traces": _pick_traces(paths, trace_limit) if count > 1 else None,
        "traces_shown": 0 if count <= 1 else min(count, trace_limit),
        "traces_total": count,
        "ohlc_repairs": repairs,
        "terminal": {
            "median_close": empirical_quantile(terminals, 0.5),
            "mean_close": float(np.mean(terminals)),
            "q10": empirical_quantile(terminals, 0.10),
            "q90": empirical_quantile(terminals, 0.90),
            "fraction_above": above / count,
            "fraction_below": below / count,
            "fraction_equal": equal / count,
            "above": above,
            "below": below,
            "equal": equal,
            "samples": count,
        },
        "scenario": scenario_share(above, below, equal, ""),
    }


def scenario_share(above: int, below: int, equal: int, horizon: str) -> dict:
    """Share of finished model draws. Not a calibrated market probability."""

    for name, value in (("above", above), ("below", below), ("equal", equal)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative whole number.")
    total = above + below + equal
    if total < 1:
        raise ValueError("Need at least one scenario.")
    step = 100 / total
    percent = above / total * 100
    if total == 1:
        text = (
            "One model scenario. A percentage of draws is not meaningful until there is more than one path. "
            "This single draw is not the chance of a profit."
        )
    else:
        ending = f"{above} of {total} model scenarios end above the last close, {below} end below"
        if equal:
            ending += f", and {equal} end unchanged"
        ending += "."
        scope = f" over {horizon}" if horizon else ""
        text = (
            f"{ending} That is {percent:.1f}% of these {total} draws{scope}. "
            f"With {total} draws the share moves in steps of {step:.1f} percentage points. "
            "These draws repeat one model. They are not independent opinions and not the chance of a profit."
        )
    return {
        "above": above,
        "below": below,
        "equal": equal,
        "total": total,
        "above_share_percent": percent,
        "step_percentage_points": step,
        "horizon": horizon,
        "small_sample": total < 30,
        "text": text,
    }


def _pick_traces(paths: list[list[dict]], limit: int) -> list[list[dict]]:
    count = len(paths)
    if count <= limit:
        chosen = paths
    else:
        indexes = [int(round(i * (count - 1) / (limit - 1))) for i in range(limit)]
        chosen = [paths[index] for index in indexes]
    traces = []
    for path in chosen:
        traces.append([{"time": row["time"], "close": float(row["close"])} for row in path])
    return traces


def is_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "outofmemory" in text or text.strip() == "oom"
