"""Opt-in replay of a few predeclared forecast variants.

Nothing here replaces the raw Kronos path on the chart. The shrinkage weight
is fixed in this file before any series is seen. Residual adjustment uses only
earlier windows. The public weights' training end is unknown, so a replay on
historical rows is not a pristine out-of-sample test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluate import TRAINING_CUTOFF_NOTE, persistence_closes

# Weight on the last observed close. The rest stays with the model path.
# This number is not fitted on the windows being scored.
PREDECLARED_SHRINKAGE = 0.25
MAX_CHECKPOINTS = 4


def nonoverlapping_origins(n_rows: int, lookback: int, pred_len: int, max_checkpoints: int = MAX_CHECKPOINTS) -> list[int]:
    """History ends whose scored future windows do not overlap.

    A later window may use the earlier future as history, because those rows
    are then in the past. The scored slices themselves stay disjoint.
    """

    if max_checkpoints < 2 or max_checkpoints > MAX_CHECKPOINTS:
        raise ValueError(f"Ask for 2 to {MAX_CHECKPOINTS} windows.")
    if lookback < 1 or pred_len < 1:
        raise ValueError("Context and horizon must be at least 1.")
    origins = []
    origin = lookback
    while origin + pred_len <= n_rows and len(origins) < max_checkpoints:
        origins.append(origin)
        origin += pred_len
    if len(origins) < 2:
        raise ValueError(
            f"Need at least {lookback + 2 * pred_len} rows for two non-overlapping windows. This series has {n_rows}."
        )
    for earlier, later in zip(origins, origins[1:]):
        if earlier + pred_len > later:
            raise ValueError("Windows overlap.")
    return origins


def shrink_toward_last(path: np.ndarray, last_close: float, weight: float = PREDECLARED_SHRINKAGE) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Shrinkage weight must be between 0 and 1.")
    values = np.asarray(path, dtype=float)
    return (1.0 - weight) * values + weight * float(last_close)


def adjust_with_prior_residuals(path: np.ndarray, prior_residuals: list[np.ndarray]) -> tuple[np.ndarray, float | None]:
    """Shift a path by the mean residual of earlier windows only.

    An empty prior means no shift. The current window is not included.
    """

    values = np.asarray(path, dtype=float)
    if not prior_residuals:
        return values.copy(), None
    stacked = np.concatenate([np.asarray(item, dtype=float).ravel() for item in prior_residuals])
    shift = float(np.mean(stacked))
    return values + shift, shift


def _direction(last_close: float, forecast_terminal: float, realized_terminal: float) -> bool | None:
    """Whether the forecast and the later close moved the same way from the last close.

    A flat forecast or a flat outcome is not counted as a match. Both flat is a match.
    """

    forecast_change = float(forecast_terminal) - float(last_close)
    realized_change = float(realized_terminal) - float(last_close)
    if forecast_change == 0.0 and realized_change == 0.0:
        return True
    if forecast_change == 0.0 or realized_change == 0.0:
        return False
    return (forecast_change > 0.0) == (realized_change > 0.0)


def _mae(forecast: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(forecast, dtype=float) - np.asarray(actual, dtype=float))))


def evaluate_methods(frame: pd.DataFrame, forecaster, lookback: int, pred_len: int, max_checkpoints: int = MAX_CHECKPOINTS) -> dict:
    origins = nonoverlapping_origins(len(frame), lookback, pred_len, max_checkpoints)
    names = ("raw_median", "shrinkage", "prior_residual", "persistence")
    errors = {name: [] for name in names}
    directions = {name: [] for name in names}
    mean_errors = []
    mean_directions = []
    mean_was_distinct = False
    windows = []
    prior_residuals: list[np.ndarray] = []
    for origin in origins:
        history = frame.iloc[origin - lookback:origin].reset_index(drop=True)
        realized = frame.iloc[origin:origin + pred_len].reset_index(drop=True)
        if history["timestamps"].max() >= realized["timestamps"].min():
            raise ValueError("A research window saw a row that was not strictly in the past.")
        produced = forecaster(history.copy())
        median = np.asarray(produced["median_closes"], dtype=float)
        if len(median) != pred_len:
            raise ValueError("The model path length did not match the horizon.")
        last_close = float(history["close"].iloc[-1])
        actual = realized["close"].astype(float).to_numpy()
        shrunk = shrink_toward_last(median, last_close)
        adjusted, shift = adjust_with_prior_residuals(median, prior_residuals)
        persistence = persistence_closes(last_close, pred_len)
        methods = {
            "raw_median": median,
            "shrinkage": shrunk,
            "prior_residual": adjusted,
            "persistence": persistence,
        }
        window = {
            "history_last_open": _iso(history["timestamps"].iloc[-1]),
            "realized_first_open": _iso(realized["timestamps"].iloc[0]),
            "prior_residual_shift": shift,
            "errors": {},
        }
        for name, values in methods.items():
            step_error = _mae(values, actual)
            errors[name].append(step_error)
            directions[name].append(_direction(last_close, float(values[-1]), float(actual[-1])))
            window["errors"][name] = step_error
        if "mean_closes" in produced and produced["mean_closes"] is not None:
            mean_path = np.asarray(produced["mean_closes"], dtype=float)
            if len(mean_path) == pred_len and not np.allclose(mean_path, median):
                mean_was_distinct = True
                mean_errors.append(_mae(mean_path, actual))
                mean_directions.append(_direction(last_close, float(mean_path[-1]), float(actual[-1])))
                window["errors"]["mean"] = mean_errors[-1]
        windows.append(window)
        prior_residuals.append(actual - median)
    rows = [_method_row(name, errors[name], directions[name]) for name in names]
    if mean_was_distinct:
        rows.insert(1, _method_row("mean", mean_errors, mean_directions))
    return {
        "issued_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "windows": len(origins),
        "lookback": lookback,
        "pred_len": pred_len,
        "shrinkage_weight_on_last_close": PREDECLARED_SHRINKAGE,
        "shrinkage_note": (
            f"{PREDECLARED_SHRINKAGE:.0%} of each raw close is replaced by the last observed close. "
            "That weight is written in the program. It was not chosen by looking at these windows."
        ),
        "residual_note": (
            "The residual shift for a window is the average of (later close minus raw path) on earlier windows only. "
            "The first window has no earlier window, so it is not shifted."
        ),
        "direction_definition": (
            "A window counts as the same direction when the path's last close and the later close "
            "are both above, or both below, the last close before the window. A flat path or a flat outcome does not count."
        ),
        "methods": rows,
        "checkpoints": windows,
        "recommended": None,
        "selection_note": "No method is selected from this replay. A lower error here is not a recommendation and not a general result.",
        "mean_note": (
            "The mean path differed from the raw median and was scored on the same windows."
            if mean_was_distinct else
            "Mean was not scored as its own method. One draw has no mean distinct from itself, and this replay does not invent one."
        ),
        "training_cutoff": TRAINING_CUTOFF_NOTE,
        "out_of_sample_note": (
            "This is a historical replay. Because the training end of the public weights is unknown, "
            "it is not a pristine out-of-sample test."
        ),
        "probability_note": (
            "Direction counts are k of N windows. They are not a calibrated probability, "
            "and the scenario share from one forecast's paths is a different thing: that share is how those draws ended, not how often the model has been right."
        ),
        "chart_note": "The chart still shows the raw Kronos path. This replay does not replace it.",
    }


def _method_row(name: str, step_errors: list[float], direction_flags: list[bool | None]) -> dict:
    counted = [flag for flag in direction_flags if flag is not None]
    same = sum(1 for flag in counted if flag)
    return {
        "name": name,
        "close_mae": float(np.mean(step_errors)) if step_errors else None,
        "direction_same": same,
        "direction_windows": len(counted),
        "recommended": False,
    }


def _iso(value) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
