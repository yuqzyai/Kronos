"""A short walk-forward check against a persistence baseline.

The checkpoint count is small on purpose. A lower error on those rows is not
a general claim that the model, the median, or any overlay is more accurate.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


TRAINING_CUTOFF_NOTE = (
    "The public checkpoint's training end is not verified in this app, "
    "so overlap between this file and the training period cannot be ruled out."
)


def checkpoint_origins(n_rows: int, lookback: int, pred_len: int, checkpoints: int) -> list[int]:
    """Exclusive ends of the history window. Realized rows are strictly after each end."""

    if checkpoints < 1:
        raise ValueError("Ask for at least one checkpoint.")
    if checkpoints > 3:
        raise ValueError("This check runs at most 3 checkpoints.")
    first = lookback
    last = n_rows - pred_len
    if last < first:
        raise ValueError(
            f"Need at least {lookback + pred_len} rows for {checkpoints} checkpoint(s). This series has {n_rows}."
        )
    if checkpoints == 1:
        return [last]
    span = last - first
    return [first + int(round(i * span / (checkpoints - 1))) for i in range(checkpoints)]


def persistence_closes(last_close: float, pred_len: int) -> np.ndarray:
    return np.full(pred_len, float(last_close), dtype=float)


def walk_forward(frame: pd.DataFrame, forecaster, lookback: int, pred_len: int, checkpoints: int) -> dict:
    origins = checkpoint_origins(len(frame), lookback, pred_len, checkpoints)
    rows = []
    kronos_errors = []
    baseline_errors = []
    covered = []
    coverage_possible = False
    for origin in origins:
        history = frame.iloc[origin - lookback:origin].reset_index(drop=True)
        realized = frame.iloc[origin:origin + pred_len].reset_index(drop=True)
        if history["timestamps"].max() >= realized["timestamps"].min():
            raise ValueError("A checkpoint history was not strictly before its realized candles.")
        forecast = forecaster(history.copy())
        median = np.asarray(forecast["median_closes"], dtype=float)
        if len(median) != pred_len:
            raise ValueError("The forecast horizon did not match the realized window.")
        actual = realized["close"].astype(float).to_numpy()
        baseline = persistence_closes(float(history["close"].iloc[-1]), pred_len)
        step_kronos = np.abs(median - actual)
        step_base = np.abs(baseline - actual)
        kronos_errors.extend(step_kronos.tolist())
        baseline_errors.extend(step_base.tolist())
        samples = forecast.get("sample_closes")
        inside = None
        if samples is not None and len(samples):
            coverage_possible = True
            matrix = np.asarray(samples, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != pred_len:
                raise ValueError("Sample closes must be (samples, horizon).")
            low = np.quantile(matrix, 0.10, axis=0, method="linear")
            high = np.quantile(matrix, 0.90, axis=0, method="linear")
            inside = int(np.sum((actual >= low) & (actual <= high)))
            covered.append(inside / pred_len)
        rows.append({
            "history_last_open": _iso(history["timestamps"].iloc[-1]),
            "realized_first_open": _iso(realized["timestamps"].iloc[0]),
            "realized_last_open": _iso(realized["timestamps"].iloc[-1]),
            "kronos_median_mae": float(np.mean(step_kronos)),
            "persistence_mae": float(np.mean(step_base)),
            "empirical_coverage_q10_q90": None if inside is None else inside / pred_len,
        })
    coverage = float(np.mean(covered)) if covered else None
    return {
        "issued_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoints": rows,
        "kronos_median_mae": float(np.mean(kronos_errors)),
        "persistence_mae": float(np.mean(baseline_errors)),
        "empirical_coverage_q10_q90": coverage,
        "coverage_note": (
            None if not coverage_possible else
            "Share of evaluated closes that landed inside the 10th to 90th percentile of the samples at that step. "
            "This is model sample dispersion on these rows, not a calibrated probability."
        ),
        "training_cutoff": TRAINING_CUTOFF_NOTE,
        "comparison_note": (
            "Errors compare the Kronos median close with a persistence forecast that repeats the last observed close. "
            "Higher-timeframe readings and perpetual data are not inputs, so they cannot change these errors. "
            "A lower number on these checkpoints is not a general accuracy claim."
        ),
    }


def forecast_record(forecast: dict, dataset_meta: dict, context: dict | None) -> dict:
    terminal = forecast.get("terminal") or {}
    band = forecast.get("band") or []
    return {
        "issued_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "forecast_record",
        "source_as_of": dataset_meta.get("as_of"),
        "fetched_at": dataset_meta.get("fetched_at"),
        "source": dataset_meta.get("source"),
        "symbol": dataset_meta.get("symbol"),
        "interval": dataset_meta.get("interval") or dataset_meta.get("interval_label"),
        "host": dataset_meta.get("host"),
        "live": dataset_meta.get("live"),
        "model": forecast.get("model"),
        "settings": {
            "lookback": forecast.get("lookback"),
            "pred_len": forecast.get("pred_len"),
            "sample_count": forecast.get("sample_count"),
            "temperature": forecast.get("temperature"),
            "top_p": forecast.get("top_p"),
            "mode": forecast.get("mode"),
            "aggregation": forecast.get("aggregation"),
            "batch_size": forecast.get("batch_size"),
        },
        "origin": forecast.get("input"),
        "forecast_opens_at": forecast.get("forecast_opens_at"),
        "terminal": terminal,
        "path_summary": [
            {
                "time": row.get("time"),
                "median_close": row.get("q50", row.get("close")),
                "mean_close": row.get("mean"),
                "q10": row.get("q10"),
                "q90": row.get("q90"),
            }
            for row in (band or forecast.get("path") or [])
        ],
        "context_status": {
            "higher_timeframes": [
                {"interval": item.get("interval"), "status": item.get("status"), "as_of": item.get("as_of")}
                for item in (context or {}).get("higher_timeframes", [])
            ],
            "derivatives": (context or {}).get("derivatives", {}).get("status"),
            "derivatives_reason": (context or {}).get("derivatives", {}).get("reason"),
        },
        "realized_later": None,
        "training_cutoff": TRAINING_CUTOFF_NOTE,
        "note": (
            "Realized candles are left empty. Fill them later only with bars whose open time is after forecast_opens_at. "
            "Context fields were not model inputs."
        ),
    }


def _iso(value) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
