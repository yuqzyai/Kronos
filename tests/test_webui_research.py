"""Research replay and the feature-parity surface of the local page.

The replay uses a stand-in forecaster. It does not download weights and it
does not claim a measured accuracy for Kronos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui"))
sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from research import (  # noqa: E402
    PREDECLARED_SHRINKAGE,
    _direction,
    adjust_with_prior_residuals,
    evaluate_methods,
    nonoverlapping_origins,
    shrink_toward_last,
)


PARITY_IDS = (
    "source-binance",
    "source-sample",
    "source-csv",
    "symbol",
    "interval",
    "load-binance",
    "refresh",
    "load-sample",
    "csv-file",
    "local-file",
    "model",
    "device",
    "lookback",
    "horizon",
    "load-model",
    "temperature",
    "top-p",
    "samples",
    "mode",
    "run",
    "cancel",
    "progress",
    "show-paths",
    "show-band",
    "fit-forecast",
    "fold-higher",
    "fold-perp",
    "fold-check",
    "fold-research",
    "run-check",
    "run-research",
    "export-record",
    "forecast-details",
    "scenario-share",
    "toggle-controls",
    "uitest-banner",
    "chart",
)


def _frame(n=80):
    close = np.linspace(100.0, 100.0 + n, n)
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamps": times,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.ones(n),
        "amount": np.ones(n),
    })


def test_windows_are_nonoverlapping_and_the_first_has_no_residual():
    assert PREDECLARED_SHRINKAGE == 0.25
    frame = _frame()
    seen = []

    def forecaster(history):
        assert len(history) == 20
        seen.append(pd.Timestamp(history["timestamps"].iloc[-1]))
        last = float(history["close"].iloc[-1])
        return {
            "median_closes": [last + 4.0] * 5,
            "mean_closes": [last + 1.0] * 5,
        }

    report = evaluate_methods(frame, forecaster, lookback=20, pred_len=5, max_checkpoints=3)
    origins = nonoverlapping_origins(len(frame), 20, 5, 3)
    assert origins == [20, 25, 30]
    assert len(seen) == 3
    for origin, stamp in zip(origins, seen):
        realized = pd.Timestamp(frame["timestamps"].iloc[origin])
        assert stamp < realized
    assert report["checkpoints"][0]["prior_residual_shift"] is None
    first_actual = frame["close"].iloc[20:25].to_numpy()
    first_last = float(frame["close"].iloc[19])
    expected_shift = float(np.mean(first_actual - (first_last + 4.0)))
    assert report["checkpoints"][1]["prior_residual_shift"] == expected_shift
    names = [row["name"] for row in report["methods"]]
    assert names == ["raw_median", "mean", "shrinkage", "prior_residual", "persistence"]
    assert report["recommended"] is None
    assert all(row["recommended"] is False for row in report["methods"])
    assert "not a recommendation" in report["selection_note"]
    assert "not a pristine out-of-sample" in report["out_of_sample_note"]
    assert "not a calibrated probability" in report["probability_note"]
    assert "does not replace" in report["chart_note"]


def test_one_draw_does_not_invent_a_separate_mean():
    frame = _frame(40)

    def forecaster(history):
        last = float(history["close"].iloc[-1])
        return {"median_closes": [last] * 4}

    report = evaluate_methods(frame, forecaster, lookback=8, pred_len=4, max_checkpoints=2)
    assert [row["name"] for row in report["methods"]] == [
        "raw_median",
        "shrinkage",
        "prior_residual",
        "persistence",
    ]
    assert "not scored" in report["mean_note"]


def test_shrinkage_weight_is_fixed_and_direction_is_explicit():
    shrunk = shrink_toward_last(np.array([110.0, 130.0]), 100.0)
    assert np.allclose(shrunk, [107.5, 122.5])
    adjusted, shift = adjust_with_prior_residuals(np.array([10.0, 12.0]), [])
    assert shift is None and np.allclose(adjusted, [10.0, 12.0])
    adjusted, shift = adjust_with_prior_residuals(np.array([10.0]), [np.array([2.0, 4.0])])
    assert shift == 3.0 and adjusted[0] == 13.0
    assert _direction(100, 110, 105) is True
    assert _direction(100, 110, 90) is False
    assert _direction(100, 100, 105) is False
    assert _direction(100, 100, 100) is True


def test_short_series_and_a_missing_dataset_are_errors():
    try:
        nonoverlapping_origins(10, 8, 4, 2)
    except ValueError as exc:
        assert "non-overlapping" in str(exc)
    else:
        raise AssertionError("A short series should be refused.")
    app = create_app()
    response = app.test_client().post("/api/research", json={"checkpoints": 3})
    assert response.status_code == 400
    assert "No dataset" in response.get_json()["error"]


def test_page_keeps_every_control_and_the_research_lab():
    page = (ROOT / "webui/templates/index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui/static/app.js").read_text(encoding="utf-8")
    style = (ROOT / "webui/static/app.css").read_text(encoding="utf-8")
    for control_id in PARITY_IDS:
        assert f'id="{control_id}"' in page
    assert 'data-layout="workspace"' in page
    assert 'data-layout-choice="focus"' in page
    assert 'data-layout-choice="workspace"' in page
    assert 'aria-controls="rail"' in page
    assert 'aria-pressed="false"' in page
    assert "Load official sample" in page
    assert "not a recommendation" in page
    assert "not live" in page.lower()
    assert "log-card" in page
    assert ".log-card { display: none" not in style
    assert 'if (name !== "focus" && name !== "workspace") return;' in script
    assert 'event.key === "p"' in script
    assert 'event.key === "Escape"' in script
    assert "/api/research" in script
    assert "/api/evaluate" in script
    assert "showPaths" in script
    assert "kronos-show-paths" in script
