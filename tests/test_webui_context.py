"""Ensemble, alignment, blocked futures, and walk-forward tests.

Network is mocked except tests/test_webui_binance_live.py.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui"))
sys.path.insert(0, str(ROOT))

from context import as_of_close, build_market_context, resample_complete_bins  # noqa: E402
from derivatives import FUTURES_HOST, FuturesClient  # noqa: E402
from ensemble import empirical_quantile, repair_ohlc, summarize_paths  # noqa: E402
from evaluate import checkpoint_origins, walk_forward  # noqa: E402
from app import create_app  # noqa: E402
from service import ModelInfo  # noqa: E402


def test_empirical_median_is_not_the_mean_and_repairs_ohlc():
    assert empirical_quantile([1, 2, 3, 100], 0.5) == 2.5
    assert empirical_quantile([1, 2, 3, 4], 0.5) == 2.5
    open_, high, low, close, changed = repair_ohlc(10, 9, 11, 10.5)
    assert changed
    assert high == 11 and low == 9
    assert low <= min(open_, close) <= max(open_, close) <= high


def test_summary_labels_dispersion_and_sample_fraction():
    def path(level):
        return [
            {"time": "2024-01-01T00:00:00Z", "open": level, "high": level + 1, "low": level - 1, "close": level, "volume": 1, "amount": 1},
            {"time": "2024-01-01T01:00:00Z", "open": level, "high": level + 1, "low": level - 1, "close": level + 1, "volume": 1, "amount": 1},
        ]

    summary = summarize_paths([path(10), path(12), path(20)], last_observed_close=11)
    assert summary["terminal"]["median_close"] == 13
    assert summary["terminal"]["mean_close"] == pytest.approx((11 + 13 + 21) / 3)
    assert summary["terminal"]["fraction_above"] == pytest.approx(2 / 3)
    assert summary["band"][0]["q50"] == 12
    assert summary["traces_shown"] == 3
    assert summary["median_path"][0]["high"] >= summary["median_path"][0]["close"]


def test_higher_timeframe_drops_bins_that_end_after_the_as_of():
    rows = []
    start = pd.Timestamp("2024-01-01T10:00:00Z")
    for i in range(19):
        stamp = start + pd.Timedelta(minutes=5 * i)
        price = 100 + i
        rows.append({"timestamps": stamp, "open": price, "high": price + 1, "low": price - 1, "close": price + 0.5, "volume": 10, "amount": 1000})
    frame = pd.DataFrame(rows)
    as_of = pd.Timestamp("2024-01-01T11:35:00Z")
    hourly = resample_complete_bins(frame, 3600, as_of)
    assert list(pd.to_datetime(hourly["close_time"], utc=True)) == [pd.Timestamp("2024-01-01T11:00:00Z")]
    assert hourly["close_time"].max() <= as_of


def test_futures_block_does_not_try_another_host_or_fill_values():
    urls = []

    class Opener:
        def open(self, request, timeout):
            urls.append(request.full_url)
            raise urllib.error.HTTPError(
                request.full_url, 451, "blocked", {}, io.BytesIO(b'{"code":0,"msg":"restricted location"}')
            )

    result = FuturesClient(opener=Opener(), clock=lambda: datetime(2026, 11, 6, tzinfo=timezone.utc)).context_for_spot(
        "BTCUSDT", "1h", pd.Timestamp("2026-11-06T18:00:00Z")
    )
    assert result["status"] == "unavailable"
    assert result["http_status"] == 451
    assert result["funding"] is None and result["open_interest"] is None and result["taker"] is None
    assert urls and all(url.startswith(FUTURES_HOST) for url in urls)
    assert len(urls) == 1
    assert "not filled in" in result["reason"]


def test_funding_after_the_as_of_is_not_treated_as_known_then():
    class Response:
        def __init__(self, payload):
            self.status = 200
            self.headers = {}
            self._raw = json.dumps(payload).encode()

        def read(self, _n=-1):
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def payload_for(url):
        if "exchangeInfo" in url:
            return {"symbols": [{"symbol": "BTCUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING"}]}
        if "fundingRate" in url:
            return [
                {"symbol": "BTCUSDT", "fundingTime": 1_700_000_000_000, "fundingRate": "0.0001", "markPrice": "100"},
                {"symbol": "BTCUSDT", "fundingTime": 1_800_000_000_000, "fundingRate": "0.0009", "markPrice": "200"},
            ]
        if url.endswith("openInterest") or "openInterest?" in url:
            return {"symbol": "BTCUSDT", "openInterest": "10", "time": 1_800_000_000_000}
        if "openInterestHist" in url:
            return [{"symbol": "BTCUSDT", "sumOpenInterest": "9", "sumOpenInterestValue": "900", "timestamp": 1_700_000_000_000}]
        if "takerlongshortRatio" in url:
            return [{"buyVol": "3", "sellVol": "1", "timestamp": 1_700_000_000_000}]
        raise AssertionError(url)

    class Opener:
        def open(self, request, timeout):
            return Response(payload_for(request.full_url))

    as_of = pd.Timestamp("2024-01-01T00:00:00Z")
    # 1_700_000_000_000 ms is 2023-11-14, 1_800_000_000_000 ms is 2027-01-15.
    result = FuturesClient(opener=Opener()).context_for_spot("BTCUSDT", "1h", as_of)
    assert result["status"] == "ok"
    assert result["funding"]["latest"]["funding_rate"] == pytest.approx(0.0001)
    assert all(row["funding_rate"] != 0.0009 for row in result["funding"]["records"])
    assert result["after_as_of"]["open_interest_contracts"] == pytest.approx(10)
    assert result["open_interest"]["latest"]["open_interest_contracts"] == pytest.approx(9)
    assert result["taker"]["latest"]["taker_buy_share"] == pytest.approx(0.75)


def test_csv_context_does_not_call_futures():
    class Boom:
        def open(self, request, timeout):
            raise AssertionError(request.full_url)

    app = create_app(futures=FuturesClient(opener=Boom()))
    client = app.test_client()
    response = client.post("/api/datasets/sample")
    assert response.status_code == 200
    body = response.get_json()
    assert body["context"]["derivatives"]["status"] == "not_applicable"
    assert body["context"]["derivatives"]["funding"] is None
    assert body["context"]["higher_timeframes"]
    assert all(item["source"].startswith("resampled") for item in body["context"]["higher_timeframes"] if item.get("status") == "ok")


def test_walk_forward_cannot_see_future_rows_and_reports_persistence():
    index = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    close = np.arange(80, dtype=float) + 10
    frame = pd.DataFrame({
        "timestamps": index,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.ones(80),
        "amount": np.ones(80),
    })
    seen = []

    def forecaster(history):
        seen.append(history["timestamps"].max())
        last = float(history["close"].iloc[-1])
        return {"median_closes": np.full(4, last), "sample_closes": np.vstack([np.full(4, last - 1), np.full(4, last + 5)])}

    report = walk_forward(frame, forecaster, lookback=32, pred_len=4, checkpoints=2)
    origins = checkpoint_origins(80, 32, 4, 2)
    for origin, max_seen in zip(origins, seen):
        assert max_seen < frame["timestamps"].iloc[origin]
    assert report["kronos_median_mae"] == report["persistence_mae"]
    assert report["empirical_coverage_q10_q90"] is not None
    assert "not a general accuracy claim" in report["comparison_note"]
    assert "cannot be ruled out" in report["training_cutoff"]


def _model(app):
    app.config["MODELS"].info = ModelInfo(
        key="kronos-base", name="Kronos-base", model_id="NeoQuasar/Kronos-base",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-base", params="102.3M", context_length=512,
        device="cpu", elapsed_seconds=0.1, loaded_at="2026-11-06T00:00:00Z",
    )


def test_cancel_stops_before_the_next_batch_and_concurrency_is_rejected():
    app = create_app(futures=FuturesClient(opener=type("B", (), {"open": lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))})()))
    gate = threading.Event()
    release = threading.Event()

    class Predictor:
        def __init__(self):
            self.calls = 0

        def predict(self, df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count, verbose):
            self.calls += 1
            gate.set()
            assert release.wait(3)
            count = len(y_timestamp)
            return pd.DataFrame({
                "open": np.ones(count), "high": np.full(count, 2.0), "low": np.full(count, 0.5),
                "close": np.ones(count), "volume": np.ones(count), "amount": np.ones(count),
            })

    _model(app)
    app.config["MODELS"].predictor = Predictor()
    app.config["MODELS"].sample_batch = 1
    client = app.test_client()
    loaded = client.post("/api/datasets/sample")
    dataset_id = loaded.get_json()["dataset"]["id"]
    started = client.post("/api/forecast", json={"dataset_id": dataset_id, "lookback": 32, "pred_len": 2, "sample_count": 2})
    assert started.status_code == 200
    job_id = started.get_json()["job_id"]
    assert gate.wait(3)
    blocked = client.post("/api/forecast", json={"dataset_id": dataset_id, "lookback": 32, "pred_len": 2, "sample_count": 1})
    assert blocked.status_code == 409
    cancelled = client.post(f"/api/forecast/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    release.set()
    final = None
    for _ in range(100):
        final = client.get(f"/api/forecast/jobs/{job_id}").get_json()
        if final["status"] != "running":
            break
        time.sleep(0.02)
    assert final["status"] == "cancelled"
    assert final["result"] is None
    assert app.config["MODELS"].predictor.calls == 1


def test_memory_fallback_retries_one_path_at_a_time_without_changing_model():
    app = create_app(futures=FuturesClient(opener=type("B", (), {"open": lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))})()))

    class Predictor:
        def sample_paths(self, df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count, verbose):
            if sample_count > 1:
                raise RuntimeError("CUDA out of memory")
            count = len(y_timestamp)
            return [pd.DataFrame({
                "open": np.ones(count), "high": np.full(count, 2.0), "low": np.full(count, 0.5),
                "close": np.full(count, 1.2), "volume": np.ones(count), "amount": np.ones(count),
            })]

    _model(app)
    app.config["MODELS"].predictor = Predictor()
    client = app.test_client()
    dataset_id = client.post("/api/datasets/sample").get_json()["dataset"]["id"]
    started = client.post("/api/forecast", json={"dataset_id": dataset_id, "lookback": 32, "pred_len": 2, "sample_count": 2})
    job_id = started.get_json()["job_id"]
    result = None
    for _ in range(100):
        payload = client.get(f"/api/forecast/jobs/{job_id}").get_json()
        if payload["status"] == "done":
            result = payload["result"]
            break
        if payload["status"] == "error":
            raise AssertionError(payload)
        time.sleep(0.02)
    assert result["model"]["name"] == "Kronos-base"
    assert "one at a time" in result["note"]
    assert "not changed" in result["note"]
