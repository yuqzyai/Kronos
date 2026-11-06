"""Web UI tests that do not download weights or call Binance.

The live Binance check is tests/test_webui_binance_live.py.
"""

from __future__ import annotations

import io
import json
import sys
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

from app import create_app  # noqa: E402
from market import (  # noqa: E402
    PRIMARY_HOST,
    SECONDARY_HOST,
    BinanceClient,
    MarketDataError,
    describe_horizon,
    load_csv_frame,
    parse_kline_row,
    split_closed_candles,
)
from service import (  # noqa: E402
    ModelInfo,
    future_timestamps,
    load_official_sample,
    server_options,
)


def _kline(open_ms, close_ms, price=100.0):
    return [
        open_ms,
        f"{price:.2f}",
        f"{price + 2:.2f}",
        f"{price - 2:.2f}",
        f"{price + 0.5:.2f}",
        "12.5",
        close_ms,
        "1300.25",
        8,
        "1",
        "1",
        "0",
    ]


class _Headers(dict):
    pass


class _Body:
    def __init__(self, payload, status=200, headers=None):
        self.status = status
        self.headers = _Headers(headers or {"X-MBX-USED-WEIGHT-1M": "4"})
        self._raw = json.dumps(payload).encode()

    def read(self, _n=-1):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Opener:
    def __init__(self, handler):
        self.handler = handler
        self.urls = []

    def open(self, request, timeout):
        self.urls.append(request.full_url)
        assert timeout == 10
        result = self.handler(request)
        if isinstance(result, Exception):
            raise result
        return result


def _clock():
    class Clock:
        def __init__(self):
            self.now = datetime(2026, 11, 6, 20, 30, tzinfo=timezone.utc)
            self.mono = 1_000.0

        def clock(self):
            return self.now

        def monotonic(self):
            self.mono += 0.01
            return self.mono

    return Clock()


def _client(handler):
    clock = _clock()
    opener = _Opener(handler)
    client = BinanceClient(opener=opener, clock=clock.clock, monotonic=clock.monotonic)
    return client, opener, clock


def _spot_then_klines(rows):
    def handler(request):
        if "/api/v3/exchangeInfo" in request.full_url:
            return _Body({"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "isSpotTradingAllowed": True}]})
        if "/api/v3/klines" in request.full_url:
            return _Body(rows)
        raise AssertionError(request.full_url)

    return handler


def test_kline_maps_quote_volume_to_amount_and_splits_the_open_candle():
    now = datetime(2026, 11, 6, 20, 30, tzinfo=timezone.utc)
    closed_row = _kline(1_762_459_200_000, 1_762_462_799_999, 100)
    forming_row = _kline(1_762_462_800_000, 1_762_466_399_999, 105)
    # The absolute times above are only ordered relative to `now` after we rebuild them.
    open_closed = int(datetime(2026, 11, 6, 19, 0, tzinfo=timezone.utc).timestamp() * 1000)
    close_closed = int(datetime(2026, 11, 6, 19, 59, 59, 999000, tzinfo=timezone.utc).timestamp() * 1000)
    open_forming = int(datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc).timestamp() * 1000)
    close_forming = int(datetime(2026, 11, 6, 20, 59, 59, 999000, tzinfo=timezone.utc).timestamp() * 1000)
    closed, forming, _warnings = split_closed_candles(
        [_kline(open_closed, close_closed, 100), _kline(open_forming, close_forming, 105)],
        now,
    )
    assert len(closed) == 1
    assert closed[0]["amount"] == 1300.25
    assert closed[0]["volume"] == 12.5
    assert forming["open"] == 105
    parsed = parse_kline_row(closed_row, 0)
    assert parsed["amount"] == 1300.25


def test_binance_client_uses_only_the_market_data_host_and_caches():
    open_closed = int(datetime(2026, 11, 6, 19, 0, tzinfo=timezone.utc).timestamp() * 1000)
    close_closed = int(datetime(2026, 11, 6, 19, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)
    open_forming = int(datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc).timestamp() * 1000)
    close_forming = int(datetime(2026, 11, 6, 20, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [_kline(open_closed, close_closed, 100), _kline(open_forming, close_forming, 101)]
    client, opener, _clock = _client(_spot_then_klines(rows))
    series = client.klines("btcusdt", "1h", 1)
    assert series.symbol == "BTCUSDT"
    assert series.live is True
    assert series.host == "data-api.binance.vision"
    assert len(series.frame) == 1
    assert series.forming is not None
    assert series.forming["closed"] is False
    assert list(series.frame.columns)[:6] == ["timestamps", "open", "high", "low", "close", "volume"]
    assert "amount" in series.frame.columns
    assert str(series.frame["timestamps"].dt.tz) == "UTC"
    again = client.klines("BTCUSDT", "1h", 1)
    assert again.from_cache is True
    assert len(opener.urls) == 2
    assert all(url.startswith(PRIMARY_HOST) for url in opener.urls)
    assert all("/api/v3/account" not in url for url in opener.urls)


def test_http_451_does_not_try_another_host_or_invent_candles():
    def handler(request):
        raise urllib.error.HTTPError(
            request.full_url,
            451,
            "Unavailable For Legal Reasons",
            _Headers(),
            io.BytesIO(b'{"code":0,"msg":"restricted location"}'),
        )

    client, opener, _clock = _client(handler)
    with pytest.raises(MarketDataError, match="451"):
        client.klines("BTCUSDT", "1h", 10)
    assert opener.urls
    assert all(url.startswith(PRIMARY_HOST) for url in opener.urls)
    assert not any(url.startswith(SECONDARY_HOST) for url in opener.urls)


def test_transport_failure_may_try_the_documented_secondary_once():
    def handler(request):
        if request.full_url.startswith(PRIMARY_HOST):
            raise urllib.error.URLError("timed out")
        if request.full_url.startswith(SECONDARY_HOST):
            raise urllib.error.HTTPError(
                request.full_url,
                451,
                "blocked",
                _Headers(),
                io.BytesIO(b'{"msg":"restricted"}'),
            )
        raise AssertionError(request.full_url)

    client, opener, _clock = _client(handler)
    with pytest.raises(MarketDataError, match="No further hosts"):
        client.klines("BTCUSDT", "1h", 10)
    assert any(url.startswith(SECONDARY_HOST) for url in opener.urls)


def test_symbol_and_interval_are_rejected_without_a_request():
    client, opener, _clock = _client(lambda request: (_ for _ in ()).throw(AssertionError("network")))
    with pytest.raises(MarketDataError, match="Symbol"):
        client.klines("BTC/USDT", "1h", 10)
    with pytest.raises(MarketDataError, match="Interval"):
        client.klines("BTCUSDT", "2m", 10)
    with pytest.raises(MarketDataError, match="allowlist"):
        client._once("https://evil.example", "/api/v3/klines", {"symbol": "BTCUSDT"})
    with pytest.raises(MarketDataError, match="path"):
        client._once(PRIMARY_HOST, "/api/v3/order", {"symbol": "BTCUSDT"})
    assert opener.urls == []


def test_rate_limit_keeps_a_labeled_cache():
    open_closed = int(datetime(2026, 11, 6, 19, 0, tzinfo=timezone.utc).timestamp() * 1000)
    close_closed = int(datetime(2026, 11, 6, 19, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [_kline(open_closed, close_closed, 100)]
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            if "exchangeInfo" in request.full_url:
                return _Body({"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "isSpotTradingAllowed": True}]})
            return _Body(rows)
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "rate",
            _Headers({"Retry-After": "30"}),
            io.BytesIO(b'{"msg":"Too much request weight"}'),
        )

    client, _opener, clock = _client(handler)
    first = client.klines("BTCUSDT", "1h", 1)
    assert first.from_cache is False
    clock.mono += 400
    cached = client.klines("BTCUSDT", "1h", 1, force=True)
    assert cached.from_cache is True
    assert any("rate limit" in warning.lower() for warning in cached.warnings)


def test_csv_requires_a_timestamp_and_rejects_broken_ohlc():
    with pytest.raises(MarketDataError, match="timestamp"):
        load_csv_frame("open,high,low,close\n1,2,0.5,1.5\n", "prices.csv")
    with pytest.raises(MarketDataError, match="high/low"):
        load_csv_frame(
            "timestamps,open,high,low,close\n2024-01-01 00:00:00,10,9,8,9.5\n2024-01-01 01:00:00,10,11,9,10\n",
            "bad.csv",
        )


def test_official_sample_is_marked_not_live_and_has_amount():
    series, label = load_official_sample()
    assert series.live is False
    assert series.source == "sample"
    assert "Not live" in series.note
    assert "amount" in series.frame.columns
    assert len(series.frame) > 400
    assert series.step_seconds == 300
    assert label.startswith("Official")


def test_forward_timestamps_start_after_the_input_open():
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    series = load_official_sample()[0]
    series.step_seconds = 3600
    series.step_kind = "timedelta"
    series.interval = "1h"
    future = future_timestamps(index[-1], series, 4)
    assert future[0] == index[-1] + pd.Timedelta(hours=1)
    assert list(future[1:] > future[:-1]) == [True, True, True]
    assert describe_horizon(24, "1 hour", 3600) == "24\u00a0candles · 1\u00a0hour\u00a0each · 24\u00a0hours"


class _FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count, verbose):
        assert sample_count == 1
        self.calls.append(pred_len)
        count = len(y_timestamp)
        base = 10 + len(self.calls)
        return pd.DataFrame(
            {
                "open": np.full(count, base),
                "high": np.full(count, base + 1),
                "low": np.full(count, base - 1),
                "close": np.linspace(base, base + 0.5, count),
                "volume": np.ones(count),
                "amount": np.full(count, 20),
            }
        )


def _hours(count=48):
    lines = ["timestamps,open,high,low,close,volume,amount"]
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(count):
        price = 100 + i * 0.1
        stamp = (start + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{stamp},{price},{price+1},{price-1},{price+0.2},5,500")
    return "\n".join(lines).encode()


def _ready_app():
    app = create_app()
    app.config["TESTING"] = True
    info = ModelInfo(
        key="kronos-base",
        name="Kronos-base",
        model_id="NeoQuasar/Kronos-base",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        params="102.3M",
        context_length=512,
        device="cpu",
        elapsed_seconds=0.1,
        loaded_at="2026-11-06T00:00:00Z",
    )
    service = app.config["MODELS"]
    service.predictor = _FakePredictor()
    service.info = info
    return app


def test_server_stays_on_localhost_with_the_debugger_off():
    options = server_options()
    assert options["host"] == "127.0.0.1"
    assert options["debug"] is False
    assert options["use_reloader"] is False


def _wait_job(client, job_id):
    for _ in range(100):
        payload = client.get(f"/api/forecast/jobs/{job_id}").get_json()
        if payload["status"] == "done":
            return payload["result"]
        if payload["status"] in {"error", "cancelled"}:
            raise AssertionError(payload)
        time.sleep(0.02)
    raise AssertionError("forecast job did not finish")


def test_forecast_requires_a_loaded_dataset_and_one_sample_is_one_path():
    app = _ready_app()
    client = app.test_client()
    missing = client.post("/api/forecast", json={"dataset_id": "none", "lookback": 32, "pred_len": 4})
    assert missing.status_code == 400
    assert "Load Binance candles, the official sample, or a CSV file first" in missing.get_json()["error"]

    uploaded = client.post("/api/datasets/csv", data={"file": (io.BytesIO(_hours()), "desk.csv")})
    assert uploaded.status_code == 200
    body = uploaded.get_json()
    assert body["dataset"]["live"] is False
    assert body["dataset"]["closed_count"] == 48
    dataset_id = body["dataset"]["id"]

    single = client.post(
        "/api/forecast",
        json={"dataset_id": dataset_id, "lookback": 32, "pred_len": 4, "sample_count": 1, "mode": "forward"},
    )
    assert single.status_code == 200
    forecast = _wait_job(client, single.get_json()["job_id"])
    assert forecast["aggregation"] == "single_sample"
    assert forecast["sample_closes"] is None
    assert len(forecast["path"]) == 4
    assert forecast["forecast_opens_at"] > forecast["boundary_open"]
    assert app.config["MODELS"].predictor.calls == [4]

    many = client.post(
        "/api/forecast",
        json={"dataset_id": dataset_id, "lookback": 32, "pred_len": 4, "sample_count": 3, "mode": "held_out"},
    )
    assert many.status_code == 200
    compared = _wait_job(client, many.get_json()["job_id"])
    assert compared["aggregation"] == "median_of_model_samples"
    assert compared["terminal"]["samples"] == 3
    assert compared["dispersion_label"] == "model sample dispersion"
    assert "probability of profit" in compared["note"]
    assert len(compared["sample_closes"]) == 3
    assert len(compared["actuals"]) == 4
    assert compared["error_stats"]["note"].startswith("Close-price error")
    assert "not a win rate" in compared["error_stats"]["note"].lower()
    assert "win_rate" not in compared
    assert app.config["MODELS"].predictor.calls == [4, 4, 4, 4]


def test_selecting_a_local_file_loads_it_in_that_request():
    app = _ready_app()
    client = app.test_client()
    listed = client.get("/api/datasets/local")
    ids = [item["id"] for item in listed.get_json()["files"]]
    assert "official-sample" in ids
    loaded = client.post("/api/datasets/local", json={"id": "official-sample"})
    assert loaded.status_code == 200
    assert loaded.get_json()["dataset"]["source"] == "sample"
    stale = client.post("/api/forecast", json={"dataset_id": "stale", "lookback": 32, "pred_len": 4})
    assert stale.status_code == 400
    assert "does not match" in stale.get_json()["error"]


def test_runtime_defaults_to_base_and_cpu_without_torch_mps():
    app = create_app()
    client = app.test_client()
    payload = client.get("/api/runtime").get_json()
    assert payload["defaults"]["model_key"] == "kronos-base"
    assert payload["defaults"]["lookback"] == 400
    assert payload["defaults"]["pred_len"] == 24
    assert payload["defaults"]["sample_count"] == 8
    assert payload["server"]["host"] == "127.0.0.1"
    assert payload["server"]["debug"] is False
    assert "mps" not in payload["devices"]["options"] or payload["devices"]["resolved_auto"] == "mps"
    assert {item["code"] for item in payload["intervals"]} >= {"1m", "1h", "1d", "1w", "1M"}


def test_interface_copy_is_english_and_local():
    blob = "\n".join(
        (ROOT / "webui" / relative).read_text(encoding="utf-8")
        for relative in ("templates/index.html", "static/app.js", "static/app.css", "README.md")
        if (ROOT / "webui" / relative).exists() or relative != "README.md"
    )
    page = (ROOT / "webui/templates/index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui/static/app.js").read_text(encoding="utf-8")
    style = (ROOT / "webui/static/app.css").read_text(encoding="utf-8")
    combined = page + script + style
    assert "plotly" not in combined.lower()
    assert "cdn." not in combined.lower()
    assert "fonts.google" not in combined.lower()
    assert not any("\u4e00" <= char <= "\u9fff" for char in combined)
    lowered = combined.lower()
    for banned in ("win rate", "win-rate", "hit rate", "paper trade"):
        assert banned not in lowered
    assert "Load official sample" in page
    assert "not live" in page.lower()


def test_bad_model_key_does_not_pretend_to_load():
    app = create_app()
    client = app.test_client()
    response = client.post("/api/model/load", json={"model_key": "kronos-giant", "device": "cpu"})
    assert response.status_code == 400
    assert "Unknown model" in response.get_json()["error"]
