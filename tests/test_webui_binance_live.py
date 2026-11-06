"""One real request to Binance's public market-data host.

This is not a mock. A failure here is a real network or HTTP error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webui"))

import pandas as pd

from derivatives import FuturesClient  # noqa: E402
from market import PRIMARY_HOST, BinanceClient, MarketDataError  # noqa: E402


def test_live_public_spot_klines():
    client = BinanceClient()
    try:
        series = client.klines("BTCUSDT", "1h", closed_limit=8)
    except MarketDataError as exc:
        pytest.fail(f"Live Binance request failed with a readable error, not synthetic prices: {exc}")
    assert series.live is True
    assert series.host == PRIMARY_HOST.replace("https://", "")
    assert series.timezone == "UTC"
    assert len(series.frame) >= 7
    assert float(series.frame["close"].iloc[-1]) > 0
    assert float(series.frame["amount"].iloc[-1]) > 0
    assert str(series.frame["timestamps"].dt.tz) == "UTC"
    if series.forming is not None:
        assert series.forming["closed"] is False


def test_live_usd_m_context_is_real_or_an_explicit_block():
    """One real call to fapi.binance.com. A geo block must stay empty, not zero-filled."""

    result = FuturesClient().context_for_spot("BTCUSDT", "1h", pd.Timestamp("2026-11-06T18:00:00Z"))
    if result["status"] == "unavailable":
        assert result["funding"] is None
        assert result["open_interest"] is None
        assert result["taker"] is None
        assert result["http_status"] in {403, 451, None}
        assert "fapi.binance.com" in result["reason"]
        return
    assert result["status"] == "ok"
    assert result["host"] == "fapi.binance.com"
    assert result["funding"]["status"] == "ok"
    assert result["funding"]["latest"]["funding_rate"] is not None
