"""Public USD-M perpetual context for a spot symbol.

Only https://fapi.binance.com is contacted, and only the documented public
GET paths below. A block is reported. Nothing is filled in with zeros, and
no other host is tried.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from market import USER_AGENT, _build_opener


FUTURES_HOST = "https://fapi.binance.com"
ALLOWED_PATHS = frozenset({
    "/fapi/v1/exchangeInfo",
    "/fapi/v1/fundingRate",
    "/fapi/v1/openInterest",
    "/futures/data/openInterestHist",
    "/futures/data/takerlongshortRatio",
})

# Documented periods for openInterestHist and takerlongshortRatio.
# 8h and weeks are not in that list, so the nearest listed period is named.
OI_PERIOD = {
    "1s": "5m",
    "1m": "5m",
    "3m": "5m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "6h",
    "12h": "12h",
    "1d": "1d",
    "3d": "1d",
    "1w": "1d",
    "1M": "1d",
}


class FuturesError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class FuturesClient:
    def __init__(self, opener=None, clock=None):
        self.opener = opener or _build_opener()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def context_for_spot(self, symbol: str, interval: str, as_of: pd.Timestamp) -> dict:
        checked = _iso(self.clock())
        limit = _ms(as_of)
        try:
            info = self._get("/fapi/v1/exchangeInfo", {"symbol": symbol})
        except FuturesError as exc:
            return _unavailable(symbol, checked, exc)
        symbol_info = _perpetual(info, symbol)
        if symbol_info is None:
            return {
                "status": "not_listed",
                "host": "fapi.binance.com",
                "checked_at": checked,
                "symbol": symbol,
                "reason": (
                    f"{symbol} is not a trading USD-M perpetual on fapi.binance.com. "
                    "Funding and open interest were not requested for a different contract, and no values were filled in."
                ),
                "funding": None,
                "open_interest": None,
                "taker": None,
                "after_as_of": None,
            }
        period = OI_PERIOD.get(interval, "1h")
        try:
            funding_rows = self._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 20})
            oi_now = self._get("/fapi/v1/openInterest", {"symbol": symbol})
            oi_hist = self._get("/futures/data/openInterestHist", {"symbol": symbol, "period": period, "limit": 30})
            taker_rows = self._get("/futures/data/takerlongshortRatio", {"symbol": symbol, "period": period, "limit": 30})
        except FuturesError as exc:
            blocked = _unavailable(symbol, checked, exc)
            blocked["perpetual"] = {"symbol": symbol, "contract": "USD-M perpetual", "status": symbol_info.get("status")}
            return blocked
        funding = _funding(funding_rows, limit)
        oi_history, oi_after = _open_interest(oi_hist, oi_now, limit, period)
        taker = _taker(taker_rows, limit, period)
        return {
            "status": "ok",
            "host": "fapi.binance.com",
            "checked_at": checked,
            "symbol": symbol,
            "as_of": _iso(as_of),
            "perpetual": {
                "symbol": symbol,
                "contract": "USD-M perpetual",
                "quote": symbol_info.get("quoteAsset"),
                "status": symbol_info.get("status"),
                "note": "Same ticker as the spot symbol, different market. These fields are not passed to Kronos.",
            },
            "funding": funding,
            "open_interest": oi_history,
            "taker": taker,
            "after_as_of": oi_after,
            "reason": None,
        }

    def _get(self, path: str, params: dict):
        if path not in ALLOWED_PATHS:
            raise FuturesError("Refused a futures path that this UI does not use.")
        url = f"{FUTURES_HOST}{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}, method="GET")
        try:
            with self.opener.open(request, timeout=10) as response:
                status = getattr(response, "status", 200)
                body = response.read(5_000_001)
                headers = response.headers
        except urllib.error.HTTPError as exc:
            body = exc.read(200_000) if exc.fp is not None else b""
            raise FuturesError(_http_message(exc.code, body), status=exc.code) from exc
        except urllib.error.URLError as exc:
            raise FuturesError(f"fapi.binance.com did not respond ({getattr(exc, 'reason', exc)}). No values were filled in.") from exc
        except TimeoutError as exc:
            raise FuturesError("fapi.binance.com timed out after 10 seconds. No values were filled in.") from exc
        if len(body) > 5_000_000:
            raise FuturesError("The futures response was larger than 5 MB and was discarded.")
        if status in (403, 451):
            raise FuturesError(_http_message(status, body), status=status)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FuturesError("fapi.binance.com returned a response that was not JSON.") from exc
        if isinstance(payload, dict) and payload.get("msg") and "code" in payload and not _looks_like_record(payload):
            raise FuturesError(f"Binance futures rejected the request ({payload.get('code')}): {payload.get('msg')}", status=status)
        return payload


def _looks_like_record(payload: dict) -> bool:
    return "openInterest" in payload or "symbol" in payload and "fundingTime" in payload


def _http_message(status: int, body: bytes) -> str:
    detail = ""
    try:
        payload = json.loads(body.decode("utf-8"))
        if isinstance(payload, dict) and payload.get("msg"):
            detail = f" {payload['msg']}"
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        detail = ""
    if status in (403, 451):
        return (
            f"fapi.binance.com refused the request (HTTP {status}).{detail} "
            "The block was not bypassed. Funding, open interest, and perpetual taker volume were not filled in."
        )
    if status == 429:
        return "fapi.binance.com rate limit (HTTP 429). Perpetual context was not replaced with zeros."
    return f"fapi.binance.com returned HTTP {status}.{detail}"


def _unavailable(symbol: str, checked: str, exc: FuturesError) -> dict:
    return {
        "status": "unavailable",
        "host": "fapi.binance.com",
        "checked_at": checked,
        "symbol": symbol,
        "http_status": exc.status,
        "reason": str(exc),
        "funding": None,
        "open_interest": None,
        "taker": None,
        "after_as_of": None,
    }


def _perpetual(payload, symbol: str) -> dict | None:
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(symbols, list):
        return None
    for item in symbols:
        if not isinstance(item, dict) or item.get("symbol") != symbol:
            continue
        if item.get("contractType") == "PERPETUAL" and item.get("quoteAsset") == "USDT" and item.get("status") == "TRADING":
            return item
    return None


def _funding(rows, as_of_ms: int) -> dict:
    if not isinstance(rows, list):
        return {"status": "absent", "reason": "Funding history was not a list."}
    aligned = []
    after = 0
    for row in rows:
        if not isinstance(row, dict) or "fundingTime" not in row or "fundingRate" not in row:
            continue
        when = int(row["fundingTime"])
        rate = float(row["fundingRate"])
        if not math.isfinite(rate):
            continue
        record = {
            "time": _iso_ms(when),
            "funding_rate": rate,
            "unit": "fraction of position per funding interval",
            "mark_price": _optional_float(row.get("markPrice")),
        }
        if when <= as_of_ms:
            aligned.append(record)
        else:
            after += 1
    aligned.sort(key=lambda item: item["time"])
    if not aligned:
        return {
            "status": "absent",
            "reason": "No funding record was at or before the spot candle as-of. Later records were not used.",
            "dropped_after_as_of": after,
        }
    return {
        "status": "ok",
        "latest": aligned[-1],
        "records": aligned[-8:],
        "dropped_after_as_of": after,
        "note": "USD-M perpetual funding. Not a spot fee and not a model input.",
    }


def _open_interest(history, current, as_of_ms: int, period: str) -> tuple[dict, dict | None]:
    aligned = []
    after_rows = 0
    if isinstance(history, list):
        for row in history:
            if not isinstance(row, dict) or "timestamp" not in row:
                continue
            when = int(row["timestamp"])
            contracts = _optional_float(row.get("sumOpenInterest"))
            value = _optional_float(row.get("sumOpenInterestValue"))
            record = {
                "time": _iso_ms(when),
                "open_interest_contracts": contracts,
                "open_interest_value_usdt": value,
                "period": period,
            }
            if when <= as_of_ms:
                aligned.append(record)
            else:
                after_rows += 1
    aligned.sort(key=lambda item: item["time"])
    history_payload = {
        "status": "ok" if aligned else "absent",
        "period": period,
        "records": aligned[-12:],
        "latest": aligned[-1] if aligned else None,
        "dropped_after_as_of": after_rows,
        "unit": "contracts, and notional value in USDT when Binance sent it",
        "history_limit": "Binance publishes about 30 days of this statistic.",
        "reason": None if aligned else "No open-interest history point was at or before the spot as-of.",
    }
    after = None
    if isinstance(current, dict) and current.get("time") is not None and current.get("openInterest") is not None:
        when = int(current["time"])
        snapshot = {
            "time": _iso_ms(when),
            "open_interest_contracts": _optional_float(current.get("openInterest")),
            "note": "Live snapshot from /fapi/v1/openInterest. A single snapshot is not a history.",
        }
        if when > as_of_ms:
            snapshot["note"] += " This snapshot is after the closed-candle as-of, so it is not shown as something known at that close."
            after = snapshot
        else:
            history_payload["snapshot_at_or_before_as_of"] = snapshot
    return history_payload, after


def _taker(rows, as_of_ms: int, period: str) -> dict:
    if not isinstance(rows, list):
        return {"status": "absent", "reason": "Taker buy/sell history was not a list."}
    aligned = []
    after = 0
    for row in rows:
        if not isinstance(row, dict) or "timestamp" not in row:
            continue
        when = int(row["timestamp"])
        buy = _optional_float(row.get("buyVol"))
        sell = _optional_float(row.get("sellVol"))
        if buy is None or sell is None:
            continue
        total = buy + sell
        share = buy / total if total > 0 else None
        record = {
            "time": _iso_ms(when),
            "taker_buy_volume": buy,
            "taker_sell_volume": sell,
            "taker_buy_share": share,
            "period": period,
        }
        if when <= as_of_ms:
            aligned.append(record)
        else:
            after += 1
    aligned.sort(key=lambda item: item["time"])
    if not aligned:
        return {
            "status": "absent",
            "reason": "No perpetual taker buy/sell bar was at or before the spot as-of.",
            "dropped_after_as_of": after,
        }
    return {
        "status": "ok",
        "latest": aligned[-1],
        "records": aligned[-12:],
        "dropped_after_as_of": after,
        "note": (
            "Executed taker buy and sell volume on the USD-M perpetual for the named period. "
            "Not an order-book snapshot, and Binance keeps about 30 days of this series."
        ),
    }


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _ms(stamp) -> int:
    return int(pd.Timestamp(stamp).tz_convert("UTC").timestamp() * 1000)


def _iso(value) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
