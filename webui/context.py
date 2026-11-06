"""Higher-timeframe measurements aligned to a base series.

These are descriptions of closed candles. They are not Kronos forecasts and
they are not combined into a score.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from market import INTERVALS, MarketDataError


# At most two coarser intervals. Each one is a documented Binance interval.
UPPER_INTERVALS = {
    "1s": ["1m", "15m"],
    "1m": ["15m", "1h"],
    "3m": ["15m", "1h"],
    "5m": ["1h", "4h"],
    "15m": ["1h", "4h"],
    "30m": ["4h", "1d"],
    "1h": ["4h", "1d"],
    "2h": ["1d"],
    "4h": ["1d", "1w"],
    "6h": ["1d"],
    "8h": ["1d"],
    "12h": ["1d", "1w"],
    "1d": ["1w"],
    "3d": ["1w"],
    "1w": ["1M"],
    "1M": [],
}


def as_of_close(series) -> pd.Timestamp:
    """Time through which the base series is closed, in UTC."""

    if series.as_of is not None and series.source == "binance":
        stamp = pd.Timestamp(series.as_of)
    else:
        stamp = pd.Timestamp(series.frame["timestamps"].iloc[-1])
        if series.step_seconds:
            stamp = stamp + pd.Timedelta(seconds=int(series.step_seconds))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _iso(value) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def filter_closed_through(frame: pd.DataFrame, as_of: pd.Timestamp, close_column: str = "close_time") -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.iloc[0:0].copy() if frame is not None else pd.DataFrame()
    if close_column not in frame.columns:
        raise ValueError("Higher-timeframe rows need a close_time so unfinished bins can be dropped.")
    closes = pd.to_datetime(frame[close_column], utc=True)
    limit = pd.Timestamp(as_of)
    if limit.tzinfo is None:
        limit = limit.tz_localize("UTC")
    return frame.loc[closes <= limit].reset_index(drop=True)


def resample_complete_bins(frame: pd.DataFrame, target_seconds: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """Clock-align loaded candles. A bin is kept only when it ended at or before as_of."""

    if target_seconds <= 0 or frame.empty:
        return pd.DataFrame()
    stamps = pd.to_datetime(frame["timestamps"], utc=True)
    epoch = (stamps - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("int64")
    bin_start = (epoch // target_seconds) * target_seconds
    work = frame.copy()
    work["_bin"] = bin_start.to_numpy()
    rows = []
    for start, group in work.groupby("_bin", sort=True):
        end = int(start) + target_seconds
        end_stamp = pd.to_datetime(end, unit="s", utc=True)
        if end_stamp > pd.Timestamp(as_of).tz_convert("UTC"):
            continue
        ordered = group.sort_values("timestamps")
        row = {
            "timestamps": pd.to_datetime(int(start), unit="s", utc=True),
            "close_time": end_stamp,
            "open": float(ordered["open"].iloc[0]),
            "high": float(ordered["high"].max()),
            "low": float(ordered["low"].min()),
            "close": float(ordered["close"].iloc[-1]),
        }
        if "volume" in ordered.columns:
            row["volume"] = float(ordered["volume"].sum())
        if "amount" in ordered.columns:
            row["amount"] = float(ordered["amount"].sum())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def diagnostics(frame: pd.DataFrame, interval_label: str, source: str, as_of: pd.Timestamp, base_bar_return: float | None) -> dict:
    note = (
        "Measured on closed candles only. Not a Kronos forecast, not an input to the model, "
        "and not a confluence score."
    )
    if frame is None or len(frame) < 2:
        return {
            "status": "insufficient",
            "interval_label": interval_label,
            "source": source,
            "as_of": _iso(as_of),
            "bars": 0 if frame is None else int(len(frame)),
            "note": note + " Fewer than two closed bars were available at this as-of.",
        }
    last = frame.iloc[-1]
    previous = frame.iloc[-2]
    bar_return = _return(float(last["open"]), float(last["close"]))
    close_to_close = _return(float(previous["close"]), float(last["close"]))
    window_return = _return(float(frame["close"].iloc[0]), float(frame["close"].iloc[-1]))
    closes = frame["close"].astype(float).to_numpy()
    log_returns = np.diff(np.log(closes))
    volatility = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else None
    volume_note = None
    volume_ratio = None
    if "volume" in frame.columns and frame["volume"].notna().all():
        median_volume = float(np.median(frame["volume"].astype(float)))
        last_volume = float(last["volume"])
        if median_volume > 0:
            volume_ratio = last_volume / median_volume
        else:
            volume_note = "Median volume is zero, so last volume was not turned into a ratio."
    else:
        volume_note = "Volume was not on these bars."
    agreement = _sign_agreement(base_bar_return, bar_return)
    return {
        "status": "ok",
        "interval_label": interval_label,
        "source": source,
        "as_of": _iso(as_of),
        "bars": int(len(frame)),
        "last_open": _iso(last["timestamps"]),
        "last_close_time": _iso(last["close_time"]) if "close_time" in frame.columns else None,
        "last_bar_return": bar_return,
        "last_close_to_close_return": close_to_close,
        "window_return": window_return,
        "log_return_volatility": volatility,
        "last_volume_over_median": volume_ratio,
        "volume_note": volume_note,
        "sign_vs_base_last_bar": agreement,
        "note": note,
    }


def _return(start: float, end: float) -> float | None:
    if not np.isfinite(start) or not np.isfinite(end) or start == 0:
        return None
    return float((end - start) / start)


def _sign_agreement(base_return: float | None, other_return: float | None) -> str:
    if base_return is None or other_return is None:
        return "not comparable"
    if base_return == 0 or other_return == 0:
        return "one bar is unchanged"
    if (base_return > 0) == (other_return > 0):
        return "same sign on the last closed bar"
    return "opposite sign on the last closed bar"


def build_market_context(series, client=None, futures=None) -> dict:
    higher = higher_timeframes(series, client if getattr(series, "source", None) == "binance" else None)
    taker = spot_taker_summary(series)
    if getattr(series, "source", None) != "binance" or not getattr(series, "symbol", None):
        derivatives = {
            "status": "not_applicable",
            "reason": "Perpetual funding and open interest are requested only for a Binance spot symbol. This file was not sent to a futures host.",
            "funding": None,
            "open_interest": None,
            "taker": None,
        }
    elif futures is None:
        derivatives = {
            "status": "unavailable",
            "reason": "No futures client is configured. No funding or open interest was invented.",
            "funding": None,
            "open_interest": None,
            "taker": None,
        }
    else:
        derivatives = futures.context_for_spot(series.symbol, series.interval or "1h", as_of_close(series))
    return {
        "higher_timeframes": higher,
        "spot_taker": taker,
        "derivatives": derivatives,
        "note": "Context is descriptive. None of it is fed to Kronos, and none of it is a score.",
    }


def base_bar_return(series) -> float | None:
    frame = series.frame
    if frame is None or frame.empty:
        return None
    last = frame.iloc[-1]
    return _return(float(last["open"]), float(last["close"]))


def spot_taker_summary(series) -> dict:
    frame = series.frame
    if "taker_buy_quote" not in frame.columns or "amount" not in frame.columns:
        return {
            "status": "absent",
            "reason": "These candles have no taker-buy quote field, so no executed-taker share was computed.",
        }
    usable = frame.dropna(subset=["taker_buy_quote", "amount"])
    usable = usable[usable["amount"] > 0]
    if usable.empty:
        return {
            "status": "absent",
            "reason": "Taker-buy quote was present but no candle had positive quote volume to divide by.",
        }
    share = usable["taker_buy_quote"].astype(float) / usable["amount"].astype(float)
    last = float(share.iloc[-1])
    window = share.tail(min(48, len(share)))
    return {
        "status": "ok",
        "source": "spot kline taker-buy quote volume",
        "last_taker_buy_share_of_quote": last,
        "recent_mean_taker_buy_share_of_quote": float(window.mean()),
        "bars": int(len(window)),
        "last_open": _iso(usable["timestamps"].iloc[-1]),
        "note": (
            "Share of quote volume executed by taker buys on these closed spot candles. "
            "The rest of the quote volume is the taker-sell side implied by that field. "
            "This is not an order book, and it is not a complete record of every order."
        ),
    }


def higher_timeframes(series, client=None) -> list[dict]:
    limit = as_of_close(series)
    base_return = base_bar_return(series)
    results = []
    if series.source == "binance" and series.interval:
        for code in UPPER_INTERVALS.get(series.interval, []):
            results.append(_from_exchange(client, series.symbol, code, limit, base_return))
        return results
    if not series.step_seconds:
        return [{
            "status": "unavailable",
            "reason": "The loaded candles have no regular spacing, so a coarser timeframe was not built.",
        }]
    for code in _resample_targets(series.step_seconds):
        spec = INTERVALS[code]
        try:
            frame = resample_complete_bins(series.frame, spec.seconds, limit)
        except Exception as exc:
            results.append({
                "status": "unavailable",
                "interval": code,
                "reason": f"Could not resample to {spec.label}: {exc}",
            })
            continue
        item = diagnostics(frame, spec.label, "resampled from the loaded candles", limit, base_return)
        item["interval"] = code
        item["note"] = (
            item["note"] + " Bins are clock-aligned. A bin is included only when its end is at or before the base as-of. "
            "Gaps in the file are left as gaps. This was not a separate exchange request."
        )
        results.append(item)
    return results


def _resample_targets(step_seconds: int) -> list[str]:
    ordered = ["5m", "15m", "1h", "4h", "1d", "1w"]
    chosen = []
    for code in ordered:
        seconds = INTERVALS[code].seconds
        if seconds and seconds >= step_seconds * 4 and seconds % step_seconds == 0:
            chosen.append(code)
        if len(chosen) == 2:
            break
    return chosen


def _from_exchange(client, symbol: str, interval: str, as_of: pd.Timestamp, base_return: float | None) -> dict:
    spec = INTERVALS[interval]
    if client is None:
        return {"status": "unavailable", "interval": interval, "reason": "No market client was available."}
    try:
        series = client.klines(symbol, interval, closed_limit=120)
    except (MarketDataError, Exception) as exc:
        return {
            "status": "unavailable",
            "interval": interval,
            "interval_label": spec.label,
            "reason": str(exc),
            "as_of": _iso(as_of),
        }
    frame = series.frame
    if "close_time" not in frame.columns and series.step_seconds:
        frame = frame.copy()
        frame["close_time"] = pd.to_datetime(frame["timestamps"], utc=True) + pd.Timedelta(seconds=series.step_seconds)
    aligned = filter_closed_through(frame, as_of)
    item = diagnostics(aligned, spec.label, "Binance spot klines", as_of, base_return)
    item["interval"] = interval
    dropped = int(len(frame) - len(aligned))
    if dropped:
        item["dropped_after_as_of"] = dropped
    return item
