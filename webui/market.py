"""Public market data and CSV parsing for the local Kronos web UI.

Binance access is limited to the documented spot market-data hosts and the
two public GET paths this interface needs. Callers cannot supply a URL.
"""

from __future__ import annotations

import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import pandas as pd


# Official public market-data host. General REST host is used only when the
# market-data host has a transport failure, never after an HTTP 451/403.
# https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md
PRIMARY_HOST = "https://data-api.binance.vision"
SECONDARY_HOST = "https://api.binance.com"
ALLOWED_HOSTS = frozenset({PRIMARY_HOST, SECONDARY_HOST})
ALLOWED_PATHS = frozenset({"/api/v3/klines", "/api/v3/exchangeInfo"})

USER_AGENT = "KronosLocalResearch/1.0 (+https://github.com/shiyu-coder/Kronos)"
MAX_BODY_BYTES = 5_000_000
REQUEST_TIMEOUT_SECONDS = 10


class MarketDataError(Exception):
    """A readable failure. Callers must not replace the result with invented prices."""


class _TransportError(MarketDataError):
    def __init__(self, public: str):
        super().__init__(public)
        self.public = public


class _BlockedError(MarketDataError):
    def __init__(self, status: int, public: str):
        super().__init__(public)
        self.status = status
        self.public = public


class _RateLimitError(MarketDataError):
    def __init__(self, public: str, retry_after: float):
        super().__init__(public)
        self.retry_after = retry_after
        self.public = public


@dataclass(frozen=True)
class IntervalSpec:
    code: str
    seconds: int | None
    label: str
    step_kind: str  # "timedelta" or "month"


# Official kline intervals, case-sensitive.
# https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#kline-intervals
INTERVALS: dict[str, IntervalSpec] = {
    "1s": IntervalSpec("1s", 1, "1 second", "timedelta"),
    "1m": IntervalSpec("1m", 60, "1 minute", "timedelta"),
    "3m": IntervalSpec("3m", 180, "3 minutes", "timedelta"),
    "5m": IntervalSpec("5m", 300, "5 minutes", "timedelta"),
    "15m": IntervalSpec("15m", 900, "15 minutes", "timedelta"),
    "30m": IntervalSpec("30m", 1800, "30 minutes", "timedelta"),
    "1h": IntervalSpec("1h", 3600, "1 hour", "timedelta"),
    "2h": IntervalSpec("2h", 7200, "2 hours", "timedelta"),
    "4h": IntervalSpec("4h", 14400, "4 hours", "timedelta"),
    "6h": IntervalSpec("6h", 21600, "6 hours", "timedelta"),
    "8h": IntervalSpec("8h", 28800, "8 hours", "timedelta"),
    "12h": IntervalSpec("12h", 43200, "12 hours", "timedelta"),
    "1d": IntervalSpec("1d", 86400, "1 day", "timedelta"),
    "3d": IntervalSpec("3d", 259200, "3 days", "timedelta"),
    "1w": IntervalSpec("1w", 604800, "1 week", "timedelta"),
    "1M": IntervalSpec("1M", None, "1 calendar month", "month"),
}


@dataclass
class CandleSeries:
    source: str
    live: bool
    symbol: str | None
    interval: str | None
    interval_label: str
    step_kind: str
    step_seconds: int | None
    timezone: str
    frame: pd.DataFrame
    forming: dict | None
    fetched_at: datetime
    as_of: datetime | None
    note: str
    warnings: list[str] = field(default_factory=list)
    host: str | None = None
    from_cache: bool = False
    cache_age_seconds: float | None = None
    used_weight_1m: int | None = None
    amount_policy: str = "provided"

    def cached_view(self, age_seconds: float, extra_warning: str | None = None) -> "CandleSeries":
        warnings = list(self.warnings)
        if extra_warning:
            warnings.append(extra_warning)
        return replace(
            self,
            from_cache=True,
            cache_age_seconds=age_seconds,
            warnings=warnings,
        )


def normalize_symbol(value: object) -> str:
    if not isinstance(value, str):
        raise MarketDataError("Symbol must be text, for example BTCUSDT.")
    symbol = value.strip().upper()
    if not symbol or len(symbol) < 5 or len(symbol) > 20 or not symbol.isalnum():
        raise MarketDataError(
            "Symbol must be 5 to 20 letters or digits, for example BTCUSDT. "
            "It is checked against Binance spot exchange info before any candles are used."
        )
    return symbol


def require_interval(value: object) -> IntervalSpec:
    if not isinstance(value, str) or value not in INTERVALS:
        allowed = ", ".join(INTERVALS)
        raise MarketDataError(f"Interval must be one of the Binance spot kline intervals: {allowed}.")
    return INTERVALS[value]


def step_from_spec(spec: IntervalSpec):
    if spec.step_kind == "month":
        return pd.DateOffset(months=1)
    return pd.Timedelta(seconds=spec.seconds)


def describe_horizon(count: int, label: str, step_seconds: int | None) -> str:
    if step_seconds is None:
        unit = "candle" if count == 1 else "candles"
        return f"{count} {unit} · {label} each"
    total = step_seconds * count
    if step_seconds % 86400 == 0:
        days = total // 86400
        span = "1\u00a0day" if days == 1 else f"{days}\u00a0days"
    elif step_seconds % 3600 == 0:
        hours = total // 3600
        span = "1\u00a0hour" if hours == 1 else f"{hours}\u00a0hours"
    elif step_seconds % 60 == 0:
        minutes = total // 60
        if minutes % 60 == 0:
            hours = minutes // 60
            span = "1\u00a0hour" if hours == 1 else f"{hours}\u00a0hours"
        else:
            span = "1\u00a0minute" if minutes == 1 else f"{minutes}\u00a0minutes"
    else:
        span = "1\u00a0second" if total == 1 else f"{total}\u00a0seconds"
    unit = "candle" if count == 1 else "candles"
    safe_label = label.replace(" ", "\u00a0")
    return f"{count}\u00a0{unit} · {safe_label}\u00a0each · {span}"


def _finite_positive(value: float, label: str, index: int) -> float:
    if not math.isfinite(value) or value <= 0:
        raise MarketDataError(f"Row {index}: {label} must be a finite number above zero.")
    return value


def _check_ohlc(open_, high, low, close, index: int) -> None:
    if high < max(open_, close, low) or low > min(open_, close, high) or low > high:
        raise MarketDataError(
            f"Row {index}: high/low do not contain the open and close "
            f"(open {open_}, high {high}, low {low}, close {close})."
        )


def parse_kline_row(row: object, index: int) -> dict:
    """Map one Binance kline array onto the Kronos OHLCV/amount schema.

    Official order: open time, open, high, low, close, volume, close time,
    quote asset volume, trades, taker base, taker quote, ignore.
    Quote asset volume is the turnover Kronos calls amount.
    """

    if not isinstance(row, (list, tuple)) or len(row) < 8:
        raise MarketDataError(
            f"Kline row {index} is missing fields. Expected open time, OHLC, volume, "
            "close time, and quote asset volume."
        )
    try:
        open_ms = int(row[0])
        close_ms = int(row[6])
        open_ = float(row[1])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        volume = float(row[5])
        amount = float(row[7])
    except (TypeError, ValueError) as exc:
        raise MarketDataError(f"Kline row {index} has a non-numeric price, volume, or timestamp.") from exc
    open_ = _finite_positive(open_, "open", index)
    high = _finite_positive(high, "high", index)
    low = _finite_positive(low, "low", index)
    close = _finite_positive(close, "close", index)
    if not math.isfinite(volume) or volume < 0 or not math.isfinite(amount) or amount < 0:
        raise MarketDataError(f"Row {index}: volume and quote volume must be finite and cannot be negative.")
    _check_ohlc(open_, high, low, close, index)
    if close_ms < open_ms:
        raise MarketDataError(f"Row {index}: close time is earlier than open time.")
    return {
        "open_time_ms": open_ms,
        "close_time_ms": close_ms,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
    }


def split_closed_candles(rows: object, now: datetime) -> tuple[list[dict], dict | None, list[str]]:
    if not isinstance(rows, list):
        raise MarketDataError("Binance kline payload was not a list of candles.")
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    parsed = [parse_kline_row(row, i) for i, row in enumerate(rows)]
    parsed.sort(key=lambda item: item["open_time_ms"])
    warnings: list[str] = []
    deduped: list[dict] = []
    for item in parsed:
        if deduped and deduped[-1]["open_time_ms"] == item["open_time_ms"]:
            deduped[-1] = item
            warnings.append("Duplicate open times were collapsed to the last copy.")
        else:
            deduped.append(item)
    closed = [item for item in deduped if item["close_time_ms"] < now_ms]
    forming_rows = [item for item in deduped if item["close_time_ms"] >= now_ms]
    forming = forming_rows[-1] if forming_rows else None
    if len(forming_rows) > 1:
        warnings.append("More than one candle was still open. The latest open candle is shown and excluded from the model.")
    return closed, forming, warnings


def frame_from_closed(closed: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamps": pd.to_datetime([row["open_time_ms"] for row in closed], unit="ms", utc=True),
            "open": [row["open"] for row in closed],
            "high": [row["high"] for row in closed],
            "low": [row["low"] for row in closed],
            "close": [row["close"] for row in closed],
            "volume": [row["volume"] for row in closed],
            "amount": [row["amount"] for row in closed],
        }
    )
    return frame


def forming_record(forming: dict | None) -> dict | None:
    if forming is None:
        return None
    opened = datetime.fromtimestamp(forming["open_time_ms"] / 1000, tz=timezone.utc)
    closed_at = datetime.fromtimestamp(forming["close_time_ms"] / 1000, tz=timezone.utc)
    return {
        "time": opened.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "close_time": closed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": forming["open"],
        "high": forming["high"],
        "low": forming["low"],
        "close": forming["close"],
        "volume": forming["volume"],
        "amount": forming["amount"],
        "closed": False,
    }


TIMESTAMP_NAMES = ("timestamps", "timestamp", "date", "time", "datetime", "datetime_utc")


def load_csv_frame(text: str, filename: str) -> tuple[pd.DataFrame, list[str], str, str, int | None]:
    """Parse a user CSV into closed candles. Missing timestamps are an error, not invented hours."""

    try:
        frame = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError as exc:
        raise MarketDataError(f"{filename} is empty.") from exc
    except Exception as exc:
        raise MarketDataError(f"{filename} could not be read as CSV ({exc}).") from exc
    if frame.empty:
        raise MarketDataError(f"{filename} has headers but no rows.")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = [name for name in ("open", "high", "low", "close") if name not in frame.columns]
    if missing:
        raise MarketDataError(
            f"{filename} is missing required columns: {', '.join(missing)}. "
            f"Found columns: {', '.join(map(str, frame.columns))}."
        )
    timestamp_name = next((name for name in TIMESTAMP_NAMES if name in frame.columns), None)
    if timestamp_name is None:
        raise MarketDataError(
            f"{filename} has no timestamp column. Expected one of: {', '.join(TIMESTAMP_NAMES)}. "
            "Timestamps were not invented."
        )
    warnings: list[str] = []
    timestamps = pd.to_datetime(frame[timestamp_name], utc=True, errors="coerce")
    if frame[timestamp_name].notna().any() and timestamps.isna().all():
        raise MarketDataError(f"{filename} has timestamp values that could not be parsed.")
    if timestamps.isna().any() and frame[timestamp_name].notna().any():
        # Rows with an unparseable stamp are dropped later with the numeric blanks.
        warnings.append("Some timestamp values could not be parsed and those rows were dropped.")
    warnings.append("Timestamps without an explicit offset were read as UTC. Offset timestamps were converted to UTC.")
    out = pd.DataFrame({"timestamps": timestamps})
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    amount_policy = "filled_by_model_as_zero"
    if "volume" in frame.columns:
        out["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        amount_policy = "filled_by_model_from_volume"
    if "amount" in frame.columns:
        out["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
        amount_policy = "provided"
    before = len(out)
    out = out.dropna().copy()
    dropped = before - len(out)
    if dropped:
        warnings.append(f"{dropped} row(s) with blank or non-numeric values were dropped.")
    if out.empty:
        raise MarketDataError(f"{filename} has no complete rows after parsing.")
    if len(out) > 50000:
        raise MarketDataError(f"{filename} has {len(out)} rows. The limit for this local UI is 50,000 rows.")
    out = out.sort_values("timestamps")
    if out["timestamps"].duplicated().any():
        dupes = int(out["timestamps"].duplicated().sum())
        out = out.drop_duplicates("timestamps", keep="last")
        warnings.append(f"{dupes} duplicate timestamp(s) were collapsed to the last copy.")
    out = out.reset_index(drop=True)
    for index, row in out.iterrows():
        _check_ohlc(float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), int(index))
        for column in ("open", "high", "low", "close"):
            if not math.isfinite(float(row[column])) or float(row[column]) <= 0:
                raise MarketDataError(f"Row {int(index)}: {column} must be a finite number above zero.")
        if "volume" in out.columns and (not math.isfinite(float(row["volume"])) or float(row["volume"]) < 0):
            raise MarketDataError(f"Row {int(index)}: volume cannot be negative.")
        if "amount" in out.columns and (not math.isfinite(float(row["amount"])) or float(row["amount"]) < 0):
            raise MarketDataError(f"Row {int(index)}: amount cannot be negative.")
    step_seconds, gap_warning = infer_step_seconds(out["timestamps"])
    if gap_warning:
        warnings.append(gap_warning)
    if "volume" not in out.columns:
        warnings.append("No volume column. Kronos fills volume and amount with zeros when it forecasts. That is the existing library behavior.")
    elif "amount" not in out.columns:
        warnings.append("No amount column. Kronos fills amount from volume times the mean of OHLC. That is the existing library behavior.")
    return out, warnings, amount_policy, "UTC", step_seconds


def infer_step_seconds(timestamps: pd.Series) -> tuple[int | None, str | None]:
    if len(timestamps) < 2:
        return None, "Only one candle is loaded, so the interval cannot be inferred."
    diffs = timestamps.sort_values().diff().dropna().dt.total_seconds()
    median = float(diffs.median())
    if not math.isfinite(median) or median <= 0:
        raise MarketDataError("Timestamps are not increasing, so a forecast interval cannot be inferred.")
    irregular = (diffs - median).abs() > 1.0
    ratio = float(irregular.mean()) if len(irregular) else 0.0
    warning = None
    if ratio > 0.05:
        warning = (
            f"{ratio:.0%} of gaps differ from the median spacing "
            f"({format_seconds(median)}). Forward forecasts continue by that median gap and do not follow an exchange calendar. "
            "Held-out comparison uses the file's own later timestamps."
        )
    return int(round(median)), warning


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds % 86400 == 0:
        days = seconds // 86400
        return "1 day" if days == 1 else f"{days} days"
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return "1 hour" if hours == 1 else f"{hours} hours"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return "1 minute" if minutes == 1 else f"{minutes} minutes"
    return "1 second" if seconds == 1 else f"{seconds} seconds"


def csv_series(text: str, filename: str, source: str, live: bool, note: str, symbol: str | None = None) -> CandleSeries:
    frame, warnings, amount_policy, timezone_name, step_seconds = load_csv_frame(text, filename)
    label = format_seconds(step_seconds) if step_seconds else "unknown interval"
    as_of = frame["timestamps"].iloc[-1].to_pydatetime() if len(frame) else None
    return CandleSeries(
        source=source,
        live=live,
        symbol=symbol,
        interval=None,
        interval_label=label,
        step_kind="timedelta" if step_seconds else "unknown",
        step_seconds=step_seconds,
        timezone=timezone_name,
        frame=frame,
        forming=None,
        fetched_at=datetime.now(timezone.utc),
        as_of=as_of,
        note=note,
        warnings=warnings,
        host=None,
        amount_policy=amount_policy,
    )


def _build_opener():
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise MarketDataError(
                f"Binance returned a redirect ({code}). This app does not follow redirects, "
                "so a response cannot send it to another host."
            )

    return urllib.request.build_opener(_NoRedirect)


class BinanceClient:
    def __init__(self, opener=None, clock=None, monotonic=None):
        self.opener = opener or _build_opener()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic or time.monotonic
        self._cache: dict[tuple, tuple[float, float, CandleSeries]] = {}
        self._symbols: dict[str, tuple[float, dict]] = {}
        self._backoff_until = 0.0
        self._last_network = 0.0
        self._last_by_key: dict[tuple, float] = {}

    def klines(self, symbol: str, interval: str, closed_limit: int, force: bool = False) -> CandleSeries:
        symbol = normalize_symbol(symbol)
        spec = require_interval(interval)
        try:
            closed_limit = int(closed_limit)
        except (TypeError, ValueError) as exc:
            raise MarketDataError("Candle count must be a whole number.") from exc
        if closed_limit < 1 or closed_limit > 999:
            raise MarketDataError(
                "Ask for between 1 and 999 closed candles. Binance returns at most 1000 klines "
                "per request, and one of them may still be forming."
            )
        api_limit = closed_limit + 1
        key = (symbol, spec.code, api_limit)
        now_m = self.monotonic()
        cached = self._cache.get(key)
        ttl = 30 if spec.seconds is None else max(30, min(300, spec.seconds / 2))
        if cached and not force and now_m - cached[0] < ttl:
            age = now_m - cached[0]
            return cached[2].cached_view(age)
        if self._backoff_until > now_m:
            wait = int(math.ceil(self._backoff_until - now_m))
            if cached:
                return cached[2].cached_view(
                    now_m - cached[0],
                    f"Binance rate limit still applies for about {wait}s. Showing the last successful load, not new prices.",
                )
            raise MarketDataError(
                f"Binance rate limit still applies for about {wait}s, and there is no cached load to show. "
                "No substitute prices were generated."
            )
        if force and key in self._last_by_key and now_m - self._last_by_key[key] < 15:
            if cached:
                age = now_m - cached[0]
                return cached[2].cached_view(
                    age,
                    "Refresh was kept at the last successful load. Binance was not called again within 15 seconds.",
                )
            raise MarketDataError("A Binance request for this series just ran. Wait a few seconds before another refresh.")
        if now_m - self._last_network < 1.0 and cached and not force:
            return cached[2].cached_view(now_m - cached[0], "Served from cache to avoid a second Binance request in the same second.")

        self._ensure_spot_symbol(symbol)
        try:
            payload, weight, host = self._get(
                "/api/v3/klines",
                {"symbol": symbol, "interval": spec.code, "limit": api_limit, "timeZone": "0"},
            )
        except _RateLimitError as exc:
            cached = self._cache.get(key)
            if cached:
                return cached[2].cached_view(self.monotonic() - cached[0], exc.public)
            raise MarketDataError(exc.public) from exc
        now = self.clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        closed, forming, warnings = split_closed_candles(payload, now)
        if not closed:
            raise MarketDataError(
                "Binance returned candles, but none were closed. The forming candle is not sent to the model, "
                "so this load was discarded instead of inventing history."
            )
        if len(closed) < closed_limit:
            warnings.append(
                f"Binance returned {len(closed)} closed candles, fewer than the {closed_limit} requested."
            )
        frame = frame_from_closed(closed)
        as_of = datetime.fromtimestamp(closed[-1]["close_time_ms"] / 1000, tz=timezone.utc)
        series = CandleSeries(
            source="binance",
            live=True,
            symbol=symbol,
            interval=spec.code,
            interval_label=spec.label,
            step_kind=spec.step_kind,
            step_seconds=spec.seconds,
            timezone="UTC",
            frame=frame,
            forming=forming_record(forming),
            fetched_at=now.astimezone(timezone.utc),
            as_of=as_of,
            note=(
                f"Binance spot public klines from {host}. "
                "Closed candles are eligible for the model. The forming candle, if shown, is excluded."
            ),
            warnings=warnings,
            host=host.replace("https://", ""),
            used_weight_1m=weight,
            amount_policy="provided",
        )
        stored_at = self.monotonic()
        self._cache[key] = (stored_at, ttl, series)
        self._last_by_key[key] = stored_at
        return series

    def _ensure_spot_symbol(self, symbol: str) -> None:
        now_m = self.monotonic()
        cached = self._symbols.get(symbol)
        if cached and now_m - cached[0] < 6 * 3600 and cached[1].get("ok"):
            return
        if cached and now_m - cached[0] < 600 and not cached[1].get("ok"):
            raise MarketDataError(cached[1]["message"])
        try:
            payload, _weight, _host = self._get("/api/v3/exchangeInfo", {"symbol": symbol})
        except MarketDataError as exc:
            text = str(exc)
            if "Invalid symbol" in text or "not listed" in text or "not an active" in text:
                self._symbols[symbol] = (now_m, {"ok": False, "message": text})
            raise
        symbols = payload.get("symbols") if isinstance(payload, dict) else None
        info = symbols[0] if isinstance(symbols, list) and symbols else None
        if not isinstance(info, dict) or info.get("symbol") != symbol:
            message = f"{symbol} was not listed in the Binance spot exchange info response."
            self._symbols[symbol] = (now_m, {"ok": False, "message": message})
            raise MarketDataError(message)
        if info.get("status") != "TRADING" or not info.get("isSpotTradingAllowed"):
            message = (
                f"{symbol} is not an active Binance spot market "
                f"(status {info.get('status')}, spot trading {info.get('isSpotTradingAllowed')})."
            )
            self._symbols[symbol] = (now_m, {"ok": False, "message": message})
            raise MarketDataError(message)
        self._symbols[symbol] = (now_m, {"ok": True})

    def _get(self, path: str, params: dict):
        try:
            return self._once(PRIMARY_HOST, path, params)
        except _TransportError as primary_error:
            try:
                return self._once(SECONDARY_HOST, path, params)
            except _BlockedError as blocked:
                raise MarketDataError(
                    f"{PRIMARY_HOST} did not respond ({primary_error.public}). "
                    f"{SECONDARY_HOST} then returned HTTP {blocked.status}. "
                    "No further hosts were tried, and no candles were invented. "
                    "Load the official sample or a CSV file."
                ) from blocked
            except _TransportError as secondary_error:
                raise MarketDataError(
                    f"Neither {PRIMARY_HOST} nor {SECONDARY_HOST} responded ({secondary_error.public}). "
                    "No candles were invented. Load the official sample or a CSV file."
                ) from secondary_error

    def _once(self, host: str, path: str, params: dict):
        if host not in ALLOWED_HOSTS:
            raise MarketDataError("Refused a host that is not in the Binance allowlist.")
        if path not in ALLOWED_PATHS:
            raise MarketDataError("Refused a Binance path that this UI does not use.")
        query = urllib.parse.urlencode(params)
        url = f"{host}{path}?{query}"
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            method="GET",
        )
        self._last_network = self.monotonic()
        try:
            with self.opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", 200)
                headers = response.headers
                body = response.read(MAX_BODY_BYTES + 1)
        except urllib.error.HTTPError as exc:
            body = exc.read(MAX_BODY_BYTES + 1) if exc.fp is not None else b""
            self._raise_http(exc.code, exc.headers, body, host)
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise _TransportError(f"connection failed ({reason})") from exc
        except TimeoutError as exc:
            raise _TransportError("timed out after 10 seconds") from exc
        except MarketDataError:
            raise
        except Exception as exc:
            raise _TransportError(f"request failed ({exc.__class__.__name__})") from exc
        if len(body) > MAX_BODY_BYTES:
            raise MarketDataError("Binance response was larger than 5 MB and was discarded.")
        if status in (403, 451):
            self._raise_http(status, headers, body, host)
        weight = _weight_header(headers)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MarketDataError(f"Binance returned a response that was not JSON from {host}.") from exc
        if isinstance(payload, dict) and "code" in payload and "msg" in payload:
            raise MarketDataError(f"Binance rejected the request ({payload.get('code')}): {payload.get('msg')}")
        return payload, weight, host

    def _raise_http(self, status: int, headers, body: bytes, host: str) -> None:
        message = _binance_message(body)
        if status in (403, 451):
            raise _BlockedError(
                status,
                f"Binance refused this request from {host} (HTTP {status}). "
                "This usually means the public API is blocked on this network. "
                "The block was not bypassed, and live candles were not replaced with synthetic prices. "
                "Load the official sample or a CSV file.",
            )
        if status == 429:
            retry_after = _retry_after(headers)
            self._backoff_until = self.monotonic() + retry_after
            raise _RateLimitError(
                f"Binance rate limit (HTTP 429). Wait about {int(math.ceil(retry_after))}s before refreshing. "
                "Cached candles, if any, are kept and labeled with their age.",
                retry_after,
            )
        if status == 418:
            retry_after = _retry_after(headers)
            self._backoff_until = self.monotonic() + retry_after
            raise MarketDataError(
                f"Binance returned HTTP 418 (temporary ban) from {host}. "
                f"Wait about {int(math.ceil(retry_after))}s. No further hosts were tried."
            )
        if status >= 500:
            raise _TransportError(f"HTTP {status} from {host}")
        if message:
            raise MarketDataError(f"Binance rejected the request (HTTP {status}): {message}")
        raise MarketDataError(f"Binance returned HTTP {status} from {host}.")


def _weight_header(headers) -> int | None:
    if headers is None:
        return None
    for key in ("X-MBX-USED-WEIGHT-1M", "x-mbx-used-weight-1m"):
        raw = headers.get(key) if hasattr(headers, "get") else None
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    return None


def _retry_after(headers) -> float:
    raw = None if headers is None or not hasattr(headers, "get") else headers.get("Retry-After")
    if raw is None:
        return 30.0
    try:
        return max(15.0, min(900.0, float(raw)))
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(raw))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(15.0, min(900.0, (when - datetime.now(timezone.utc)).total_seconds()))
        except (TypeError, ValueError, OSError):
            return 30.0


def _binance_message(body: bytes) -> str | None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return None
    if isinstance(payload, dict) and payload.get("msg"):
        return str(payload["msg"])
    return None
