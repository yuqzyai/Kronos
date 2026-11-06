"""Dataset, model, and forecast orchestration for the local Kronos web UI.

Forecasting calls KronosPredictor.predict with sample_count=1. A request for
several samples repeats that call. The chart can then show those draws and
their average. It does not turn one average into several paths.
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market import CandleSeries, describe_horizon


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "tests" / "data" / "regression_input.csv"
DATA_DIR = ROOT / "data"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"

MIN_LOOKBACK = 32
MAX_PRED_LEN = 120
DEFAULT_LOOKBACK = 400
DEFAULT_PRED_LEN = 24
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_SAMPLES = 1
MAX_SAMPLES = 4

MODELS = {
    "kronos-mini": {
        "key": "kronos-mini",
        "name": "Kronos-mini",
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "context_length": 2048,
        "params": "4.1M",
        "description": "Smallest public Kronos model.",
    },
    "kronos-small": {
        "key": "kronos-small",
        "name": "Kronos-small",
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "24.7M",
        "description": "Mid-size public Kronos model.",
    },
    "kronos-base": {
        "key": "kronos-base",
        "name": "Kronos-base",
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "102.3M",
        "description": "Largest public Kronos model.",
    },
}
DEFAULT_MODEL = "kronos-base"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class LoadedDataset:
    id: str
    series: CandleSeries
    label: str


@dataclass
class ModelInfo:
    key: str
    name: str
    model_id: str
    tokenizer_id: str
    params: str
    context_length: int
    device: str
    elapsed_seconds: float
    loaded_at: str


class SessionLog:
    def __init__(self):
        self.events: list[dict] = []

    def add(self, message: str) -> None:
        self.events.append({"time": _iso(datetime.now(timezone.utc)), "message": message})
        self.events = self.events[-40:]


class DatasetStore:
    def __init__(self):
        self.current: LoadedDataset | None = None

    def put(self, series: CandleSeries, label: str) -> LoadedDataset:
        loaded = LoadedDataset(id=secrets.token_hex(8), series=series, label=label)
        self.current = loaded
        return loaded


class ModelService:
    def __init__(self):
        self.lock = threading.Lock()
        self.predictor = None
        self.info: ModelInfo | None = None
        self.busy: str | None = None
        self.import_error: str | None = None

    def status(self) -> dict:
        with self.lock:
            info = self.info
            busy = self.busy
            loaded = self.predictor is not None and info is not None
        payload = {
            "loaded": loaded,
            "busy": busy,
            "import_error": self.import_error,
        }
        if info is not None:
            payload["current"] = {
                "key": info.key,
                "name": info.name,
                "model_id": info.model_id,
                "tokenizer_id": info.tokenizer_id,
                "params": info.params,
                "context_length": info.context_length,
                "device": info.device,
                "elapsed_seconds": info.elapsed_seconds,
                "loaded_at": info.loaded_at,
            }
        return payload

    def load(self, model_key: str, device_choice: str, probe: dict) -> ModelInfo:
        if model_key not in MODELS:
            raise ApiError(f"Unknown model '{model_key}'. Choose kronos-mini, kronos-small, or kronos-base.")
        config = MODELS[model_key]
        device = resolve_device(device_choice, probe)
        with self.lock:
            if self.busy:
                raise ApiError("The model is already busy. Wait for the current load or forecast to finish.", 409)
            self.busy = "loading"
        started = time.perf_counter()
        try:
            Kronos, KronosTokenizer, KronosPredictor, error = import_kronos()
            if error:
                self.import_error = error
                raise ApiError(
                    "The Kronos library did not import in this Python, so weights were not loaded. "
                    f"{error}",
                    400,
                )
            tokenizer = KronosTokenizer.from_pretrained(config["tokenizer_id"])
            model = Kronos.from_pretrained(config["model_id"])
            predictor = KronosPredictor(
                model,
                tokenizer,
                device=device,
                max_context=config["context_length"],
            )
            try:
                device = str(next(predictor.model.parameters()).device)
            except Exception:
                device = str(device)
            elapsed = time.perf_counter() - started
            info = ModelInfo(
                key=config["key"],
                name=config["name"],
                model_id=config["model_id"],
                tokenizer_id=config["tokenizer_id"],
                params=config["params"],
                context_length=config["context_length"],
                device=str(device),
                elapsed_seconds=round(elapsed, 3),
                loaded_at=_iso(datetime.now(timezone.utc)),
            )
            with self.lock:
                self.predictor = predictor
                self.info = info
                self.import_error = None
            return info
        finally:
            with self.lock:
                self.busy = None

    def forecast(self, dataset: LoadedDataset, params: dict) -> dict:
        with self.lock:
            if self.busy:
                raise ApiError("The model is busy. Wait for the current load or forecast to finish.", 409)
            predictor = self.predictor
            info = self.info
            if predictor is None or info is None:
                raise ApiError("Load the model first. Data can be on the chart before the weights are loaded.")
            self.busy = "forecasting"
        started = time.perf_counter()
        try:
            return _run_forecast(predictor, info, dataset, params)
        finally:
            with self.lock:
                self.busy = None


def import_kronos():
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    except Exception as exc:  # library missing or a broken environment
        return None, None, None, f"{exc.__class__.__name__}: {exc}"
    return Kronos, KronosTokenizer, KronosPredictor, None


def probe_devices() -> dict:
    try:
        import torch
    except Exception as exc:
        return {
            "torch": False,
            "detail": f"{exc.__class__.__name__}: {exc}",
            "options": ["auto", "cpu"],
            "resolved_auto": "cpu",
            "auto_reason": "PyTorch is not importable here, so Auto would use CPU and model loading will fail until it imports.",
        }
    mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    cuda = bool(torch.cuda.is_available())
    if mps:
        resolved = "mps"
        reason = "Apple Silicon MPS is available in this process, so Auto uses MPS."
    elif cuda:
        resolved = "cuda:0"
        reason = "MPS is not available. CUDA is available, so Auto uses cuda:0."
    else:
        resolved = "cpu"
        reason = "MPS is not available in this process, so Auto uses CPU."
    options = ["auto", "cpu"]
    if cuda:
        options.append("cuda:0")
    if mps:
        options.append("mps")
    return {
        "torch": True,
        "detail": f"torch {torch.__version__}",
        "options": options,
        "resolved_auto": resolved,
        "auto_reason": reason,
    }


def resolve_device(choice: object, probe: dict) -> str:
    if not isinstance(choice, str) or choice not in probe["options"]:
        allowed = ", ".join(probe["options"])
        raise ApiError(f"Device must be one of: {allowed}.")
    if choice == "auto":
        if not probe.get("torch"):
            raise ApiError(
                "Auto resolves to CPU, but PyTorch is not importable in this process, so the model was not loaded. "
                + str(probe.get("detail") or "")
            )
        return probe["resolved_auto"]
    if choice != "cpu" and not probe.get("torch"):
        raise ApiError("That device needs PyTorch, which did not import in this process.")
    return choice


def server_options() -> dict:
    return {
        "host": "127.0.0.1",
        "port": 7070,
        "debug": False,
        "use_reloader": False,
        "threaded": True,
    }


def metadata(dataset: LoadedDataset) -> dict:
    series = dataset.series
    frame = series.frame
    last = frame.iloc[-1] if len(frame) else None
    first = frame.iloc[0] if len(frame) else None
    return {
        "id": dataset.id,
        "label": dataset.label,
        "source": series.source,
        "live": series.live,
        "symbol": series.symbol,
        "interval": series.interval,
        "interval_label": series.interval_label,
        "step_seconds": series.step_seconds,
        "step_kind": series.step_kind,
        "timezone": series.timezone,
        "fetched_at": _iso(series.fetched_at),
        "as_of": _iso(series.as_of) if series.as_of is not None else None,
        "first_open": _iso(first["timestamps"]) if first is not None else None,
        "last_open": _iso(last["timestamps"]) if last is not None else None,
        "last_close_price": float(last["close"]) if last is not None else None,
        "previous_close_price": float(frame["close"].iloc[-2]) if len(frame) > 1 else None,
        "closed_count": int(len(frame)),
        "forming": series.forming,
        "note": series.note,
        "warnings": list(series.warnings),
        "host": series.host,
        "from_cache": series.from_cache,
        "cache_age_seconds": series.cache_age_seconds,
        "used_weight_1m": series.used_weight_1m,
        "amount_policy": series.amount_policy,
        "columns": [column for column in ("open", "high", "low", "close", "volume", "amount") if column in frame.columns],
    }


def candle_records(dataset: LoadedDataset) -> list[dict]:
    frame = dataset.series.frame
    records = []
    for row in frame.itertuples(index=False):
        item = {
            "time": _iso(row.timestamps),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "closed": True,
        }
        if hasattr(row, "volume"):
            item["volume"] = float(row.volume)
        if hasattr(row, "amount"):
            item["amount"] = float(row.amount)
        records.append(item)
    return records


def dataset_payload(dataset: LoadedDataset) -> dict:
    return {"dataset": metadata(dataset), "candles": candle_records(dataset)}


def list_local_files() -> list[dict]:
    items = []
    if SAMPLE_PATH.is_file():
        items.append(
            {
                "id": "official-sample",
                "label": "Official historical sample (tests/data/regression_input.csv) · not live",
                "kind": "sample",
            }
        )
    items.extend(_csv_children(DATA_DIR, "data"))
    items.extend(_csv_children(UPLOAD_DIR, "upload"))
    return items


def load_local_file(file_id: str) -> tuple[CandleSeries, str]:
    if file_id == "official-sample":
        return load_official_sample()
    kind, name = _split_local_id(file_id)
    root = DATA_DIR if kind == "data" else UPLOAD_DIR
    path = _safe_child(root, name)
    text = path.read_text(encoding="utf-8")
    from market import csv_series

    series = csv_series(
        text,
        path.name,
        source="csv",
        live=False,
        note=f"Local CSV {path.name}. Not live market data.",
    )
    return series, path.name


def load_official_sample() -> tuple[CandleSeries, str]:
    if not SAMPLE_PATH.is_file():
        raise ApiError("The official sample file tests/data/regression_input.csv is not in this checkout.", 404)
    from market import csv_series

    text = SAMPLE_PATH.read_text(encoding="utf-8")
    series = csv_series(
        text,
        SAMPLE_PATH.name,
        source="sample",
        live=False,
        symbol=None,
        note=(
            "Official historical sample shipped with the Kronos tests "
            "(tests/data/regression_input.csv). Not live market data."
        ),
    )
    return series, "Official historical sample"


def _split_local_id(file_id: str) -> tuple[str, str]:
    if not isinstance(file_id, str) or ":" not in file_id:
        raise ApiError("Unknown local file.")
    kind, name = file_id.split(":", 1)
    if kind not in {"data", "upload"} or not name:
        raise ApiError("Unknown local file.")
    return kind, name


def _safe_child(root: Path, name: str) -> Path:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ApiError("That file name is not allowed.")
    root = root.resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file() or path.suffix.lower() != ".csv":
        raise ApiError("That file is not an available CSV in the local data folders.")
    return path


def _csv_children(root: Path, kind: str) -> list[dict]:
    if not root.is_dir():
        return []
    items = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() == ".csv" and path.name == Path(path.name).name:
            items.append({"id": f"{kind}:{path.name}", "label": path.name, "kind": kind})
    return items


def _run_forecast(predictor, info: ModelInfo, dataset: LoadedDataset, params: dict) -> dict:
    started = time.perf_counter()
    lookback = _parse_int(params.get("lookback", DEFAULT_LOOKBACK), "Context", MIN_LOOKBACK, info.context_length)
    pred_len = _parse_int(params.get("pred_len", DEFAULT_PRED_LEN), "Horizon", 1, MAX_PRED_LEN)
    temperature = _parse_float(params.get("temperature", DEFAULT_TEMPERATURE), "Temperature", 0.1, 2.0)
    top_p = _parse_float(params.get("top_p", DEFAULT_TOP_P), "top_p", 0.5, 1.0)
    sample_count = _parse_int(params.get("sample_count", DEFAULT_SAMPLES), "Samples", 1, MAX_SAMPLES)
    mode = params.get("mode", "forward")
    if mode not in {"forward", "held_out"}:
        raise ApiError("Mode must be 'forward' or 'held_out'.")
    if lookback > info.context_length:
        raise ApiError(
            f"Context {lookback} is above {info.name}'s maximum of {info.context_length}. "
            "The extra rows were not truncated in secret."
        )
    frame = dataset.series.frame
    window = _window(frame, dataset.series, lookback, pred_len, mode)
    columns = [column for column in ("open", "high", "low", "close", "volume", "amount") if column in window.input_frame.columns]
    x_df = window.input_frame[columns].reset_index(drop=True)
    x_timestamp = pd.Series(pd.DatetimeIndex(window.input_frame["timestamps"]), name="timestamps")
    y_timestamp = pd.Series(pd.DatetimeIndex(window.future_index), name="timestamps")
    samples = []
    for _ in range(sample_count):
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_k=0,
            top_p=top_p,
            sample_count=1,
            verbose=False,
        )
        samples.append(_frame_records(pred_df, window.future_index))
    if sample_count == 1:
        path = samples[0]
        aggregation = "single_sample"
        sample_closes = None
        note = "One call to KronosPredictor.predict with sample_count=1. The candles are that single path, not an ensemble drawn as extra futures."
    else:
        path = _average_paths(samples)
        aggregation = "mean_of_independent_samples"
        sample_closes = [[{"time": row["time"], "close": row["close"]} for row in sample] for sample in samples]
        note = (
            f"{sample_count} separate calls to KronosPredictor.predict with sample_count=1. "
            "Each thin line is one of those draws. The candles are their arithmetic mean. "
            "The spread is not a calibrated confidence interval."
        )
    actual_records = None
    error_stats = None
    if window.actual_frame is not None:
        actual_records = _ohlc_records(window.actual_frame)
        error_stats = _close_errors(path, actual_records)
    elapsed = round(time.perf_counter() - started, 3)
    horizon_text = describe_horizon(pred_len, dataset.series.interval_label, dataset.series.step_seconds)
    return {
        "dataset_id": dataset.id,
        "mode": mode,
        "aggregation": aggregation,
        "sample_count": sample_count,
        "temperature": temperature,
        "top_p": top_p,
        "lookback": lookback,
        "pred_len": pred_len,
        "horizon_text": horizon_text,
        "boundary_open": _iso(window.boundary_open),
        "forecast_opens_at": path[0]["time"] if path else None,
        "path": path,
        "sample_closes": sample_closes,
        "actuals": actual_records,
        "error_stats": error_stats,
        "elapsed_seconds": elapsed,
        "note": note,
        "model": {
            "name": info.name,
            "model_id": info.model_id,
            "params": info.params,
            "device": info.device,
            "context_length": info.context_length,
        },
        "input": {
            "rows": lookback,
            "first_open": _iso(window.input_frame["timestamps"].iloc[0]),
            "last_open": _iso(window.boundary_open),
            "last_close_price": float(window.input_frame["close"].iloc[-1]),
        },
    }


@dataclass
class ForecastWindow:
    input_frame: pd.DataFrame
    future_index: pd.DatetimeIndex
    boundary_open: datetime
    actual_frame: pd.DataFrame | None = None


def _window(frame: pd.DataFrame, series: CandleSeries, lookback: int, pred_len: int, mode: str) -> ForecastWindow:
    if mode == "forward":
        if len(frame) < lookback:
            raise ApiError(
                f"This dataset has {len(frame)} closed candles. Context {lookback} does not fit. "
                "Lower the context or load more history."
            )
        history = frame.iloc[-lookback:].copy()
        future_index = future_timestamps(history["timestamps"].iloc[-1], series, pred_len)
        return ForecastWindow(history, future_index, history["timestamps"].iloc[-1].to_pydatetime())
    need = lookback + pred_len
    if len(frame) < need:
        raise ApiError(
            f"Held-out comparison needs {need} closed candles ({lookback} of context plus {pred_len} later candles). "
            f"This dataset has {len(frame)}."
        )
    block = frame.iloc[-need:].copy()
    history = block.iloc[:lookback].copy()
    actual = block.iloc[lookback:].copy()
    future_index = pd.DatetimeIndex(actual["timestamps"])
    if future_index[0] <= history["timestamps"].iloc[-1]:
        raise ApiError("The held-out candles do not start after the input. The forecast was not run.")
    return ForecastWindow(history, future_index, history["timestamps"].iloc[-1].to_pydatetime(), actual)


def future_timestamps(last_open, series: CandleSeries, pred_len: int) -> pd.DatetimeIndex:
    last = pd.Timestamp(last_open)
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")
    if series.step_kind == "month" or (series.interval == "1M"):
        cursor = last
        values = []
        for _ in range(pred_len):
            cursor = cursor + pd.DateOffset(months=1)
            values.append(cursor)
        return pd.DatetimeIndex(values)
    if series.step_seconds is None:
        raise ApiError("The interval could not be inferred, so a forward forecast has no timestamps to continue. Use held-out comparison on a file that already contains the later candles.")
    step = pd.Timedelta(seconds=int(series.step_seconds))
    values = []
    cursor = last
    for _ in range(pred_len):
        cursor = cursor + step
        values.append(cursor)
    index = pd.DatetimeIndex(values)
    if index[0] <= last:
        raise ApiError("The forecast timestamp did not fall after the input close. The forecast was not run.")
    return index


def _average_paths(samples: list[list[dict]]) -> list[dict]:
    keys = ("open", "high", "low", "close", "volume", "amount")
    averaged = []
    for rows in zip(*samples):
        item = {"time": rows[0]["time"]}
        for key in keys:
            values = [row[key] for row in rows if key in row and row[key] is not None]
            if not values:
                continue
            item[key] = float(np.mean(values))
        high_floor = max(item.get("open", 0), item.get("close", 0))
        low_cap = min(item.get("open", 0), item.get("close", 0))
        if "high" in item:
            item["high"] = max(item["high"], high_floor)
        if "low" in item:
            item["low"] = min(item["low"], low_cap)
        averaged.append(item)
    return averaged


def _frame_records(pred_df: pd.DataFrame, future_index: pd.DatetimeIndex) -> list[dict]:
    records = []
    values = pred_df.reset_index(drop=True)
    if len(values) != len(future_index):
        raise ApiError(
            f"The model returned {len(values)} rows for a horizon of {len(future_index)}. The result was discarded."
        )
    for i, timestamp in enumerate(future_index):
        row = values.iloc[i]
        item = {"time": _iso(timestamp)}
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in values.columns:
                number = float(row[column])
                if not math.isfinite(number):
                    raise ApiError("The model returned a non-finite price. The result was discarded.")
                item[column] = number
        records.append(item)
    return records


def _ohlc_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for row in frame.itertuples(index=False):
        item = {
            "time": _iso(row.timestamps),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        if hasattr(row, "volume"):
            item["volume"] = float(row.volume)
        if hasattr(row, "amount"):
            item["amount"] = float(row.amount)
        records.append(item)
    return records


def _close_errors(path: list[dict], actuals: list[dict]) -> dict | None:
    pairs = []
    for pred, actual in zip(path, actuals):
        if pred.get("close") is None or actual.get("close") in (None, 0):
            continue
        pairs.append((float(pred["close"]), float(actual["close"])))
    if not pairs:
        return None
    errors = [abs(pred - actual) for pred, actual in pairs]
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean([error * error for error in errors])))
    mape = float(np.mean([error / abs(actual) * 100 for error, (_pred, actual) in zip(errors, pairs)]))
    return {
        "points": len(pairs),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "note": "Close-price error against later candles in this dataset. Not a win rate and not a trading result.",
    }


def _parse_int(value, name: str, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{name} must be a whole number.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ApiError(f"{name} must be a whole number.")
    if number < low or number > high:
        raise ApiError(f"{name} must be between {low} and {high}.")
    return number


def _parse_float(value, name: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{name} must be a number.") from exc
    if not math.isfinite(number) or number < low or number > high:
        raise ApiError(f"{name} must be between {low} and {high}.")
    return number


def _iso(value) -> str:
    if value is None:
        return ""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
