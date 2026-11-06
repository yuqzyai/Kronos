"""Local Kronos forecast interface.

Binds to 127.0.0.1 only, with the debugger and the reloader off.
The process reads public Binance spot candles or local CSV files and calls
the existing Kronos predictor. It has no account, order, or payment routes.
"""

from __future__ import annotations

import os
import re
import secrets
import sys
import traceback
from pathlib import Path

import pandas as pd

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEB_DIR = Path(__file__).resolve().parent
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from context import build_market_context  # noqa: E402
from derivatives import FuturesClient  # noqa: E402
from evaluate import checkpoint_origins, forecast_record, walk_forward  # noqa: E402
from market import INTERVALS, MarketDataError, BinanceClient, csv_series  # noqa: E402
from service import (  # noqa: E402
    DEFAULT_LOOKBACK,
    DEFAULT_MODEL,
    DEFAULT_PRED_LEN,
    DEFAULT_SAMPLES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MODELS,
    UPLOAD_DIR,
    ApiError,
    DatasetStore,
    JobManager,
    ModelService,
    SessionLog,
    dataset_payload,
    list_local_files,
    load_local_file,
    load_official_sample,
    metadata,
    probe_devices,
    server_options,
)


def create_app(binance: BinanceClient | None = None, store: DatasetStore | None = None, models: ModelService | None = None, futures: FuturesClient | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["BINANCE"] = binance or BinanceClient()
    app.config["FUTURES"] = futures or FuturesClient()
    app.config["STORE"] = store or DatasetStore()
    app.config["MODELS"] = models or ModelService()
    app.config["LOG"] = SessionLog()
    app.config["JOBS"] = JobManager(app.config["MODELS"], app.config["LOG"])
    app.config["PROBE"] = None

    @app.errorhandler(ApiError)
    def _api_error(error: ApiError):
        return jsonify({"error": str(error)}), error.status

    @app.errorhandler(MarketDataError)
    def _market_error(error: MarketDataError):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_error):
        return jsonify({"error": "The CSV is larger than 8 MB. It was not stored."}), 413

    @app.errorhandler(Exception)
    def _unexpected(error: Exception):
        if isinstance(error, ApiError):
            return jsonify({"error": str(error)}), error.status
        if isinstance(error, MarketDataError):
            return jsonify({"error": str(error)}), 400
        if isinstance(error, RequestEntityTooLarge):
            return jsonify({"error": "The CSV is larger than 8 MB. It was not stored."}), 413
        traceback.print_exc()
        return jsonify({"error": "The request failed. The server log has the detail. No prices were invented."}), 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        options = server_options()
        return jsonify({"ok": True, "host": options["host"], "port": options["port"], "debug": options["debug"]})

    @app.get("/api/runtime")
    def runtime():
        probe = _probe(app)
        model_state = app.config["MODELS"].status()
        current = app.config["STORE"].current
        return jsonify(
            {
                "server": server_options(),
                "devices": probe,
                "models": list(MODELS.values()),
                "defaults": {
                    "model_key": DEFAULT_MODEL,
                    "lookback": DEFAULT_LOOKBACK,
                    "pred_len": DEFAULT_PRED_LEN,
                    "sample_count": DEFAULT_SAMPLES,
                    "temperature": DEFAULT_TEMPERATURE,
                    "top_p": DEFAULT_TOP_P,
                    "mode": "forward",
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                },
                "intervals": [
                    {"code": spec.code, "label": spec.label, "seconds": spec.seconds}
                    for spec in INTERVALS.values()
                ],
                "model": model_state,
                "dataset": None if current is None else metadata(current),
                "log": app.config["LOG"].events,
                "popular_symbols": [
                    "BTCUSDT",
                    "ETHUSDT",
                    "BNBUSDT",
                    "SOLUSDT",
                    "XRPUSDT",
                    "DOGEUSDT",
                    "ADAUSDT",
                    "AVAXUSDT",
                    "LINKUSDT",
                    "LTCUSDT",
                ],
            }
        )

    @app.get("/api/datasets/local")
    def local_files():
        return jsonify({"files": list_local_files()})

    @app.post("/api/datasets/sample")
    def load_sample():
        series, label = load_official_sample()
        dataset = _remember(app, series, label)
        app.config["LOG"].add(
            f"Loaded official sample · {len(series.frame)} closed candles · {series.interval_label} · not live"
        )
        return jsonify(dataset_payload(dataset))

    @app.post("/api/datasets/local")
    def load_local():
        body = request.get_json(silent=True) or {}
        series, label = load_local_file(str(body.get("id", "")))
        dataset = _remember(app, series, label)
        app.config["LOG"].add(f"Loaded local CSV {label} · {len(series.frame)} closed candles · not live")
        return jsonify(dataset_payload(dataset))

    @app.post("/api/datasets/csv")
    def load_csv():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise ApiError("Choose a CSV file. Selecting a file loads it; there is no second step.")
        filename = Path(upload.filename).name
        if not filename.lower().endswith(".csv"):
            raise ApiError("The file needs a .csv name. It was not stored.")
        raw = upload.read()
        if not raw:
            raise ApiError("The CSV file is empty.")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ApiError("The CSV must be UTF-8 text.") from exc
        series = csv_series(
            text,
            filename,
            source="csv",
            live=False,
            note=f"Uploaded CSV {filename}. Not live market data.",
        )
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        stored_name = _storage_name(filename)
        (UPLOAD_DIR / stored_name).write_text(text, encoding="utf-8")
        dataset = _remember(app, series, filename)
        app.config["LOG"].add(f"Loaded CSV {filename} · {len(series.frame)} closed candles · not live")
        return jsonify(dataset_payload(dataset))

    @app.post("/api/datasets/binance")
    def load_binance():
        body = request.get_json(silent=True) or {}
        try:
            lookback = int(body.get("lookback", DEFAULT_LOOKBACK))
            pred_len = int(body.get("pred_len", DEFAULT_PRED_LEN))
        except (TypeError, ValueError) as exc:
            raise ApiError("Context and horizon must be whole numbers.") from exc
        closed_limit = max(lookback, lookback + pred_len)
        closed_limit = min(999, max(1, closed_limit))
        series = app.config["BINANCE"].klines(
            body.get("symbol", "BTCUSDT"),
            body.get("interval", "1h"),
            closed_limit,
            force=bool(body.get("force", False)),
        )
        label = f"{series.symbol} · {series.interval}"
        dataset = _remember(app, series, label)
        cache = " · cache" if series.from_cache else ""
        forming = " · forming candle excluded" if series.forming else " · no forming candle in this response"
        app.config["LOG"].add(
            f"Binance spot {series.symbol} {series.interval} · {len(series.frame)} closed{forming}{cache}"
        )
        return jsonify(dataset_payload(dataset))

    @app.get("/api/datasets/current")
    def current_dataset():
        current = app.config["STORE"].current
        if current is None:
            return jsonify({"dataset": None, "candles": []})
        return jsonify(dataset_payload(current))

    @app.post("/api/model/load")
    def load_model():
        body = request.get_json(silent=True) or {}
        info = app.config["MODELS"].load(
            body.get("model_key", DEFAULT_MODEL),
            body.get("device", "auto"),
            _probe(app),
        )
        app.config["LOG"].add(
            f"Loaded {info.name} ({info.params}) on {info.device} in {info.elapsed_seconds:.1f}s · context max {info.context_length}"
        )
        return jsonify({"model": app.config["MODELS"].status(), "log": app.config["LOG"].events})

    @app.post("/api/forecast")
    def forecast():
        body = request.get_json(silent=True) or {}
        current = app.config["STORE"].current
        if current is None:
            raise ApiError("No dataset is loaded. Load Binance candles, the official sample, or a CSV file first.")
        requested = body.get("dataset_id")
        if requested != current.id:
            raise ApiError("The loaded dataset does not match this forecast request. Load the data again, then forecast.")
        started = app.config["JOBS"].start(current, body)
        started["log"] = app.config["LOG"].events
        return jsonify(started)

    @app.get("/api/forecast/jobs/<job_id>")
    def forecast_job(job_id):
        payload = app.config["JOBS"].get(job_id)
        payload["log"] = app.config["LOG"].events
        return jsonify(payload)

    @app.post("/api/forecast/jobs/<job_id>/cancel")
    def cancel_forecast(job_id):
        payload = app.config["JOBS"].cancel(job_id)
        payload["log"] = app.config["LOG"].events
        return jsonify(payload)

    @app.get("/api/forecast/jobs/<job_id>/record")
    def forecast_record_route(job_id):
        job = app.config["JOBS"].jobs.get(job_id)
        if job is None:
            raise ApiError("No forecast with that id is on this server.", 404)
        if job["status"] != "done" or not job.get("result"):
            raise ApiError("A forecast record is available after the run finishes. A cancelled run is not exported as a result.")
        snapshot = job.get("snapshot") or {}
        return jsonify(forecast_record(job["result"], snapshot.get("meta") or {}, snapshot.get("context") or {}))

    @app.post("/api/evaluate")
    def evaluate():
        body = request.get_json(silent=True) or {}
        current = app.config["STORE"].current
        if current is None:
            raise ApiError("No dataset is loaded. Load Binance candles, the official sample, or a CSV file first.")
        info = app.config["MODELS"].info
        predictor = app.config["MODELS"].predictor
        if info is None or predictor is None:
            raise ApiError("Load the model first. The check compares Kronos with persistence and does not run without it.")
        try:
            checkpoints = int(body.get("checkpoints", 2))
        except (TypeError, ValueError) as exc:
            raise ApiError("Checkpoints must be a whole number from 1 to 3.") from exc
        pred_len = min(24, max(1, int(body.get("pred_len", DEFAULT_PRED_LEN))))
        room = len(current.series.frame) - pred_len
        lookback = min(int(body.get("lookback", DEFAULT_LOOKBACK)), info.context_length, room)
        try:
            checkpoint_origins(len(current.series.frame), lookback, pred_len, checkpoints)
        except ValueError as exc:
            raise ApiError(str(exc)) from exc
        frame = current.series.frame.copy()

        def forecaster(history):
            return _kronos_eval_forecast(predictor, history, pred_len)

        report = walk_forward(frame, forecaster, lookback, pred_len, checkpoints)
        report["model"] = {"name": info.name, "device": info.device}
        report["settings"] = {"lookback": lookback, "pred_len": pred_len, "checkpoints": checkpoints, "samples": 1}
        app.config["LOG"].add(
            f"Walk-forward · {checkpoints} checkpoints · median MAE {report['kronos_median_mae']:.6g} · persistence MAE {report['persistence_mae']:.6g}"
        )
        report["log"] = app.config["LOG"].events
        return jsonify(report)

    return app


def _remember(app: Flask, series, label):
    futures = app.config["FUTURES"] if series.source == "binance" else None
    client = app.config["BINANCE"] if series.source == "binance" else None
    context = build_market_context(series, client, futures)
    return app.config["STORE"].put(series, label, context)


def _kronos_eval_forecast(predictor, history, pred_len: int) -> dict:
    step = history["timestamps"].sort_values().diff().dropna().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ApiError("The checkpoint history has no positive spacing, so future timestamps were not invented.")
    last = pd.Timestamp(history["timestamps"].iloc[-1])
    future = pd.DatetimeIndex([last + step * (i + 1) for i in range(pred_len)])
    columns = [column for column in ("open", "high", "low", "close", "volume", "amount") if column in history.columns]
    pred = predictor.predict(
        df=history[columns].reset_index(drop=True),
        x_timestamp=pd.Series(pd.DatetimeIndex(history["timestamps"]), name="timestamps"),
        y_timestamp=pd.Series(future, name="timestamps"),
        pred_len=pred_len,
        sample_count=1,
        verbose=False,
    )
    return {"median_closes": pred["close"].astype(float).to_list()}


def _probe(app: Flask) -> dict:
    cached = app.config.get("PROBE")
    if cached is None:
        cached = probe_devices()
        app.config["PROBE"] = cached
    return cached


def _storage_name(filename: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    if not base.lower().endswith(".csv"):
        base += ".csv"
    return f"{secrets.token_hex(4)}_{base}"


app = create_app()


if __name__ == "__main__":
    options = server_options()
    # Ignore environment flags that would turn the debugger or a public bind back on.
    os.environ.pop("FLASK_DEBUG", None)
    os.environ["FLASK_RUN_HOST"] = options["host"]
    print(f"Kronos Web UI at http://{options['host']}:{options['port']}")
    print("Localhost only. Debugger and reloader are off.")
    app.run(**options)
