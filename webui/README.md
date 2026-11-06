# Kronos local forecast UI

A localhost interface for the Kronos candlestick model. It loads real candles, loads a public Kronos checkpoint, and draws the model's own forecast. It does not place orders, keep an account, or invent prices when a feed fails.

`predict` still returns the mean of its internal samples. The UI also asks the same sampler for the individual paths, in batches of 4, and draws their empirical median. That does not change the model weights.

## Run

From the repository root, with the existing virtualenv activated:

```bash
pip install -r requirements.txt -r webui/requirements.txt
cd webui
python run.py
```

Then open http://127.0.0.1:7070

The server binds to `127.0.0.1` only. The Flask debugger and the reloader are off. `run.py` does not install packages and does not open a browser.

`webui/start.sh` is the same start path.

## What the screen does

1. Load data. Binance spot public klines, the official sample, or a CSV. Choosing a file loads it in that same action.
2. Load a model. The default is Kronos-base (102.3M parameters), the largest public checkpoint. Context defaults to 400 and cannot exceed that model's maximum (512 for base and small, 2048 for mini).
3. Run a forecast. The default horizon is 24 candles and the default sample count is 8.

Auto device selection uses Apple Silicon MPS when that PyTorch process actually has it, then CUDA if MPS is absent, otherwise CPU.

The default run draws 8 paths, in batches of 4, with a cap of 16. The candles are the empirical median. The mean is separate, and a few paths are traced. The band is model sample dispersion, not a calibrated confidence interval and not a probability of profit. More paths reduce sampler noise. They are not evidence the median is closer to the future price. One path draws no band. Temperature and top-p only change token sampling.

Forward mode continues after the last closed candle. Held-out mode forecasts the last horizon candles already in the file and reports close-price MAE, RMSE, and MAPE of the median path against those later candles. That is an error on this file, not a trading result.

Higher timeframes are closed-candle measurements aligned to the base as-of. For Binance they are separate public kline requests. For a file they are clock bins that end at or before the as-of. They are not forecasts and they are not a score.

USD-M funding, open interest, and taker buy/sell volume come only from `https://fapi.binance.com`. If that host is blocked, the panel says so and leaves the fields empty. Spot taker-buy share, when the kline response includes it, is quote-volume executed by taker buys. Order-book depth is not shown.

A walk-forward check compares one Kronos sample per checkpoint with a persistence forecast on this file. The public weights' training end is not known here. Context overlays are not inputs, so they cannot change that error.

The research lab is a separate, opt-in replay. It scores the raw Kronos path, a predeclared 25% mix toward the last close, a residual shift taken only from earlier windows, and a last-close repeat. The scored windows do not overlap. Every method is reported. None is selected, and the result is not drawn on the chart. Because the training end is unknown, the replay is not a pristine out-of-sample test. Direction counts are matches on those windows, not a calibrated probability, and they are not the scenario share of one forecast's paths.

## Binance spot

Public market data only, no API key:

- `GET /api/v3/klines`
- `GET /api/v3/exchangeInfo?symbol=...`

The host is `https://data-api.binance.vision`, which the [Binance spot REST documentation](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md) specifies for public market data. `https://api.binance.com` is contacted only if the market-data host has a transport failure. An HTTP 451 or 403 is shown as a failure and is not routed around. No other host, and no user-supplied URL, is requested.

Intervals are the documented kline intervals (`1s` through `1M`). Each request asks for at most 1000 klines. Quote asset volume is stored as `amount`. Open time is the candle timestamp, in UTC. A candle whose close time is still in the future is shown as forming and is not passed to the model. The forecast's first open is one interval after the last closed candle's open.

Responses are cached per symbol, interval, and length (about 30 seconds to 5 minutes). Refresh is a button. A rate limit keeps the last successful series, labeled with its age, or shows the Binance error if nothing is cached.

If Binance cannot be reached, load the official sample or a CSV. The chart is left empty rather than filled with substitute prices.

## CSV and the official sample

The official sample button loads `tests/data/regression_input.csv` and labels it as not live.

A CSV needs `open`, `high`, `low`, `close`, and one timestamp column named `timestamps`, `timestamp`, `date`, `time`, `datetime`, or `datetime_utc`. `volume` and `amount` are optional. Missing timestamps are an error. Naive timestamps are read as UTC. If `amount` is absent and `volume` is present, Kronos fills amount from volume times the mean of OHLC when forecasting. If both are absent, Kronos fills them with zeros. Both cases are the existing library behavior and are written on the data panel.

Uploads are stored under `webui/uploads/` and are limited to 8 MB and 50,000 rows.

## Tests

From the repository root:

```bash
python -m pytest tests/test_webui.py tests/test_webui_context.py tests/test_webui_research.py tests/test_webui_binance_live.py
```

`tests/test_webui.py` uses a fake predictor and a fake HTTP client. `tests/test_webui_binance_live.py` performs one real public request. Neither test downloads model weights.

## Deploy and roll back

On the machine that already has the virtualenv, cache, and data:

1. Leave `.venv`, the Hugging Face cache, and `data/` in place. This UI does not write into them.
2. Fetch the branch and check it out, or apply the patch from the delivery bundle onto a clean copy of the recorded base commit.
3. Install only if the environment does not already satisfy `requirements.txt` and `webui/requirements.txt`:
   `pip install -r requirements.txt -r webui/requirements.txt`
4. Stop the old listener yourself if it is still bound to port 7070. Do not kill an unrelated process.
5. Start with `cd webui && python run.py`.
6. Open http://127.0.0.1:7070 and load the official sample. The chart should show its candles and the forecast button should stay disabled until you load a model.

To roll back, check out the previous commit (the base SHA recorded in the delivery manifest) with `git checkout <sha>` or `git switch -`. Do not use `git reset --hard`, and do not delete `.venv`, model caches, or data files. Uploads in `webui/uploads/` are local files; leave them unless you put them there during a trial.

## License

Kronos is copyright 2025 ShiYu and distributed under the MIT License. See the repository `LICENSE`.
