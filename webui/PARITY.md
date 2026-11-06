# Feature parity

Every item below is still in the local page. A control may sit behind Focus → Controls, or inside a closed disclosure. It is not removed.

| Capability | Where it lives | How to reach it |
| --- | --- | --- |
| Binance spot load and refresh | Data → Binance | Visible in Workspace. In Focus, Controls opens the rail. |
| Official sample, labeled not live | Data → Sample | Same rail. |
| CSV upload and a file already on the machine | Data → CSV | Choosing a file loads it. |
| Model, device, context, horizon | Model | Number fields and selects in the rail. |
| Temperature, top-p, sample count | Model → Sampling | Native disclosure. Keyboard opens it. |
| Forward and held-out | Forecast → Window | Select in the rail. |
| Run, progress, cancel | Rail foot | Run stays disabled until data and the model are ready. Cancel appears while a job is running. |
| Median line and forecast candles | Chart | Drawn after a forecast. The median is the middle path, not a probability. |
| Show paths and Show dispersion | Chart tools | Off until pressed. Also `p` and `b` while the chart has focus. Neither starts a new forecast. |
| Pan, zoom, fit | Chart | Drag, wheel, Fit forecast, and `f`, Home, or Escape while the chart has focus. Fit does not call the server. |
| Higher timeframes | Side card | Summary is a button. Enter or Space toggles it. |
| Perpetual market | Side card | Same. A blocked host stays empty. |
| Walk-forward, 2 checkpoints | Side card | Run 2 checkpoints. Unchanged contract, at most 3 windows, possibly overlapping. |
| Research lab | Side card, added | Run research replay. Opt-in. Does not replace the chart or the 2-checkpoint check. |
| Focus and Workspace | Top bar | Radio buttons. Session, Data, and Forecast stay in both. |
| Provenance and details | Forecast → Details, Data, Session | Details is a disclosure. The record download stays on the walk-forward card. |
| Scenario share | Forecast | Shown after a multi-path run. Counts of draws, not a chance of profit. |

Research methods, all reported, none selected: raw Kronos, mean only when it is a different path, a fixed 25% mix toward the last close, a residual shift from earlier windows only, and repeating the last close. A lower error on that replay is not a recommendation. The training cutoff is unknown, so the replay is not pristine out-of-sample evidence.

## One measured CPU replay

Kronos-mini (`NeoQuasar/Kronos-mini` with `NeoQuasar/Kronos-Tokenizer-2k`) on CPU, two non-overlapping windows, context 64, horizon 4, one draw per window, on `tests/data/regression_input.csv` (2,500 rows). Wall time on this machine was about 2 seconds, including the weight load. This is not Kronos-base, not the default 400/24 setting, and not a general result.

| Method | Close MAE | Same direction |
| --- | --- | --- |
| Raw Kronos | 0.00561 | 1 of 2 |
| Last-close mix (25%, predeclared) | 0.00608 | 1 of 2 |
| Earlier residual | 0.00675 | 1 of 2 |
| Repeat last close | 0.01000 | 1 of 2 |

Mean was not a separate path, because each window was one draw. On these two windows the raw path had a lower close error than the three alternatives. That describes this replay only. No method is selected. The sample file may overlap the unknown training period.
