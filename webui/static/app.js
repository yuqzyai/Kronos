(function () {
  "use strict";

  const state = {
    runtime: null,
    dataset: null,
    candles: [],
    forecast: null,
    context: null,
    showPaths: storedFlag("kronos-show-paths"),
    showBand: storedFlag("kronos-show-band"),
    jobId: null,
    progressText: "",
    log: [],
    busy: null,
    source: "binance",
  };

  function storedFlag(key) {
    try {
      return sessionStorage.getItem(key) === "1";
    } catch (_error) {
      return false;
    }
  }

  function storeFlag(key, value) {
    try {
      sessionStorage.setItem(key, value ? "1" : "0");
    } catch (_error) {
      /* private mode can reject storage; the toggle still applies to this view */
    }
  }

  const el = (id) => document.getElementById(id);
  const chart = new PriceChart(el("chart"));

  function formatPrice(value) {
    if (value == null || !Number.isFinite(Number(value))) return "—";
    const number = Number(value);
    const abs = Math.abs(number);
    const digits = abs >= 1000 ? 2 : abs >= 1 ? 4 : 6;
    return number.toLocaleString("en-US", {
      minimumFractionDigits: Math.min(2, digits),
      maximumFractionDigits: digits,
    });
  }

  function formatUtc(iso, withSeconds) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    const pad = (n) => String(n).padStart(2, "0");
    const base = `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}`;
    return (withSeconds ? `${base}:${pad(date.getUTCSeconds())}` : base) + " UTC";
  }

  function clock(iso) {
    if (!iso) return "";
    return iso.slice(11, 19) + "Z";
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    let body = null;
    try {
      body = await response.json();
    } catch (_error) {
      throw new Error("The server returned a response that was not JSON.");
    }
    if (!response.ok) {
      throw new Error(body && body.error ? body.error : "The request failed.");
    }
    return body;
  }

  function setBanner(kind, message) {
    const banner = el("banner");
    if (!message) {
      banner.hidden = true;
      banner.className = "banner";
      return;
    }
    banner.hidden = false;
    banner.className = kind === "ok" ? "banner ok" : "banner";
    el("banner-text").textContent = message;
  }

  function selectedModel() {
    const key = el("model").value;
    const models = (state.runtime && state.runtime.models) || [];
    return models.find((model) => model.key === key) || null;
  }

  function numbers() {
    return {
      lookback: Number(el("lookback").value),
      predLen: Number(el("horizon").value),
      samples: Number(el("samples").value),
      temperature: Number(el("temperature").value),
      topP: Number(el("top-p").value),
      mode: el("mode").value,
    };
  }

  function validateClient() {
    const model = selectedModel();
    const maxContext = model ? model.context_length : 512;
    const { lookback, predLen, samples } = numbers();
    const problems = [];
    if (!Number.isInteger(lookback) || lookback < 32 || lookback > maxContext) {
      problems.push(`Context must be a whole number from 32 to ${maxContext}.`);
    }
    if (!Number.isInteger(predLen) || predLen < 1 || predLen > 120) {
      problems.push("Horizon must be a whole number from 1 to 120.");
    }
    if (!Number.isInteger(samples) || samples < 1 || samples > 16) {
      problems.push("Samples must be a whole number from 1 to 16.");
    }
    return { problems, maxContext };
  }

  function render() {
    const dataset = state.dataset;
    const modelStatus = state.runtime ? state.runtime.model : null;
    const loadedModel = modelStatus && modelStatus.loaded ? modelStatus.current : null;
    const check = validateClient();
    const closed = dataset ? dataset.closed_count : 0;
    const { lookback, predLen, mode } = numbers();

    el("context-line").textContent = dataset
      ? `${dataset.label} · ${dataset.interval_label} · ${dataset.timezone}`
      : "No dataset loaded";

    const pill = el("source-pill");
    pill.className = "pill";
    if (!dataset) {
      pill.querySelector("span").textContent = "Idle";
    } else if (dataset.live) {
      pill.className = "pill live";
      pill.querySelector("span").textContent = dataset.from_cache ? "Live · cache" : "Live";
    } else if (dataset.source === "sample") {
      pill.className = "pill sample";
      pill.querySelector("span").textContent = "Sample · not live";
    } else {
      pill.className = "pill csv";
      pill.querySelector("span").textContent = "CSV · not live";
    }

    const last = el("last-close");
    if (dataset && dataset.last_close_price != null) {
      last.textContent = formatPrice(dataset.last_close_price);
      const prev = dataset.previous_close_price;
      last.className = prev == null ? "v" : dataset.last_close_price >= prev ? "v up" : "v down";
    } else {
      last.textContent = "—";
      last.className = "v";
    }

    el("ready-data").textContent = dataset
      ? `${dataset.live ? "Live" : "Ready"} · ${closed} closed`
      : "Not loaded";
    el("ready-model").textContent = state.busy === "model"
      ? "Loading weights"
      : loadedModel
        ? `${loadedModel.name} · ${loadedModel.device}`
        : "Not loaded";

    let forecastReason = "";
    if (!dataset) forecastReason = "Load data and the model first.";
    else if (!loadedModel) forecastReason = "Data is loaded. Load the model before forecasting.";
    else if (state.busy === "model") forecastReason = "The model is still loading.";
    else if (state.busy === "forecast") forecastReason = state.progressText || "Forecast is running. Finished samples are counted. There is no estimated percent.";
    else if (check.problems.length) forecastReason = check.problems[0];
    else if (mode === "forward" && closed < lookback) {
      forecastReason = `Need ${lookback} closed candles. This dataset has ${closed}.`;
    } else if (mode === "held_out" && closed < lookback + predLen) {
      forecastReason = `Held-out needs ${lookback + predLen} closed candles. This dataset has ${closed}.`;
    }
    el("ready-forecast").textContent = forecastReason ? "Blocked" : "Ready";
    el("run").disabled = Boolean(forecastReason) || state.busy != null;
    el("run-reason").textContent = state.busy === "forecast"
      ? "The chart keeps the previous forecast until this run finishes or is cancelled."
      : (forecastReason || `Ready · ${numbers().lookback} context candles · horizon ${numbers().predLen} · ${numbers().samples} samples.`);
    const progress = el("progress");
    progress.hidden = state.busy !== "forecast";
    progress.textContent = state.progressText || "";
    el("cancel").hidden = state.busy !== "forecast";
    el("refresh").disabled = !(dataset && dataset.source === "binance") || state.busy != null;
    el("load-binance").disabled = state.busy != null;
    el("load-sample").disabled = state.busy != null;
    el("load-model").disabled = state.busy != null;
    el("csv-file").disabled = state.busy != null;
    el("local-file").disabled = state.busy != null;

    const hint = el("context-hint");
    const chosen = selectedModel();
    if (chosen) {
      hint.textContent = `${chosen.name} (${chosen.params}) accepts at most ${chosen.context_length} context candles. Default is 400.`;
      el("lookback").max = String(chosen.context_length);
    }

    renderModelCard(loadedModel, chosen);
    renderForecastPanel();
    renderDataPanel();
    renderContext();
    renderLog();
    renderLegend();
    chart.setData({
      candles: state.candles,
      forming: dataset ? dataset.forming : null,
      forecast: state.forecast,
      horizon: numbers().predLen,
    });
    const summary = el("chart-summary");
    if (!dataset) {
      summary.textContent = "No candles loaded. Load Binance spot candles, the official sample, or a CSV file.";
    } else {
      const forecastText = state.forecast
        ? ` Forecast: ${state.forecast.horizon_text || ""}, ${(state.forecast.aggregation || "forecast").replaceAll("_", " ")}.`
        : " Forecast region is empty until a run finishes.";
      summary.textContent = `${dataset.label}. ${closed} closed candles. Timezone ${dataset.timezone}. ${dataset.note}${forecastText} Arrow keys move the candle readout. Shift plus arrows pan. Home returns to the latest candles.`;
    }
  }

  function renderModelCard(loaded, chosen) {
    const card = el("model-card");
    const spec = loaded || chosen;
    if (!spec) {
      card.innerHTML = "<div class='card-title'><b>KRONOS</b><span class='state'>NO MODEL</span></div>";
      return;
    }
    const name = (loaded ? loaded.name : spec.name).toUpperCase();
    const stateLabel = loaded ? "LOADED" : "NOT LOADED";
    const stateClass = loaded ? "state on" : "state";
    const rows = [
      ["params", loaded ? loaded.params : spec.params],
      ["max", String(loaded ? loaded.context_length : spec.context_length)],
      ["context", el("lookback").value],
      ["horizon", el("horizon").value],
      ["samples", el("samples").value],
      ["device", loaded ? loaded.device : `auto → ${(state.runtime && state.runtime.devices.resolved_auto) || "cpu"}`],
    ];
    card.innerHTML = `<div class="card-title"><b>${name}</b><span class="${stateClass}">${stateLabel}</span></div>
      <p class="card-id">${loaded ? loaded.model_id : spec.model_id}</p>
      <div class="card-metrics">${rows.map(([key, value]) => `<div><span>${key}</span><b>${value}</b></div>`).join("")}</div>`;
  }

  function renderForecastPanel() {
    const forecast = state.forecast;
    const kicker = el("forecast-kicker");
    const figure = el("forecast-figure");
    const label = el("forecast-figure-label");
    const facts = el("forecast-facts");
    if (!forecast) {
      const box = el("scenario-share");
      if (box) box.hidden = true;
      kicker.textContent = "Waiting";
      figure.textContent = "—";
      figure.className = "figure flat";
      label.textContent = "Horizon close";
      facts.innerHTML = factRows([
        ["Horizon", horizonPreview()],
        ["Path", "Nothing has been sampled"],
        ["Region", "Empty until a run finishes"],
      ]);
      return;
    }
    const last = forecast.path[forecast.path.length - 1];
    const first = forecast.path[0];
    const observed = forecast.input.last_close_price;
    const terminal = forecast.terminal;
    figure.textContent = formatPrice(terminal ? terminal.median_close : last.close);
    figure.className = "figure";
    label.textContent = forecast.sample_count > 1 ? "Median horizon close" : "Horizon close";
    kicker.textContent = forecast.mode === "held_out" ? "Held-out" : "Forward";
    const lead = el("forecast-lead");
    if (lead) {
      if (forecast.sample_count > 1 && terminal) {
        lead.textContent = `Median close ${forecast.horizon_text} from now is ${formatPrice(terminal.median_close)}, the middle of ${forecast.sample_count} sampled paths. That is not a chance the market ends there.`;
      } else {
        lead.textContent = `One sampled path. Horizon close is ${formatPrice(last.close)}. A single path has no dispersion band.`;
      }
    }
    renderScenario(forecast);
    const rows = [
      ["Horizon", forecast.horizon_text],
      ["Opens", formatUtc(forecast.forecast_opens_at)],
      ["After input", formatUtc(forecast.boundary_open)],
      ["Input close", formatPrice(observed)],
      ["First open", formatPrice(first.open)],
      ["Path", forecast.sample_count > 1 ? `Median of ${forecast.sample_count} paths` : "Single sampled path"],
      ["Elapsed", `${forecast.elapsed_seconds.toFixed(1)} s`],
      ["Model", `${forecast.model.name} · ${forecast.model.device}`],
    ];
    if (forecast.scenario && forecast.scenario.total > 1) {
      rows.push(["Draws", forecast.scenario.text]);
    }
    if (terminal && forecast.sample_count > 1) {
      rows.push(["Mean close", formatPrice(terminal.mean_close)]);
      rows.push(["Sample q10–q90", `${formatPrice(terminal.q10)} – ${formatPrice(terminal.q90)}`]);
      rows.push(["Dispersion", "Model sample dispersion, not a confidence interval"]);
      rows.push(["Above last close", `${Math.round(terminal.fraction_above * terminal.samples)} of ${terminal.samples} samples`]);
      rows.push(["Fraction", "Fraction of samples, not a market probability"]);
    }
    if (forecast.error_stats) {
      rows.push(["Close MAE", formatPrice(forecast.error_stats.mae)]);
      rows.push(["Close RMSE", formatPrice(forecast.error_stats.rmse)]);
      rows.push(["Close MAPE", `${forecast.error_stats.mape.toFixed(2)}%`]);
      rows.push(["Error", forecast.error_stats.note]);
    }
    facts.innerHTML = factRows(rows);
  }

  function horizonPreview() {
    const count = numbers().predLen;
    if (!Number.isInteger(count) || count < 1) return "Set a horizon";
    const dataset = state.dataset;
    if (!dataset) return `${count} candles · interval appears with the data`;
    const seconds = dataset.step_seconds;
    if (!seconds) return `${count} candles · ${dataset.interval_label}`;
    let span;
    const total = seconds * count;
    if (seconds % 86400 === 0) {
      const days = total / 86400;
      span = days === 1 ? "1\u00a0day" : `${days}\u00a0days`;
    } else if (seconds % 3600 === 0) {
      const hours = total / 3600;
      span = hours === 1 ? "1\u00a0hour" : `${hours}\u00a0hours`;
    } else if (seconds % 60 === 0) {
      const minutes = total / 60;
      if (minutes % 60 === 0) {
        const hours = minutes / 60;
        span = hours === 1 ? "1\u00a0hour" : `${hours}\u00a0hours`;
      } else {
        span = minutes === 1 ? "1\u00a0minute" : `${minutes}\u00a0minutes`;
      }
    } else {
      span = total === 1 ? "1\u00a0second" : `${total}\u00a0seconds`;
    }
    const each = dataset.interval_label.replaceAll(" ", "\u00a0");
    return `${count}\u00a0candles · ${each}\u00a0each · ${span}`;
  }

  function renderDataPanel() {
    const dataset = state.dataset;
    const kicker = el("data-kicker");
    const facts = el("data-facts");
    if (!dataset) {
      kicker.textContent = "No source";
      facts.innerHTML = factRows([
        ["Source", "Nothing loaded"],
        ["Timezone", "UTC when a series is loaded"],
      ]);
      return;
    }
    kicker.textContent = dataset.live ? "Binance spot" : "Not live";
    const rows = [
      ["Source", dataset.live ? "Public spot klines" : dataset.source === "sample" ? "Official sample" : "CSV file"],
      ["Host", dataset.host || "Local file"],
      ["Symbol", dataset.symbol || "—"],
      ["Interval", dataset.interval ? `${dataset.interval} · ${dataset.interval_label}` : dataset.interval_label],
      ["As of", formatUtc(dataset.as_of, true)],
      ["Refreshed", formatUtc(dataset.fetched_at, true)],
      ["Timezone", dataset.timezone],
      ["Closed", String(dataset.closed_count)],
      ["Forming", dataset.forming ? "Shown, excluded from the model" : "None"],
      ["Cache", dataset.from_cache ? `Yes · ${Math.round(dataset.cache_age_seconds || 0)}s old` : "Fresh load"],
    ];
    if (dataset.used_weight_1m != null) rows.push(["Weight 1m", String(dataset.used_weight_1m)]);
    rows.push(["Amount", amountLabel(dataset.amount_policy)]);
    if (dataset.warnings && dataset.warnings.length) rows.push(["Note", dataset.warnings[0]]);
    facts.innerHTML = factRows(rows);
  }

  function renderScenario(forecast) {
    const box = el("scenario-share");
    if (!box) return;
    const share = forecast && forecast.scenario;
    if (!share || share.total < 2) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    el("scenario-count").textContent = `${share.above} of ${share.total}`;
    const percent = Number(share.above_share_percent).toFixed(1);
    const step = Number(share.step_percentage_points);
    el("scenario-text").textContent = `${percent}% of these ${share.total} draws end above the last close. The share moves in steps of ${step} points. One model, repeated. Not a chance of profit.`;
  }

  function renderContext() {
    renderHigher(state.context);
    renderPerpetual(state.context);
  }

  function renderHigher(context) {
    const status = el("higher-status");
    const body = el("higher-body");
    if (!status || !body) return;
    const rows = (context && context.higher_timeframes) || [];
    if (!rows.length) {
      status.textContent = "Waiting";
      body.innerHTML = "<p>Load candles to compare slower timeframes.</p>";
      return;
    }
    const ready = rows.filter((item) => item.status === "ok").length;
    status.textContent = ready ? `${ready} aligned` : "Unavailable";
    body.innerHTML = rows.map((item) => {
      const name = item.interval_label || item.interval || "Higher timeframe";
      if (item.status !== "ok") {
        return `<div class="tf-row"><strong>${escapeHtml(name)}</strong><p class="status-bad">${escapeHtml(item.reason || "No aligned candles.")}</p></div>`;
      }
      const change = item.last_bar_return;
      const changeText = change == null ? "—" : `${change > 0 ? "+" : ""}${(change * 100).toFixed(2)}%`;
      const changeClass = change == null ? "chg" : change >= 0 ? "chg up" : "chg down";
      const vol = item.log_return_volatility == null ? null : `${(item.log_return_volatility * 100).toFixed(2)}%`;
      const volume = item.last_volume_over_median == null
        ? (item.volume_note || "Volume was not compared.")
        : `Volume on that bar was ${Number(item.last_volume_over_median).toFixed(2)}× the median of this window.`;
      const direction = plainDirection(item.sign_vs_base_last_bar);
      return `<div class="tf-row"><div><strong>${escapeHtml(name)}</strong><span class="${changeClass}">${changeText}</span></div><p>Last closed bar. ${vol ? `Typical bar-to-bar move ${vol}. ` : ""}${escapeHtml(volume)} ${escapeHtml(direction)}</p><details><summary>Time and source</summary><p>${escapeHtml(item.source || "Closed candles")}. ${item.bars} bars through ${escapeHtml(formatUtc(item.as_of))}.${item.last_close_time ? ` Last bar closed ${escapeHtml(formatUtc(item.last_close_time))}.` : ""}</p></details></div>`;
    }).join("");
  }

  function plainDirection(value) {
    if (value === "same sign on the last closed bar") return "The last base bar moved the same way.";
    if (value === "opposite sign on the last closed bar") return "The last base bar moved the other way.";
    if (value === "one bar is unchanged") return "One of those bars was unchanged.";
    return "Direction was not compared.";
  }

  function renderPerpetual(context) {
    const status = el("perp-status");
    const body = el("perp-body");
    if (!status || !body) return;
    const deriv = (context && context.derivatives) || null;
    const spot = context && context.spot_taker;
    if (!deriv) {
      status.textContent = "Waiting";
      body.innerHTML = "<p>Load a Binance spot symbol to ask the public USD-M endpoints.</p>";
      return;
    }
    if (deriv.status === "not_applicable") {
      status.textContent = "Not used";
      body.innerHTML = `<p>${escapeHtml(deriv.reason || "This file was not sent to a futures host.")}</p>${spotBlock(spot)}`;
      return;
    }
    if (deriv.status !== "ok") {
      status.textContent = "Unavailable";
      const when = deriv.checked_at ? ` Checked ${formatUtc(deriv.checked_at, true)}.` : "";
      body.innerHTML = `<p class="status-bad">${escapeHtml(deriv.reason || "The futures host did not return data.")}${escapeHtml(when)} Nothing was filled in.</p>${spotBlock(spot)}`;
      return;
    }
    const fund = deriv.funding && deriv.funding.latest;
    const oi = deriv.open_interest && deriv.open_interest.latest;
    const taker = deriv.taker && deriv.taker.latest;
    status.textContent = "Aligned";
    const fundLine = fund
      ? `Funding ${(fund.funding_rate * 100).toFixed(4)}% per interval at ${formatUtc(fund.time)}.`
      : "No funding record was at or before this candle close.";
    const oiLine = oi
      ? `Open interest ${formatPrice(oi.open_interest_contracts)} contracts${oi.open_interest_value_usdt != null ? `, about ${formatPrice(oi.open_interest_value_usdt)} USDT notional` : ""} at ${formatUtc(oi.time)}.`
      : "No open-interest history point was at or before this candle close.";
    const takerLine = taker && taker.taker_buy_share != null
      ? `Taker buys were ${(taker.taker_buy_share * 100).toFixed(1)}% of taker volume on the last ${taker.period} perpetual bar (${formatUtc(taker.time)}).`
      : "No perpetual taker bar was at or before this candle close.";
    const after = deriv.after_as_of
      ? `<p>A later open-interest snapshot exists at ${formatUtc(deriv.after_as_of.time)} (${formatPrice(deriv.after_as_of.open_interest_contracts)} contracts). It is after the candle close, so it is not treated as known then.</p>`
      : "";
    body.innerHTML = `<p>${fundLine}</p><p>${oiLine}</p><p>${takerLine}</p>${after}<details><summary>Units and source</summary><p>Host fapi.binance.com. Funding raw fraction ${fund ? fund.funding_rate : "—"}. Open interest is in contracts, and USDT notional only when Binance sent it. Taker figures are executed buy and sell volume, not an order book. ${escapeHtml((deriv.perpetual && deriv.perpetual.note) || "")}</p></details>${spotBlock(spot)}`;
  }

  function spotBlock(spot) {
    if (!spot || spot.status === "absent") {
      return `<p>${escapeHtml((spot && spot.reason) || "This spot response had no taker-buy field.")}</p>`;
    }
    if (spot.status !== "ok") return "";
    return `<p>On the spot candles, taker buys were ${(spot.last_taker_buy_share_of_quote * 100).toFixed(1)}% of quote volume in the last bar, and ${(spot.recent_mean_taker_buy_share_of_quote * 100).toFixed(1)}% on average over ${spot.bars} bars.</p>`;
  }

  function amountLabel(policy) {
    if (policy === "provided") return "Present in the source";
    if (policy === "filled_by_model_from_volume") return "Model fills it from volume";
    return "Model fills volume and amount with zero";
  }

  function renderLog() {
    const list = el("log");
    if (!state.log.length) {
      list.innerHTML = "<li>No requests yet.</li>";
      return;
    }
    list.innerHTML = state.log.map((event) => `<li><time>${clock(event.time)}</time>${escapeHtml(event.message)}</li>`).join("");
    list.scrollTop = list.scrollHeight;
  }

  function renderLegend() {
    const items = [
      ["up", "Up close"],
      ["down", "Down close"],
    ];
    if (state.dataset && state.dataset.forming) items.push(["forming", "Forming"]);
    if (state.forecast) items.push(["median", state.forecast.sample_count > 1 ? "Median close" : "Sample close"]);
    if (state.forecast) items.push(["forecast", "Forecast candles"]);
    if (state.showPaths && state.forecast && state.forecast.sample_count > 1) {
      items.push(["sample", `${state.forecast.traces_shown || state.forecast.sample_count} paths`]);
    }
    if (state.showBand && state.forecast && state.forecast.band) items.push(["sample", "Dispersion"]);
    el("legend").innerHTML = items.map(([kind, label]) => `<li><i class="swatch ${kind}"></i>${escapeHtml(label)}</li>`).join("");
    const note = el("legend-note");
    if (!note) return;
    if (!state.forecast) {
      note.textContent = "The space right of the line is the forecast region. It stays empty until a run finishes.";
    } else if (state.forecast.sample_count > 1) {
      note.textContent = state.showPaths
        ? `The pale lines are ${state.forecast.traces_shown || state.forecast.sample_count} real paths. The median line stays the middle close. It is not a probability the price finishes there.`
        : `Median line is the middle close of ${state.forecast.sample_count} real paths at each step. It is not a probability the price finishes there. Paths stay hidden until Show paths.`;
    } else {
      note.textContent = "One sampled path. There is no second line and no dispersion band.";
    }
    const paths = el("show-paths");
    const band = el("show-band");
    if (paths) {
      paths.setAttribute("aria-pressed", state.showPaths ? "true" : "false");
      paths.disabled = !(state.forecast && state.forecast.sample_count > 1);
    }
    if (band) {
      band.setAttribute("aria-pressed", state.showBand ? "true" : "false");
      band.disabled = !(state.forecast && state.forecast.band);
    }
  }

  function factRows(rows) {
    return rows.map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd>`).join("");
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[char]));
  }

  function applyDataset(payload, banner) {
    state.dataset = payload.dataset;
    state.candles = payload.candles || [];
    state.context = payload.context || null;
    state.forecast = null;
    state.jobId = null;
    if (payload.log) state.log = payload.log;
    setBanner(null, "");
    render();
  }

  async function withBusy(kind, action) {
    state.busy = kind;
    render();
    try {
      return await action();
    } finally {
      state.busy = null;
      render();
    }
  }

  async function loadBinance(force) {
    const symbol = el("symbol").value.trim();
    if (!symbol) {
      setBanner("error", "Enter a spot symbol, for example BTCUSDT.");
      return;
    }
    await withBusy("data", async () => {
      const { lookback, predLen } = numbers();
      const payload = await api("/api/datasets/binance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          interval: el("interval").value,
          lookback,
          pred_len: predLen,
          force: Boolean(force),
        }),
      });
      const runtime = await api("/api/runtime");
      state.runtime = runtime;
      state.log = runtime.log;
      applyDataset(payload);
    });
  }

  async function loadSample() {
    await withBusy("data", async () => {
      const payload = await api("/api/datasets/sample", { method: "POST" });
      const runtime = await api("/api/runtime");
      state.log = runtime.log;
      state.runtime = runtime;
      applyDataset(payload);
    });
  }

  async function loadCsv(file) {
    if (!file) return;
    const body = new FormData();
    body.append("file", file);
    await withBusy("data", async () => {
      const payload = await api("/api/datasets/csv", { method: "POST", body });
      const runtime = await api("/api/runtime");
      state.log = runtime.log;
      state.runtime = runtime;
      applyDataset(payload);
    });
  }

  async function loadLocal(id) {
    if (!id) return;
    await withBusy("data", async () => {
      const payload = await api("/api/datasets/local", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
      });
      const runtime = await api("/api/runtime");
      state.log = runtime.log;
      state.runtime = runtime;
      applyDataset(payload);
    });
  }

  async function loadModel() {
    await withBusy("model", async () => {
      const payload = await api("/api/model/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_key: el("model").value, device: el("device").value }),
      });
      state.log = payload.log;
      const runtime = await api("/api/runtime");
      state.runtime = runtime;
      state.log = runtime.log;
      setBanner(null, "");
    });
  }

  async function runForecast() {
    if (!state.dataset) {
      setBanner("error", "No dataset is loaded. Load Binance candles, the official sample, or a CSV file first.");
      return;
    }
    const check = validateClient();
    if (check.problems.length) {
      setBanner("error", check.problems[0]);
      return;
    }
    await withBusy("forecast", async () => {
      const { lookback, predLen, samples, temperature, topP, mode } = numbers();
      const started = await api("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: state.dataset.id,
          lookback,
          pred_len: predLen,
          sample_count: samples,
          temperature,
          top_p: topP,
          mode,
        }),
      });
      state.jobId = started.job_id;
      state.progressText = `Sampled 0 of ${started.total}.`;
      let job = started;
      while (job.status === "running") {
        await wait(400);
        job = await api(`/api/forecast/jobs/${started.job_id}`);
        state.progressText = `Sampled ${job.completed} of ${job.total}.`;
        state.log = job.log || state.log;
        render();
      }
      state.log = job.log || state.log;
      if (job.status === "done" && job.result) {
        state.forecast = job.result;
        setBanner(null, "");
      } else if (job.status === "cancelled") {
        setBanner("ok", job.error || "Cancelled. The chart was not updated with a partial fan.");
      } else {
        throw new Error(job.error || "The forecast failed.");
      }
    });
  }

  function togglePaths() {
    if (!(state.forecast && state.forecast.sample_count > 1)) return;
    state.showPaths = !state.showPaths;
    storeFlag("kronos-show-paths", state.showPaths);
    render();
  }

  function toggleBand() {
    if (!(state.forecast && state.forecast.band)) return;
    state.showBand = !state.showBand;
    storeFlag("kronos-show-band", state.showBand);
    render();
  }

  function wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function cancelForecast() {
    if (!state.jobId) return;
    await api(`/api/forecast/jobs/${state.jobId}/cancel`, { method: "POST" });
  }

  async function runCheck() {
    await withBusy("check", async () => {
      const report = await api("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoints: 2, lookback: numbers().lookback, pred_len: Math.min(24, numbers().predLen) }),
      });
      state.log = report.log || state.log;
      renderCheck(report);
    });
  }

  function renderCheck(report) {
    const root = el("check-result");
    if (!root || !report) return;
    const windows = (report.checkpoints || []).length;
    const horizon = report.settings ? report.settings.pred_len : "—";
    const modelError = report.kronos_median_mae;
    const baseError = report.persistence_mae;
    let comparison = "On this slice the two errors matched.";
    if (modelError < baseError) comparison = "On this slice the model error was lower than repeating the last close.";
    else if (modelError > baseError) comparison = "On this slice repeating the last close had the lower error.";
    root.innerHTML = `<p>${windows} windows, horizon ${horizon} candles, one sample each. Model mean absolute error ${formatPrice(modelError)} price units. Last-close baseline ${formatPrice(baseError)} price units. ${escapeHtml(comparison)}</p><p>${escapeHtml(report.training_cutoff || "")}</p>`;
  }

  async function exportRecord() {
    if (!state.jobId) {
      setBanner("error", "Finish a forecast before downloading its record.");
      return;
    }
    const record = await api(`/api/forecast/jobs/${state.jobId}/record`);
    const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "kronos-forecast-record.json";
    link.click();
    URL.revokeObjectURL(url);
  }

  function setSource(source) {
    state.source = source;
    ["binance", "sample", "csv"].forEach((name) => {
      const button = el(`source-${name}`);
      button.setAttribute("aria-checked", name === source ? "true" : "false");
      el(`panel-${name}`).hidden = name !== source;
    });
  }

  function fillRuntime(runtime) {
    state.runtime = runtime;
    state.log = runtime.log || [];
    const symbols = el("symbols");
    symbols.innerHTML = runtime.popular_symbols.map((symbol) => `<option value="${symbol}"></option>`).join("");
    const interval = el("interval");
    interval.innerHTML = runtime.intervals.map((item) => {
      const selected = item.code === runtime.defaults.interval ? " selected" : "";
      return `<option value="${item.code}"${selected}>${item.code} · ${item.label}</option>`;
    }).join("");
    const model = el("model");
    model.innerHTML = runtime.models.map((item) => {
      const selected = item.key === runtime.defaults.model_key ? " selected" : "";
      return `<option value="${item.key}"${selected}>${item.name} · ${item.params}</option>`;
    }).join("");
    const device = el("device");
    device.innerHTML = runtime.devices.options.map((option) => `<option value="${option}">${option}</option>`).join("");
    el("device-reason").textContent = runtime.devices.auto_reason;
    el("lookback").value = runtime.defaults.lookback;
    el("horizon").value = runtime.defaults.pred_len;
    el("samples").value = runtime.defaults.sample_count;
    el("temperature").value = runtime.defaults.temperature;
    el("top-p").value = runtime.defaults.top_p;
    el("temperature-value").textContent = Number(runtime.defaults.temperature).toFixed(1);
    el("top-p-value").textContent = Number(runtime.defaults.top_p).toFixed(2);
  }

  async function refreshLocalFiles() {
    const payload = await api("/api/datasets/local");
    const select = el("local-file");
    const current = select.value;
    select.innerHTML = `<option value="">None selected</option>` + payload.files
      .filter((file) => file.kind !== "sample")
      .map((file) => `<option value="${escapeHtml(file.id)}">${escapeHtml(file.label)}</option>`)
      .join("");
    select.value = current;
  }

  function bind() {
    el("banner-dismiss").addEventListener("click", () => setBanner(null, ""));
    ["binance", "sample", "csv"].forEach((name) => {
      el(`source-${name}`).addEventListener("click", () => setSource(name));
    });
    el("load-binance").addEventListener("click", () => guard(loadBinance(false)));
    el("refresh").addEventListener("click", () => guard(loadBinance(true)));
    el("load-sample").addEventListener("click", () => guard(loadSample()));
    el("load-model").addEventListener("click", () => guard(loadModel()));
    el("run").addEventListener("click", () => guard(runForecast()));
    el("cancel").addEventListener("click", () => guard(cancelForecast()));
    el("run-check").addEventListener("click", () => guard(runCheck()));
    el("export-record").addEventListener("click", () => guard(exportRecord()));
    el("show-paths").addEventListener("click", togglePaths);
    el("show-band").addEventListener("click", toggleBand);
    el("fit-forecast").addEventListener("click", () => {
      chart.pan = 0;
      chart.fitted = true;
      chart.draw();
    });
    document.querySelectorAll("[data-layout-choice]").forEach((button) => {
      button.addEventListener("click", () => setLayout(button.getAttribute("data-layout-choice")));
    });
    el("toggle-controls").addEventListener("click", () => {
      const open = el("rail").classList.toggle("is-open");
      el("toggle-controls").setAttribute("aria-expanded", open ? "true" : "false");
    });
    ["fold-higher", "fold-perp", "fold-check"].forEach((id) => bindDisclosure(id, `kronos-${id}`));
    el("csv-file").addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      guard(loadCsv(file)).finally(() => {
        event.target.value = "";
      });
    });
    el("local-file").addEventListener("change", (event) => {
      const id = event.target.value;
      if (!id) return;
      guard(loadLocal(id));
    });
    el("temperature").addEventListener("input", (event) => {
      el("temperature-value").textContent = Number(event.target.value).toFixed(1);
    });
    el("top-p").addEventListener("input", (event) => {
      el("top-p-value").textContent = Number(event.target.value).toFixed(2);
    });
    ["lookback", "horizon", "samples", "model", "mode", "device"].forEach((id) => {
      el(id).addEventListener("change", render);
      el(id).addEventListener("input", render);
    });
    document.querySelector(".segment").addEventListener("keydown", (event) => {
      const order = ["binance", "sample", "csv"];
      const index = order.indexOf(state.source);
      if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        setSource(order[(index + 1) % order.length]);
        el(`source-${state.source}`).focus();
        event.preventDefault();
      } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        setSource(order[(index + order.length - 1) % order.length]);
        el(`source-${state.source}`).focus();
        event.preventDefault();
      }
    });
    window.addEventListener("resize", () => chart.draw());
  }

  function guard(promise) {
    return Promise.resolve(promise).catch((error) => {
      setBanner("error", error.message || "The request failed.");
      render();
    });
  }

  function PriceChart(canvas) {
    this.canvas = canvas;
    this.candlePx = 8;
    this.pan = 0;
    this.fitted = true;
    this.cursor = null;
    this.drag = null;
    this.data = { candles: [], forming: null, forecast: null, horizon: 24 };
    const self = this;
    canvas.addEventListener("mousemove", (event) => self.hover(event));
    canvas.addEventListener("mouseleave", () => { self.cursor = null; self.draw(); });
    canvas.addEventListener("wheel", (event) => self.zoom(event), { passive: false });
    canvas.addEventListener("pointerdown", (event) => self.down(event));
    canvas.addEventListener("pointermove", (event) => self.move(event));
    canvas.addEventListener("pointerup", () => { self.drag = null; });
    canvas.addEventListener("keydown", (event) => self.key(event));
    if (window.ResizeObserver) {
      const observer = new ResizeObserver(() => self.draw());
      observer.observe(canvas);
    }
  }

  PriceChart.prototype.setData = function (data) {
    const previous = this.data || {};
    const sameSeries = previous.candles === data.candles && previous.forecast === data.forecast && previous.forming === data.forming;
    this.data = data;
    if (!sameSeries) {
      this.pan = 0;
      this.cursor = null;
      this.fitted = true;
    }
    this.draw();
  };

  PriceChart.prototype.layout = function () {
    const canvas = this.canvas;
    const width = canvas.clientWidth || 640;
    const height = canvas.clientHeight || 420;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width, height };
  };

  PriceChart.prototype.scene = function (width) {
    const observed = this.data.candles || [];
    const forecast = this.data.forecast;
    const forming = this.data.forming;
    const pad = { l: 10, r: 78, t: 18, b: 22 };
    const plotW = Math.max(40, width - pad.l - pad.r);
    const forward = forecast && forecast.mode === "forward" ? forecast.path : [];
    const futureN = forward.length;
    const reserve = futureN || Math.max(10, Math.min(28, this.data.horizon || 24));
    let pastCount;
    let slot;
    if (this.fitted) {
      const pastTarget = Math.max(20, Math.round(reserve * 0.66 / 0.34));
      pastCount = Math.min(observed.length, pastTarget);
      slot = plotW / Math.max(1, pastCount + reserve);
    } else {
      slot = this.candlePx + Math.max(2, this.candlePx * 0.42);
      const capacity = Math.max(reserve + 8, Math.floor(plotW / slot));
      pastCount = Math.min(observed.length, Math.max(8, capacity - reserve));
    }
    this._slot = slot;
    const end = Math.max(0, observed.length - this.pan);
    const start = Math.max(0, end - pastCount);
    const visible = observed.slice(start, end);
    const bars = visible.map((candle, i) => {
      const held = Boolean(forecast && forecast.mode === "held_out" && candle.time > forecast.boundary_open);
      return {
        key: candle.time,
        index: start + i,
        observed: held ? null : candle,
        actual: held ? candle : null,
        forecast: null,
        forming: null,
        samples: null,
      };
    });
    if (forecast && forecast.mode === "held_out") {
      forecast.path.forEach((point, index) => {
        const host = bars.find((bar) => bar.key === point.time);
        if (!host) return;
        host.forecast = point;
        host.samples = state.showPaths ? sampleAt(forecast, index) : null;
        host.q50 = bandAt(forecast, index, "q50");
        host.q10 = state.showBand ? bandAt(forecast, index, "q10") : null;
        host.q90 = state.showBand ? bandAt(forecast, index, "q90") : null;
        host.q25 = state.showBand ? bandAt(forecast, index, "q25") : null;
        host.q75 = state.showBand ? bandAt(forecast, index, "q75") : null;
      });
    }
    const futureBars = [];
    if (forward.length) {
      forward.forEach((point, index) => {
        futureBars.push({
          key: point.time,
          index: 100000 + index,
          observed: null,
          actual: null,
          forecast: point,
          forming: forming && index === 0 && forming.time === point.time ? forming : null,
          samples: state.showPaths ? sampleAt(forecast, index) : null,
          q50: bandAt(forecast, index, "q50"),
          q10: state.showBand ? bandAt(forecast, index, "q10") : null,
          q90: state.showBand ? bandAt(forecast, index, "q90") : null,
          q25: state.showBand ? bandAt(forecast, index, "q25") : null,
          q75: state.showBand ? bandAt(forecast, index, "q75") : null,
        });
      });
    } else if (forming) {
      futureBars.push({
        key: forming.time,
        index: -1,
        observed: null,
        actual: null,
        forecast: null,
        forming,
        samples: null,
      });
    }
    const heldOut = Boolean(forecast && forecast.mode === "held_out");
    const all = heldOut ? bars : bars.concat(futureBars);
    all.forEach((bar, i) => {
      bar.x = pad.l + i * slot + slot / 2;
      bar.slot = slot;
    });
    const boundaryIndex = heldOut ? all.findIndex((bar) => bar.actual || bar.forecast) : bars.length;
    const boundaryX = boundaryIndex < 0 ? pad.l : pad.l + boundaryIndex * slot;
    return { pad, bars: all, boundaryX, forecast };
  };

  PriceChart.prototype.draw = function () {
    const { ctx, width, height } = this.layout();
    const scene = this.scene(width);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#061012";
    ctx.fillRect(0, 0, width, height);
    const glow = ctx.createRadialGradient(width * 0.74, height * 0.42, 12, width * 0.72, height * 0.48, width * 0.48);
    glow.addColorStop(0, "rgba(24, 110, 112, 0.16)");
    glow.addColorStop(1, "rgba(6, 16, 18, 0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, width, height);

    if (!(this.data.candles || []).length) {
      ctx.fillStyle = "#93aeb4";
      ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("Load candles to draw the chart. Nothing is drawn from invented prices.", 24, height / 2);
      el("readout").textContent = "";
      return;
    }

    const volH = Math.max(34, Math.round(height * 0.15));
    const priceTop = scene.pad.t;
    const priceBottom = height - scene.pad.b - volH - 10;
    ctx.fillStyle = "rgba(94, 224, 208, 0.04)";
    ctx.fillRect(scene.boundaryX, priceTop, width - scene.boundaryX - scene.pad.r + 6, priceBottom - priceTop);
    const range = priceRange(scene.bars);
    drawGrid(ctx, scene, width, priceTop, priceBottom, range);
    if (state.showBand) drawBand(ctx, scene.bars, priceTop, priceBottom, range);
    scene.bars.forEach((bar) => {
      if (bar.actual) drawCandle(ctx, bar.x, bar.slot, bar.actual, priceTop, priceBottom, range, "actual");
      if (bar.observed) drawCandle(ctx, bar.x, bar.slot, bar.observed, priceTop, priceBottom, range, "observed");
      if (bar.forecast) drawCandle(ctx, bar.x, bar.slot, bar.forecast, priceTop, priceBottom, range, "forecast");
      if (bar.forming) drawForming(ctx, bar.x, bar.slot, bar.forming, priceTop, priceBottom, range);
    });
    if (state.showPaths) drawSampleLines(ctx, scene.bars, priceTop, priceBottom, range);
    drawMedianLine(ctx, scene.bars, priceTop, priceBottom, range);
    drawVolume(ctx, scene.bars, priceBottom + 8, height - scene.pad.b);
    drawBoundary(ctx, scene.boundaryX, priceTop, height - scene.pad.b, scene.forecast);
    drawPriceAxis(ctx, width, scene.pad.r, priceTop, priceBottom, range, scene.bars);
    drawTimeAxis(ctx, scene.bars, height - 6);
    this.paintReadout(scene.bars);
  };

  PriceChart.prototype.paintReadout = function (bars) {
    const readout = el("readout");
    const usable = bars.filter((bar) => bar.observed || bar.actual || bar.forecast || bar.forming);
    if (!usable.length) {
      readout.textContent = "";
      return;
    }
    const bar = this.cursor != null ? bars.find((item) => item.index === this.cursor) || usable[usable.length - 1] : usable[usable.length - 1];
    const candle = bar.forecast || bar.observed || bar.actual || bar.forming;
    if (!candle) {
      readout.textContent = "";
      return;
    }
    const kind = bar.forecast ? "Forecast" : bar.forming && !bar.observed ? "Forming" : bar.actual ? "Later candle" : "Observed";
    readout.textContent = `${kind} ${formatUtc(candle.time)}  O ${formatPrice(candle.open)}  H ${formatPrice(candle.high)}  L ${formatPrice(candle.low)}  C ${formatPrice(candle.close)}`;
  };

  PriceChart.prototype.eventX = function (event) {
    const rect = this.canvas.getBoundingClientRect();
    return event.clientX - rect.left;
  };

  PriceChart.prototype.hover = function (event) {
    const scene = this.scene(this.canvas.clientWidth || 640);
    const x = this.eventX(event);
    let best = null;
    let bestDist = 1e9;
    scene.bars.forEach((bar) => {
      const dist = Math.abs(bar.x - x);
      if (dist < bestDist) {
        best = bar;
        bestDist = dist;
      }
    });
    this.cursor = best ? best.index : null;
    this.draw();
  };

  PriceChart.prototype.zoom = function (event) {
    event.preventDefault();
    this.leaveFit();
    const next = this.candlePx + (event.deltaY > 0 ? -1 : 1);
    this.candlePx = Math.max(4, Math.min(16, next));
    this.draw();
  };

  PriceChart.prototype.leaveFit = function () {
    if (!this.fitted) return;
    const slot = this._slot || 12;
    this.candlePx = Math.max(4, Math.min(16, slot * 0.62));
    this.fitted = false;
  };

  PriceChart.prototype.down = function (event) {
    this.canvas.setPointerCapture(event.pointerId);
    this.drag = { x: event.clientX, pan: this.pan };
  };

  PriceChart.prototype.move = function (event) {
    if (!this.drag) return;
    this.leaveFit();
    const slot = this._slot || (this.candlePx + Math.max(2, this.candlePx * 0.45));
    const delta = Math.round((this.drag.x - event.clientX) / slot);
    const maxPan = Math.max(0, (this.data.candles || []).length - 1);
    this.pan = Math.max(0, Math.min(maxPan, this.drag.pan + delta));
    this.draw();
  };

  PriceChart.prototype.key = function (event) {
    const candles = this.data.candles || [];
    if (event.key === "Home" || event.key === "f" || event.key === "F") {
      this.pan = 0;
      this.fitted = true;
      this.draw();
      event.preventDefault();
      return;
    }
    if (event.key === "p" || event.key === "P") {
      togglePaths();
      event.preventDefault();
      return;
    }
    if (event.key === "b" || event.key === "B") {
      toggleBand();
      event.preventDefault();
      return;
    }
    if (event.key === "Escape") {
      this.pan = 0;
      this.fitted = true;
      this.draw();
      event.preventDefault();
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      if (event.shiftKey) {
        const maxPan = Math.max(0, candles.length - 1);
        this.pan = Math.max(0, Math.min(maxPan, this.pan + (event.key === "ArrowLeft" ? 4 : -4)));
      } else if (candles.length) {
        const current = this.cursor == null ? candles.length - 1 : this.cursor;
        const next = event.key === "ArrowLeft" ? Math.max(0, current - 1) : Math.min(candles.length - 1, current + 1);
        this.cursor = next;
      }
      this.draw();
      event.preventDefault();
    }
  };

  function bandAt(forecast, index, key) {
    if (!forecast || !Array.isArray(forecast.band) || !forecast.band[index]) return null;
    const value = forecast.band[index][key];
    return Number.isFinite(value) ? value : null;
  }

  function meanAt(forecast, index) {
    if (!forecast || !Array.isArray(forecast.mean_path) || !forecast.mean_path[index]) return null;
    const value = forecast.mean_path[index].close;
    return Number.isFinite(value) ? value : null;
  }

  function sampleAt(forecast, index) {
    if (!forecast || forecast.sample_count < 2 || !Array.isArray(forecast.sample_closes)) return null;
    const closes = forecast.sample_closes.map((series) => series[index] && series[index].close);
    return closes.every((value) => Number.isFinite(value)) ? closes : null;
  }

  function priceRange(bars) {
    let low = Infinity;
    let high = -Infinity;
    bars.forEach((bar) => {
      [bar.observed, bar.actual, bar.forecast, bar.forming].forEach((candle) => {
        if (!candle) return;
        low = Math.min(low, candle.low);
        high = Math.max(high, candle.high);
      });
      (bar.samples || []).forEach((close) => {
        low = Math.min(low, close);
        high = Math.max(high, close);
      });
      [bar.q10, bar.q90, bar.q50].forEach((value) => {
        if (Number.isFinite(value)) {
          low = Math.min(low, value);
          high = Math.max(high, value);
        }
      });
    });
    if (!Number.isFinite(low) || !Number.isFinite(high) || low === high) {
      low = Number.isFinite(low) ? low * 0.99 : 0;
      high = Number.isFinite(high) ? high * 1.01 : 1;
    }
    const pad = (high - low) * 0.08;
    return { low: low - pad, high: high + pad };
  }

  function yOf(price, top, bottom, range) {
    const span = range.high - range.low || 1;
    return top + (range.high - price) / span * (bottom - top);
  }

  function drawGrid(ctx, model, width, top, bottom, range) {
    ctx.save();
    ctx.strokeStyle = "rgba(146, 196, 192, 0.08)";
    ctx.lineWidth = 1;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    ctx.fillStyle = "#6f8b92";
    const steps = 4;
    for (let i = 0; i <= steps; i += 1) {
      const price = range.low + (range.high - range.low) * (i / steps);
      const y = yOf(price, top, bottom, range);
      ctx.beginPath();
      ctx.moveTo(model.pad.l, y);
      ctx.lineTo(width - model.pad.r, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawCandle(ctx, x, slot, candle, top, bottom, range, role) {
    const up = candle.close >= candle.open;
    const color = up ? "#1fbf86" : "#f25d6c";
    const yHigh = yOf(candle.high, top, bottom, range);
    const yLow = yOf(candle.low, top, bottom, range);
    const yOpen = yOf(candle.open, top, bottom, range);
    const yClose = yOf(candle.close, top, bottom, range);
    const bodyTop = Math.min(yOpen, yClose);
    const bodyH = Math.max(1, Math.abs(yClose - yOpen));
    const width = Math.max(3, slot * 0.62);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, yHigh);
    ctx.lineTo(x, yLow);
    ctx.stroke();
    if (role === "forecast") {
      ctx.globalAlpha = 0.72;
      ctx.fillStyle = up ? "rgba(31, 191, 134, 0.7)" : "rgba(242, 93, 108, 0.65)";
      ctx.strokeStyle = "rgba(210, 228, 224, 0.55)";
      ctx.strokeRect(x - width / 2, bodyTop, width, bodyH);
      ctx.fillRect(x - width / 2, bodyTop, width, bodyH);
    } else if (role === "actual") {
      ctx.strokeStyle = "#e6c36a";
      ctx.strokeRect(x - width / 2, bodyTop, width, bodyH);
    } else {
      ctx.fillStyle = color;
      ctx.fillRect(x - width / 2, bodyTop, width, bodyH);
    }
    ctx.restore();
  }

  function drawForming(ctx, x, slot, candle, top, bottom, range) {
    const yHigh = yOf(candle.high, top, bottom, range);
    const yLow = yOf(candle.low, top, bottom, range);
    const yOpen = yOf(candle.open, top, bottom, range);
    const yClose = yOf(candle.close, top, bottom, range);
    const bodyTop = Math.min(yOpen, yClose);
    const bodyH = Math.max(1, Math.abs(yClose - yOpen));
    const width = Math.max(3, slot * 0.62);
    ctx.save();
    ctx.setLineDash([3, 2]);
    ctx.strokeStyle = "#93aeb4";
    ctx.beginPath();
    ctx.moveTo(x, yHigh);
    ctx.lineTo(x, yLow);
    ctx.stroke();
    ctx.strokeRect(x - width / 2, bodyTop, width, bodyH);
    ctx.restore();
  }

  function drawBand(ctx, bars, top, bottom, range) {
    const points = bars.filter((bar) => Number.isFinite(bar.q10) && Number.isFinite(bar.q90));
    if (points.length < 2) return;
    const paint = (lowKey, highKey, color) => {
      ctx.beginPath();
      points.forEach((bar, index) => {
        const y = yOf(bar[highKey], top, bottom, range);
        if (index === 0) ctx.moveTo(bar.x, y);
        else ctx.lineTo(bar.x, y);
      });
      for (let index = points.length - 1; index >= 0; index -= 1) {
        ctx.lineTo(points[index].x, yOf(points[index][lowKey], top, bottom, range));
      }
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    };
    ctx.save();
    paint("q25", "q75", "rgba(94, 224, 208, 0.08)");
    ctx.restore();
  }

  function drawMedianLine(ctx, bars, top, bottom, range) {
    const points = bars.filter((bar) => Number.isFinite(bar.q50) || (bar.forecast && Number.isFinite(bar.forecast.close)));
    if (points.length < 2) return;
    ctx.save();
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = "#f6e7a8";
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    points.forEach((bar, index) => {
      const price = Number.isFinite(bar.q50) ? bar.q50 : bar.forecast.close;
      const y = yOf(price, top, bottom, range);
      if (index === 0) ctx.moveTo(bar.x, y);
      else ctx.lineTo(bar.x, y);
    });
    ctx.stroke();
    ctx.restore();
  }

  function drawSampleLines(ctx, bars, top, bottom, range) {
    const count = bars.reduce((max, bar) => Math.max(max, bar.samples ? bar.samples.length : 0), 0);
    if (!count) return;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.38;
    for (let series = 0; series < count; series += 1) {
      ctx.strokeStyle = "rgba(176, 214, 208, 0.95)";
      ctx.beginPath();
      let started = false;
      bars.forEach((bar) => {
        if (!bar.samples || !Number.isFinite(bar.samples[series])) return;
        const y = yOf(bar.samples[series], top, bottom, range);
        if (!started) {
          ctx.moveTo(bar.x, y);
          started = true;
        } else {
          ctx.lineTo(bar.x, y);
        }
      });
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawVolume(ctx, bars, top, bottom) {
    let max = 0;
    bars.forEach((bar) => {
      const candle = bar.observed || bar.actual || bar.forecast || bar.forming;
      if (candle && candle.volume > max) max = candle.volume;
    });
    if (max <= 0) return;
    bars.forEach((bar) => {
      const candle = bar.forecast || bar.observed || bar.actual || bar.forming;
      if (!candle || candle.volume == null) return;
      const h = (candle.volume / max) * (bottom - top);
      ctx.fillStyle = candle.close >= candle.open ? "rgba(31, 191, 134, 0.28)" : "rgba(242, 93, 108, 0.28)";
      const width = Math.max(2, (bar.slot || 8) * 0.5);
      ctx.fillRect(bar.x - width / 2, bottom - h, width, h);
    });
  }

  function drawBoundary(ctx, x, top, bottom, forecast) {
    ctx.save();
    ctx.strokeStyle = "#5ee0d0";
    ctx.lineWidth = 1;
    ctx.shadowColor = "rgba(94, 224, 208, 0.45)";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.shadowBlur = 0;
    const label = forecast && forecast.mode === "held_out" ? "START" : "NOW";
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = "#102826";
    ctx.strokeStyle = "#5ee0d0";
    ctx.fillRect(x - textW / 2 - 6, top + 4, textW + 12, 16);
    ctx.strokeRect(x - textW / 2 - 6, top + 4, textW + 12, 16);
    ctx.fillStyle = "#d8fffa";
    ctx.fillText(label, x - textW / 2, top + 16);
    if (!forecast) {
      ctx.fillStyle = "#6f8b92";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText("forecast region", x + 12, (top + bottom) / 2);
    }
    ctx.restore();
  }

  function drawPriceAxis(ctx, width, right, top, bottom, range, bars) {
    ctx.save();
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    ctx.fillStyle = "#93aeb4";
    const steps = 4;
    for (let i = 0; i <= steps; i += 1) {
      const price = range.low + (range.high - range.low) * (i / steps);
      const y = yOf(price, top, bottom, range);
      ctx.fillText(formatPrice(price), width - right + 8, y + 4);
    }
    const last = [...bars].reverse().find((bar) => bar.observed || bar.actual);
    const mark = last && (last.observed || last.actual);
    if (mark) {
      const y = yOf(mark.close, top, bottom, range);
      ctx.fillStyle = mark.close >= mark.open ? "#1fbf86" : "#f25d6c";
      ctx.fillRect(width - right + 4, y - 8, right - 10, 16);
      ctx.fillStyle = "#061012";
      ctx.fillText(formatPrice(mark.close), width - right + 8, y + 4);
    }
    ctx.restore();
  }

  function drawTimeAxis(ctx, bars, y) {
    const labeled = bars.filter((bar) => bar.observed || bar.actual || bar.forecast || bar.forming);
    if (!labeled.length) return;
    ctx.save();
    ctx.fillStyle = "#6f8b92";
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    const picks = [labeled[0], labeled[Math.floor(labeled.length / 2)], labeled[labeled.length - 1]];
    const seen = new Set();
    picks.forEach((bar) => {
      if (!bar || seen.has(bar.x)) return;
      seen.add(bar.x);
      const candle = bar.forecast || bar.observed || bar.actual || bar.forming;
      ctx.fillText(formatUtc(candle.time).replace(" UTC", ""), Math.max(8, bar.x - 48), y);
    });
    ctx.restore();
  }

  function setLayout(name) {
    if (name !== "focus" && name !== "workspace") return;
    document.body.dataset.layout = name;
    try { sessionStorage.setItem("kronos-layout", name); } catch (_error) { /* keep the choice for this view */ }
    document.querySelectorAll("[data-layout-choice]").forEach((button) => {
      button.setAttribute("aria-checked", button.getAttribute("data-layout-choice") === name ? "true" : "false");
    });
    if (name === "workspace") {
      el("rail").classList.remove("is-open");
      el("toggle-controls").setAttribute("aria-expanded", "false");
    }
  }

  function bindDisclosure(id, key) {
    const node = el(id);
    if (!node) return;
    try {
      const saved = sessionStorage.getItem(key);
      if (saved === "open") node.open = true;
      if (saved === "closed") node.open = false;
    } catch (_error) { /* default closed */ }
    node.addEventListener("toggle", () => {
      try { sessionStorage.setItem(key, node.open ? "open" : "closed"); } catch (_error) { /* ignore */ }
    });
  }

  function installUiTestHook() {
    if (!window.__KRONOS_UITEST) return;
    document.body.classList.add("uitest");
    const banner = el("uitest-banner");
    if (banner) banner.hidden = false;
    window.__kronosApplyFixture = function (payload) {
      state.dataset = payload.dataset;
      state.candles = payload.candles || [];
      state.context = payload.context || null;
      state.forecast = payload.forecast || null;
      state.log = payload.log || state.log;
      if (payload.model) {
        state.runtime = state.runtime || { model: { loaded: true, current: payload.model }, devices: { resolved_auto: "cpu" }, models: [] };
        state.runtime.model = { loaded: true, current: payload.model };
      }
      chart.fitted = true;
      chart.pan = 0;
      if (payload.check) renderCheck(payload.check);
      render();
    };
  }

  async function boot() {
    installUiTestHook();
    bind();
    setSource("binance");
    let savedLayout = "workspace";
    try { savedLayout = sessionStorage.getItem("kronos-layout") || "workspace"; } catch (_error) { savedLayout = "workspace"; }
    setLayout(savedLayout);
    try {
      const runtime = await api("/api/runtime");
      fillRuntime(runtime);
      await refreshLocalFiles();
      if (runtime.dataset) {
        const current = await api("/api/datasets/current");
        state.dataset = current.dataset;
        state.candles = current.candles || [];
      }
      render();
    } catch (error) {
      setBanner("error", error.message);
      render();
    }
  }

  boot();
})();
