// app.js — SSE client + panel wiring (training-dashboard.md). Plain ES6 module, no
// build step, no framework. Subscribes to GET /api/events, routes each message by
// `type` to the panel that owns it, and wires the three control buttons to
// POST /api/training/{start,stop,save}.
import { LossChart } from "./chart.js";

const $ = (id) => document.getElementById(id);
const fmt4 = (v) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(4) : "—");

// Defaults mirror constants.py; overwritten by GET /api/config at bootstrap so this
// file never silently drifts from the Python source of truth (decisions-auditor
// finding). The fetch is best-effort — if it fails, the page still renders with
// these values rather than breaking entirely.
const cfg = {
  grad_clip_norm: 1.0,
  grad_norm_instability_multiplier: 10.0,
  deploy_gate_da_threshold: 0.535,
  fold_table_da_amber_floor: 0.5,
  fold_table_qcov_healthy_lower: 0.85,
  fold_table_qcov_healthy_upper: 0.95,
  patience_amber_floor: 5,
  patience_orange_floor: 8,
  patience_glow_floor: 7,
  early_stopping_patience: 10,
  max_epochs: 100,
  loss_chart_ema_window: 5,
  loss_chart_visible_epochs: 200,
  batch_trace_max_points: 40,
  alert_auto_dismiss_seconds: 8.0,
  alert_max_stack: 3,
};

const state = {
  scenario: "idle",
  curFold: 0,
  totalFolds: 84, // placeholder until the first real epoch/batch message arrives
  curEpoch: 0,
  maxEpochs: cfg.max_epochs,
  bestEpoch: 0,
  bestVal: Infinity,
  patience: 0,
  maxPatience: cfg.early_stopping_patience,
  prevPinball: 0,
  prevDirection: 0,
  prevTotal: 0,
  pendingBtn: null,
  alertSeq: 0,
  foldRows: [], // newest first
  minValLoss: Infinity,
};

let chart = null;

// ---- header / status ----
function setStatus(s, text) {
  state.scenario = s;
  const dot = $("status-dot");
  dot.classList.remove("pulse");
  let color = "#666";
  if (s === "data-missing" || s === "error") color = "#e74c3c";
  else if (s === "running") {
    color = "#4a9eff";
    dot.classList.add("pulse");
  } else if (s === "saving") color = "#4a9eff";
  else if (s === "early-stopped") color = "#f39c12";
  else if (s === "stopped" || s === "completed") color = "#2ecc71";
  dot.style.background = color;
  dot.style.boxShadow = `0 0 9px ${color}99`;
  if (text) $("status-text").textContent = text;
  $("data-gate-banner").hidden = s !== "data-missing";
  updateButtons();
}

function defaultStatusText(s) {
  if (s === "data-missing") return "Data missing — check KRAKEN_DATA_PATH";
  if (s === "running") return `Running — fold ${state.curFold} / ${state.totalFolds}, epoch ${state.curEpoch} / ${state.maxEpochs}`;
  if (s === "saving") return "Saving checkpoint…";
  if (s === "error") return "Error — see alert below";
  if (s === "stopped") return "Stopped — checkpoint saved";
  if (s === "completed") return "Completed — models promoted to benchmark";
  return "Idle — ready to start";
}

function updateButtons() {
  const active = state.scenario === "running" || state.scenario === "saving";
  const dataMissing = state.scenario === "data-missing";
  const btnStart = $("btn-start");
  const btnStop = $("btn-stop");
  const btnSave = $("btn-save");
  btnStart.hidden = active;
  btnStop.hidden = !active;
  btnSave.hidden = !active;
  btnStart.disabled = dataMissing || state.pendingBtn === "start";
  btnStop.disabled = state.scenario === "saving" || state.pendingBtn === "stop";
  btnSave.disabled = state.scenario === "saving" || state.pendingBtn === "save";
  setSpinner(btnStart, state.pendingBtn === "start", "▶ Start");
  setSpinner(btnStop, state.pendingBtn === "stop", "■ Stop");
  setSpinner(btnSave, state.pendingBtn === "save", "Save checkpoint");
}

function setSpinner(btn, pending, label) {
  btn.innerHTML = pending
    ? '<span class="btn-spinner"></span>'
    : `<span class="btn-label">${label}</span>`;
}

async function postControl(path, btnKey) {
  state.pendingBtn = btnKey;
  updateButtons();
  try {
    const res = await fetch(path, { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      pushAlert({ level: "error", message: `Request failed — ${body.detail || res.status}` });
    }
  } catch (err) {
    pushAlert({ level: "error", message: `Request failed — ${err.message}` });
  } finally {
    state.pendingBtn = null;
    updateButtons();
  }
}

$("btn-start").addEventListener("click", () => postControl("/api/training/start", "start"));
$("btn-stop").addEventListener("click", () => postControl("/api/training/stop", "stop"));
$("btn-save").addEventListener("click", () => postControl("/api/training/save", "save"));
$("btn-log-toggle").addEventListener("click", () => {
  const isLog = chart.toggleLogScale();
  $("btn-log-toggle").textContent = isLog ? "LOG" : "LINEAR";
});

// ---- progress gauges ----
function renderFoldGauge() {
  $("fold-value").textContent = `${state.curFold} / ${state.totalFolds}`;
  const segs = $("fold-segs");
  segs.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (let i = 0; i < state.totalFolds; i++) {
    const s = document.createElement("span");
    if (i < state.curFold) s.style.background = "#4a9eff";
    else if (i === state.curFold && state.scenario === "running") s.style.background = "#7cbaff";
    frag.appendChild(s);
  }
  segs.appendChild(frag);
}

function renderEpochGauge() {
  $("epoch-value").textContent = `${state.curEpoch} / ${state.maxEpochs}`;
  const segs = $("epoch-segs");
  segs.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (let i = 0; i < state.maxEpochs; i++) {
    const s = document.createElement("span");
    if (i === state.bestEpoch && state.bestVal < Infinity) s.style.background = "#2ecc71";
    else if (i > state.bestEpoch && i <= state.curEpoch) s.style.background = "#a06a1a";
    else if (i <= state.curEpoch) s.style.background = "#4a9eff";
    frag.appendChild(s);
  }
  segs.appendChild(frag);
  $("best-caption").textContent =
    state.bestVal < Infinity ? `best · ep ${state.bestEpoch} · ${fmt4(state.bestVal)}` : "best epoch";
}

// ---- patience ----
function renderPatience() {
  const pat = state.patience;
  const max = state.maxPatience;
  $("patience-value").textContent = `${pat} / ${max}`;
  let color = "#4a9eff";
  if (pat >= max) color = "#e74c3c";
  else if (pat >= cfg.patience_orange_floor) color = "#e67e22";
  else if (pat >= cfg.patience_amber_floor) color = "#f39c12";
  $("patience-value").style.color = color;
  $("patience-fill").style.width = `${Math.min(100, (pat / max) * 100)}%`;
  $("patience-fill").style.background = color;
  const track = $("patience-track");
  track.classList.remove("glow-amber", "glow-red");
  const nearStop = pat >= cfg.patience_glow_floor;
  if (pat >= max) track.classList.add("glow-red");
  else if (nearStop) track.classList.add("glow-amber");
  $("patience-hint").textContent =
    pat >= max
      ? "exhausted — early stop firing"
      : nearStop
        ? "close to early stop"
        : `${max - pat} epochs of headroom`;
}

// ---- ETA strip ----
function fmtHMS(s) {
  if (s == null) return "—";
  s = Math.round(s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`
    : `${m}:${String(ss).padStart(2, "0")}`;
}
function fmtDHM(s) {
  if (s == null) return "—";
  s = Math.round(s);
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  return d > 0 ? `${d}d ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}` : `${h}:${String(m).padStart(2, "0")}:00`;
}
function fmtDur(s) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

function renderEta(epochEtaS, foldEtaS, totalEtaS) {
  const epochRow = $("eta-epoch").closest(".eta-row");
  const foldRow = $("eta-fold").closest(".eta-row");
  const totalRow = $("eta-total").closest(".eta-row");
  const live = epochEtaS != null;
  epochRow.classList.toggle("live", live);
  foldRow.classList.toggle("live", live);
  totalRow.classList.toggle("live", live);
  $("eta-epoch").textContent = fmtHMS(epochEtaS);
  $("eta-fold").textContent = fmtHMS(foldEtaS);
  $("eta-total").textContent = fmtDHM(totalEtaS);
}

// ---- loss component strip ----
function delta(cur, prev) {
  const d = cur - prev;
  const sign = d < 0 ? "−" : "+";
  return { text: `(${sign}${Math.abs(d).toFixed(4)})`, color: d < 0 ? "#2ecc71" : "#e74c3c" };
}

function renderComponentStrip(payload) {
  $("m-pinball").textContent = fmt4(payload.train_pinball);
  $("m-direction").textContent = fmt4(payload.train_direction);
  $("m-total").textContent = fmt4(payload.val_total);
  const dp = delta(payload.val_pinball, state.prevPinball);
  const dd = delta(payload.val_direction, state.prevDirection);
  const dt = delta(payload.val_total, state.prevTotal);
  $("m-pinball-delta").textContent = dp.text;
  $("m-pinball-delta").style.color = dp.color;
  $("m-direction-delta").textContent = dd.text;
  $("m-direction-delta").style.color = dd.color;
  $("m-total-delta").textContent = dt.text;
  $("m-total-delta").style.color = dt.color;
  state.prevPinball = payload.val_pinball;
  state.prevDirection = payload.val_direction;
  state.prevTotal = payload.val_total;
}

function renderBatchMetrics(payload) {
  $("m-lr").textContent = typeof payload.lr === "number" ? payload.lr.toExponential(2) : "—";
  const gn = payload.grad_norm;
  const clip = cfg.grad_clip_norm;
  let color = "#e8e8e8";
  let hint = "stable";
  if (gn > clip * cfg.grad_norm_instability_multiplier) {
    color = "#e74c3c";
    hint = "unstable — high";
  } else if (gn > clip) {
    color = "#f39c12";
    hint = "clipping";
  }
  $("m-grad").textContent = typeof gn === "number" ? gn.toFixed(4) : "—";
  $("m-grad").style.color = color;
  $("m-grad-hint").textContent = hint;
  $("m-grad-hint").style.color = color;
}

// ---- fold history table ----
function renderFoldTable() {
  const body = $("fold-table-body");
  body.innerHTML = "";
  state.minValLoss = state.foldRows.reduce((m, r) => Math.min(m, r.val_loss), Infinity);
  const frag = document.createDocumentFragment();
  state.foldRows.forEach((r) => {
    const tr = document.createElement("tr");
    if (r._flash) tr.classList.add("flash");
    const isBest = Math.abs(r.val_loss - state.minValLoss) < 1e-9;
    const valColor = isBest ? "#2ecc71" : "#cfcfcf";
    let daColor = "#cfcfcf";
    if (r.da > cfg.deploy_gate_da_threshold) daColor = "#2ecc71";
    else if (r.da >= cfg.fold_table_da_amber_floor) daColor = "#f39c12";
    else daColor = "#e74c3c";
    let qColor = "#cfcfcf";
    if (r.q_coverage < cfg.fold_table_qcov_healthy_lower || r.q_coverage > cfg.fold_table_qcov_healthy_upper) qColor = "#e74c3c";
    else qColor = "#2ecc71";

    tr.innerHTML = `
      <td>${r.fold}</td>
      <td class="num">${fmt4(r.train_loss)}</td>
      <td class="num" style="color:${valColor};font-weight:600">${fmt4(r.val_loss)}</td>
      <td class="num" style="color:${daColor}">${Number.isFinite(r.da) ? (r.da * 100).toFixed(1) + "%" : "—"}</td>
      <td class="num" style="color:${qColor}">${fmt4(r.q_coverage)}</td>
      <td class="num">${fmtDur(r.duration_s)}</td>
    `;
    frag.appendChild(tr);
  });
  body.appendChild(frag);
}

function addFoldRow(rec) {
  state.foldRows = [{ ...rec, _flash: true }, ...state.foldRows];
  renderFoldTable();
  setTimeout(() => {
    const row = state.foldRows.find((r) => r.fold === rec.fold);
    if (row) row._flash = false;
  }, 600);
}

// ---- W&B panel ----
function renderWandb(mode, runId) {
  const online = mode === "online";
  const offline = mode === "offline";
  $("wandb-panel").hidden = !online;
  $("wandb-offline").hidden = !offline;
  if (online && runId) {
    $("wandb-run-id").textContent = runId;
    $("wandb-run-id").title = runId;
    $("wandb-link").href = `https://wandb.ai/user/${encodeURIComponent("btc-bot-v3-predictor")}/runs/${encodeURIComponent(runId)}`;
  }
}

// ---- alerts ----
const ALERT_META = {
  good: { icon: "✓", cls: "alert-good" },
  warn: { icon: "⚠", cls: "alert-warn" },
  error: { icon: "✕", cls: "alert-error" },
  save: { icon: "↓", cls: "alert-save" },
};

function pushAlert({ level, message }) {
  const meta = ALERT_META[level] || ALERT_META.good;
  const id = state.alertSeq++;
  const el = document.createElement("div");
  el.className = `alert ${meta.cls}`;
  el.dataset.id = String(id);
  el.innerHTML = `
    <span class="alert-icon">${meta.icon}</span>
    <span class="alert-msg"></span>
    <button class="alert-close" type="button">×</button>
  `;
  el.querySelector(".alert-msg").textContent = message;
  el.querySelector(".alert-close").addEventListener("click", () => el.remove());

  const container = $("alerts");
  container.prepend(el);
  while (container.children.length > cfg.alert_max_stack) {
    container.removeChild(container.lastElementChild);
  }
  setTimeout(() => el.remove(), cfg.alert_auto_dismiss_seconds * 1000);
}

// ---- SSE message routing ----
function handleMessage(payload) {
  switch (payload.type) {
    case "status": {
      if (typeof payload.wandb_mode === "string") renderWandb(payload.wandb_mode, payload.wandb_run_id);
      const text = payload.state === "completed" && typeof payload.promoted === "number"
        ? `Completed — ${payload.promoted} model${payload.promoted === 1 ? "" : "s"} promoted to benchmark`
        : defaultStatusText(payload.state);
      setStatus(payload.state, text);
      break;
    }
    case "batch":
      if (typeof payload.total_folds === "number") state.totalFolds = payload.total_folds;
      state.curFold = payload.fold;
      state.curEpoch = payload.epoch;
      chart.addBatch(payload.total);
      renderBatchMetrics(payload);
      renderFoldGauge();
      renderEpochGauge();
      if (state.scenario === "running") setStatus("running", defaultStatusText("running"));
      break;
    case "epoch":
      if (typeof payload.total_folds === "number") state.totalFolds = payload.total_folds;
      if (typeof payload.max_epochs === "number") state.maxEpochs = payload.max_epochs;
      if (typeof payload.max_patience === "number") state.maxPatience = payload.max_patience;
      state.curFold = payload.fold;
      state.curEpoch = payload.epoch;
      state.patience = payload.patience;
      if (payload.val_total < state.bestVal) {
        state.bestVal = payload.val_total;
        state.bestEpoch = payload.epoch;
      }
      chart.addEpoch({
        fold: payload.fold, epoch: payload.epoch,
        trainTotal: payload.train_total, valTotal: payload.val_total,
      });
      renderComponentStrip(payload);
      renderFoldGauge();
      renderEpochGauge();
      renderPatience();
      renderEta(payload.epoch_eta_s, payload.fold_eta_s, payload.total_eta_s);
      if (state.scenario === "running") setStatus("running", defaultStatusText("running"));
      break;
    case "fold_complete":
      addFoldRow(payload);
      state.bestVal = Infinity;
      state.bestEpoch = 0;
      break;
    case "alert":
      pushAlert(payload);
      if (payload.level === "warn" && /early stopping/i.test(payload.message || "")) {
        chart.markEarlyStop();
      }
      break;
    default:
      break;
  }
}

// ---- bootstrap ----
async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return;
    Object.assign(cfg, await res.json());
    state.maxEpochs = cfg.max_epochs;
    state.maxPatience = cfg.early_stopping_patience;
  } catch {
    // fall back to the defaults baked into cfg above
  }
}

async function loadHistory() {
  try {
    const res = await fetch("/api/training/history");
    if (!res.ok) return;
    const rows = await res.json();
    state.foldRows = rows.slice().reverse();
    renderFoldTable();
  } catch {
    // history is best-effort seed data; the live stream is authoritative
  }
}

function connect() {
  const es = new EventSource("/api/events");
  es.onmessage = (e) => {
    try {
      handleMessage(JSON.parse(e.data));
    } catch {
      // malformed frame — drop it, the stream self-heals on the next message
    }
  };
  es.onerror = () => {
    setStatus("error", "Connection lost — reconnecting…");
  };
}

async function main() {
  await loadConfig();
  chart = new LossChart($("loss-chart"), $("hover-tip"), {
    emaWindow: cfg.loss_chart_ema_window,
    visibleEpochs: cfg.loss_chart_visible_epochs,
    batchTraceMax: cfg.batch_trace_max_points,
  });
  renderFoldGauge();
  renderEpochGauge();
  renderPatience();
  renderEta(null, null, null);
  await loadHistory();
  connect();
}

main();
