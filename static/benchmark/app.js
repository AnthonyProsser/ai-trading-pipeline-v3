/* Model Benchmark frontend — vanilla JS, no build step.
   All thresholds come from /api/config (never hardcode a Python constant here). */
"use strict";

const $ = (id) => document.getElementById(id);

let CONFIG = null;
let RUNNING_STEM = null;

/* ---------------------------------------------------------------- formatting */

function fmtPct(logret, digits = 2) {
  if (logret === null || logret === undefined) return "—";
  const pct = (Math.exp(logret) - 1) * 100;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(digits)}%`;
}

function fmtNum(v, digits = 2) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(digits);
}

function fmtRatio(v) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(2);
}

function signClass(v) {
  if (v === null || v === undefined) return "dim";
  return v > 0 ? "pos" : v < 0 ? "neg" : "dim";
}

function td(html, cls) {
  const cell = document.createElement("td");
  if (cls) cell.className = cls;
  cell.innerHTML = html;
  return cell;
}

function esc(text) {
  const div = document.createElement("div");
  div.textContent = String(text);
  return div.innerHTML;
}

/* ---------------------------------------------------------------- alerts */

function alertBox(message, level) {
  const box = document.createElement("div");
  box.className = `alert ${level || ""}`;
  box.textContent = message;
  $("alerts").appendChild(box);
  setTimeout(() => box.remove(), 8000);
}

/* ---------------------------------------------------------------- config + rule */

async function loadConfig() {
  CONFIG = await (await fetch("/api/config")).json();
  const feePct = (CONFIG.fee_threshold * 100).toFixed(2);
  $("rule-text").innerHTML =
    `Fixed measuring instrument — identical for every model, never tuned per model: ` +
    `enter only when the predicted <b>${CONFIG.horizon}-step</b> cumulative close move ` +
    `clears the round-trip fee (<b>|q50| &gt; ${feePct}%</b>) AND the ` +
    `<b>[q10, q90]</b> interval does not straddle zero; hold to horizon, pay the fee, ` +
    `never overlap trades. Nulls: buy-and-hold, and <b>${CONFIG.null_draws}</b> ` +
    `random-entry draws at matched trade frequency.`;
}

/* ---------------------------------------------------------------- models table */

function nameCell(model) {
  const cell = document.createElement("td");
  cell.className = "name-cell";
  const btn = document.createElement("button");
  btn.className = "name-btn";
  btn.title = "Click to rename (registry display name only — files are never renamed)";
  btn.textContent = model.display_name;
  btn.addEventListener("click", () => {
    const input = document.createElement("input");
    input.className = "name-input";
    input.value = model.display_name === model.stem ? "" : model.display_name;
    input.placeholder = model.stem;
    cell.replaceChildren(input);
    input.focus();
    const commit = async () => {
      const r = await fetch(`/api/models/${encodeURIComponent(model.stem)}/display-name`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: input.value }),
      });
      if (!r.ok) alertBox(`Rename failed (${r.status})`, "error");
      await refreshAll();
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") commit();
      if (e.key === "Escape") refreshAll();
    });
    input.addEventListener("blur", commit);
  });
  cell.appendChild(btn);
  if (model.display_name !== model.stem) {
    const sub = document.createElement("span");
    sub.className = "stem-sub";
    sub.textContent = model.stem;
    cell.appendChild(sub);
  }
  return cell;
}

function benchmarkCell(model) {
  const cell = document.createElement("td");
  if (model.has_benchmark && model.result && model.result.trading) {
    const net = model.result.trading.net_return;
    const chip = document.createElement("span");
    chip.className = `score-chip ${signClass(net)}`;
    chip.textContent = `net ${fmtPct(net)}`;
    chip.title = "Benchmarked — see leaderboard. Delete its .benchmark.json to re-run.";
    cell.appendChild(chip);
  } else if (!model.compatible) {
    const span = document.createElement("span");
    span.className = "incompat";
    span.textContent = model.incompatible_reason || "incompatible";
    cell.appendChild(span);
  } else {
    const btn = document.createElement("button");
    btn.className = "btn btn-bench";
    btn.textContent = "Start benchmark";
    btn.disabled = RUNNING_STEM !== null;
    if (RUNNING_STEM !== null) btn.title = `Busy: benchmarking ${RUNNING_STEM}`;
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      const r = await fetch(`/api/benchmark/${encodeURIComponent(model.stem)}/start`, {
        method: "POST",
      });
      if (!r.ok) {
        const detail = (await r.json()).detail || r.status;
        alertBox(`Could not start: ${detail}`, "error");
        btn.disabled = false;
      }
    });
    cell.appendChild(btn);
  }
  return cell;
}

function renderModels(models) {
  const body = $("models-body");
  body.replaceChildren();
  for (const m of models) {
    const row = document.createElement("tr");
    row.appendChild(nameCell(m));
    row.appendChild(td(m.fold_index === null ? "—" : String(m.fold_index), "num"));
    row.appendChild(td(esc(m.git_sha || "—"), "mono"));
    row.appendChild(td(m.meta && m.meta.lookback ? String(m.meta.lookback) : "—", "num"));
    row.appendChild(td(esc((m.meta && m.meta.trained_through_ts_utc) || "—"), "mono"));
    const cov = m.meta ? m.meta.train_q90_coverage : null;
    row.appendChild(td(cov === null || cov === undefined ? "—" : fmtNum(cov, 4), "num"));
    row.appendChild(benchmarkCell(m));
    body.appendChild(row);
  }
  $("model-count").textContent = `${models.length} models`;
}

/* ---------------------------------------------------------------- leaderboard */

function pCell(p) {
  if (p === null || p === undefined) return td("—", "num dim");
  // 0.05 mirrors the pre-live permutation gate's significance level; it is display
  // emphasis only — the number itself is always shown.
  const cls = p < 0.05 ? "sig" : "insig";
  return td(fmtNum(p, 3), `num ${cls}`);
}

function calCell(cal) {
  if (cal === null || cal === undefined) return td("—", "num dim");
  const inBand = cal >= CONFIG.deploy_gate_cal_lower && cal <= CONFIG.deploy_gate_cal_upper;
  return td(fmtNum(cal, 3), `num ${inBand ? "pos" : "warn"}`);
}

function daCell(da) {
  if (da === null || da === undefined) return td("—", "num dim");
  return td(fmtNum(da, 3), `num ${da > CONFIG.deploy_gate_da_threshold ? "pos" : "dim"}`);
}

function renderLeaderboard(payload) {
  const body = $("leaderboard-body");
  body.replaceChildren();
  const rows = payload.models;
  $("leaderboard-empty").classList.toggle("hidden", rows.length > 0);
  for (const r of rows) {
    const t = r.trading || {};
    const b = r.baselines || {};
    const s = r.statistical || {};
    const e = r.economic || {};
    const row = document.createElement("tr");
    if (r.rank === 1) row.className = "top-rank";
    row.appendChild(td(String(r.rank), "num rank"));
    const name = td(
      `${esc(r.display_name)}<span class="stem-sub">${esc(r.stem)}</span>`, "name-cell"
    );
    row.appendChild(name);
    row.appendChild(td(r.fold_index === null ? "—" : String(r.fold_index), "num"));
    row.appendChild(td(fmtPct(t.net_return), `num ${signClass(t.net_return)}`));
    const vsBh = t.net_return !== null && t.net_return !== undefined &&
      b.buy_and_hold_net !== null && b.buy_and_hold_net !== undefined
      ? t.net_return - b.buy_and_hold_net : null;
    row.appendChild(td(fmtPct(vsBh), `num ${signClass(vsBh)}`));
    row.appendChild(pCell(b.p_value));
    row.appendChild(td(fmtNum(t.sharpe), `num ${signClass(t.sharpe)}`));
    row.appendChild(td(fmtPct(t.max_drawdown !== undefined ? -t.max_drawdown : null), "num neg"));
    row.appendChild(td(t.trade_count === undefined ? "—" : String(t.trade_count), "num"));
    row.appendChild(td(t.hit_rate === null || t.hit_rate === undefined
      ? "—" : `${(t.hit_rate * 100).toFixed(0)}%`, "num"));
    row.appendChild(td(fmtRatio(e.mean_captured_fraction), "num"));
    row.appendChild(td(fmtRatio(e.mean_adverse_ratio), "num"));
    row.appendChild(daCell(s.directional_accuracy));
    row.appendChild(calCell(s.calibration_rate));
    body.appendChild(row);
  }

  const runs = $("runs-summary");
  runs.replaceChildren();
  for (const g of payload.runs) {
    const card = document.createElement("div");
    card.className = "run-card";
    card.innerHTML =
      `run <b>${esc(g.git_sha)}</b> · constants <b>${esc(g.constants_sha8)}</b> · ` +
      `${g.n_models} model${g.n_models === 1 ? "" : "s"} · mean net ` +
      `<b class="${signClass(g.mean_net_return)}">${fmtPct(g.mean_net_return)}</b>`;
    runs.appendChild(card);
  }
}

/* ---------------------------------------------------------------- refresh + SSE */

async function refreshAll() {
  const [modelsPayload, leaderboard] = await Promise.all([
    (await fetch("/api/models")).json(),
    (await fetch("/api/leaderboard")).json(),
  ]);
  RUNNING_STEM = modelsPayload.running_stem;
  renderModels(modelsPayload.models);
  renderLeaderboard(leaderboard);
}

function setStatus(state, detail) {
  const dot = $("status-dot");
  dot.className = `status-dot ${state === "running" ? "running" : state === "error" ? "error" : "idle"}`;
  $("status-text").textContent = detail || state;
}

function connectEvents() {
  const source = new EventSource("/api/events");
  source.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "status") {
      RUNNING_STEM = msg.stem || null;
      setStatus(msg.state, msg.state === "running" ? `benchmarking ${msg.stem}` : "idle");
      $("progress-banner").classList.toggle("hidden", msg.state !== "running");
      if (msg.state !== "running") refreshAll();
    } else if (msg.type === "benchmark_progress") {
      $("progress-banner").classList.remove("hidden");
      $("progress-text").textContent =
        `Benchmarking ${msg.stem} — ${msg.stage}` +
        (msg.n_windows ? ` (${msg.n_windows} windows)` : "");
    } else if (msg.type === "benchmark_complete") {
      alertBox(
        `Benchmark complete: ${msg.stem} — net ${fmtPct(msg.net_return)}, ` +
        `${msg.trade_count} trades, p(null) ${fmtNum(msg.p_value, 3)}`,
        "good"
      );
    } else if (msg.type === "alert") {
      alertBox(msg.message, msg.level);
    }
  };
  source.onerror = () => setStatus("error", "stream disconnected — retrying…");
  source.onopen = () => refreshAll();
}

(async function init() {
  await loadConfig();
  await refreshAll();
  setStatus("idle", "idle");
  connectEvents();
})();
