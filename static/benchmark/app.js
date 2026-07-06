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

function fmtHitPct(v) {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(0)}%`;
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
  setTimeout(() => box.remove(), CONFIG.alert_auto_dismiss_seconds * 1000);
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
  // CONFIG.null_significance_level mirrors the pre-live permutation gate's significance
  // level; it is display emphasis only — the number itself is always shown.
  const cls = p < CONFIG.null_significance_level ? "sig" : "insig";
  return td(fmtNum(p, 3), `num ${cls}`);
}

function gradeTooltip(grade) {
  if (!CONFIG) return "";
  const p = CONFIG.profitable_p_value_max;
  const n = CONFIG.profitable_min_trades;
  if (grade === "profitable")
    return `Profitable: positive net-of-fee expectancy per trade, beats the ` +
      `random-entry null (p < ${p}), and >= ${n} trades.`;
  if (grade === "insufficient")
    return `Insufficient: fewer than ${n} trades (or none) — expectancy too ` +
      `noisy to grade. Neither green nor red.`;
  return `Not profitable: fails positive per-trade expectancy and/or does not ` +
    `beat the random-entry null (p < ${p}).`;
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

// Which run drill-downs are open, by "git_sha:constants_sha8" — survives the re-render
// on every SSE refresh so a benchmark completing does not collapse the user's open run.
const EXPANDED = new Set();

function runKey(run) {
  return `${run.git_sha}:${run.constants_sha8}`;
}

// The drill-down sub-table header: the fold-checkpoint columns, with Exp. leading the
// metrics since per-trade expectancy is the within-run rank basis.
const MEMBER_COLS = [
  "#", "Model", "Fold", "Exp.", "DA", "Cal.", "Pinball", "Net", "vs B&H", "p(null)",
  "Sharpe", "Max DD", "Trades", "Hit", "Net win", "Capt.", "Adv.",
];

function renderMemberRow(r) {
  const t = r.trading || {};
  const b = r.baselines || {};
  const s = r.statistical || {};
  const e = r.economic || {};
  const row = document.createElement("tr");
  // Green-grade tint: profitable = positive per-trade expectancy AND beats the
  // random-entry null (p < profitable_p_value_max) AND >= profitable_min_trades trades;
  // insufficient = below the trade floor.
  if (r.profitability === "profitable") row.className = "row-profitable";
  else if (r.profitability === "insufficient") row.className = "row-insufficient";
  row.title = gradeTooltip(r.profitability);
  row.appendChild(td(String(r.rank), "num rank"));
  row.appendChild(td(
    `${esc(r.display_name)}<span class="stem-sub">${esc(r.stem)}</span>`, "name-cell"
  ));
  row.appendChild(td(r.fold_index === null ? "—" : String(r.fold_index), "num"));
  // Expectancy leads — the within-run rank basis (expectancy desc, DA desc tie-break).
  row.appendChild(td(fmtPct(r.expectancy, 3), `num ${signClass(r.expectancy)}`));
  row.appendChild(daCell(s.directional_accuracy));
  row.appendChild(calCell(s.calibration_rate));
  row.appendChild(td(fmtNum(s.pinball, 4), "num"));
  row.appendChild(td(fmtPct(t.net_return), `num ${signClass(t.net_return)}`));
  const vsBh = t.net_return !== null && t.net_return !== undefined &&
    b.buy_and_hold_net !== null && b.buy_and_hold_net !== undefined
    ? t.net_return - b.buy_and_hold_net : null;
  row.appendChild(td(fmtPct(vsBh), `num ${signClass(vsBh)}`));
  row.appendChild(pCell(b.p_value));
  row.appendChild(td(fmtNum(t.sharpe), `num ${signClass(t.sharpe)}`));
  row.appendChild(td(fmtPct(t.max_drawdown !== undefined ? -t.max_drawdown : null), "num neg"));
  row.appendChild(td(t.trade_count === undefined ? "—" : String(t.trade_count), "num"));
  // Hit = directional (right side, ~50% expected); Net win = profitable after fee.
  row.appendChild(td(fmtHitPct(t.directional_hit_rate), "num"));
  row.appendChild(td(fmtHitPct(t.hit_rate), "num"));
  row.appendChild(td(fmtRatio(e.mean_captured_fraction), "num"));
  row.appendChild(td(fmtRatio(e.mean_adverse_ratio), "num"));
  return row;
}

function renderRunMembers(run) {
  const table = document.createElement("table");
  table.className = "member-table";
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  for (const label of MEMBER_COLS) {
    const th = document.createElement("th");
    if (label !== "Model") th.className = "num";
    th.textContent = label;
    htr.appendChild(th);
  }
  thead.appendChild(htr);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const m of run.models) tbody.appendChild(renderMemberRow(m));
  table.appendChild(tbody);
  return table;
}

function renderLeaderboard(payload) {
  const body = $("leaderboard-body");
  body.replaceChildren();
  const runs = payload.runs || [];
  $("leaderboard-empty").classList.toggle("hidden", runs.length > 0);
  for (const run of runs) {
    const key = runKey(run);
    const open = EXPANDED.has(key);

    const runRow = document.createElement("tr");
    runRow.className = "run-row";
    const caret = td(open ? "▾" : "▸", "caret");
    runRow.appendChild(caret);
    runRow.appendChild(td(
      `${esc(run.git_sha)}<span class="stem-sub">constants ${esc(run.constants_sha8)}</span>`,
      "name-cell"
    ));
    const foldsExtra = run.n_checkpoints > run.n_benchmarked
      ? ` <span class="dim">of ${run.n_checkpoints}</span>` : "";
    runRow.appendChild(td(`${run.n_benchmarked}${foldsExtra}`, "num"));
    // Denominator = benchmarked folds; insufficient counts here but never as profitable.
    const frac = run.profitable_fraction;
    const profClass = frac === null || frac === undefined ? "dim" : frac > 0 ? "pos" : "neg";
    const insuff = run.n_insufficient > 0
      ? ` <span class="dim">+${run.n_insufficient} insuff.</span>` : "";
    runRow.appendChild(td(
      `<b class="${profClass}">${run.n_profitable}/${run.n_benchmarked}</b>${insuff}`, "num"
    ));
    runRow.appendChild(td(fmtPct(run.mean_expectancy, 3), `num ${signClass(run.mean_expectancy)}`));
    runRow.appendChild(td(fmtNum(run.mean_da, 3), "num"));
    runRow.appendChild(td(fmtPct(run.mean_net_return), `num ${signClass(run.mean_net_return)}`));

    const detailRow = document.createElement("tr");
    detailRow.className = "run-detail";
    if (!open) detailRow.classList.add("hidden");
    const detailCell = document.createElement("td");
    detailCell.colSpan = 7;
    detailCell.appendChild(renderRunMembers(run));
    detailRow.appendChild(detailCell);

    runRow.addEventListener("click", () => {
      // classList.toggle returns true when 'hidden' is now present (i.e. collapsed).
      const collapsed = detailRow.classList.toggle("hidden");
      caret.textContent = collapsed ? "▸" : "▾";
      if (collapsed) EXPANDED.delete(key);
      else EXPANDED.add(key);
    });

    body.appendChild(runRow);
    body.appendChild(detailRow);
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
