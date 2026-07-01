// chart.js — canvas loss-chart renderer (training-dashboard.md §"B — Loss chart").
// Plain <canvas>, hand-rolled — no chart library, no D3, no build step.

// Defaults mirror TrainingUIConfig; overridden by the real values from GET /api/config
// (app.js passes them in) so this file never silently drifts from constants.py.
const DEFAULT_EMA_WINDOW = 5;
const DEFAULT_VISIBLE_EPOCHS = 200;
const DEFAULT_BATCH_TRACE_MAX = 40;

export class LossChart {
  constructor(canvas, tooltipEl, config = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.tooltipEl = tooltipEl;
    this.emaWindow = config.emaWindow ?? DEFAULT_EMA_WINDOW;
    this.visibleEpochs = config.visibleEpochs ?? DEFAULT_VISIBLE_EPOCHS;
    this.batchTraceMax = config.batchTraceMax ?? DEFAULT_BATCH_TRACE_MAX;
    this.series = []; // {g, fold, epoch, trainTotal, valTotal}
    this.foldBoundaries = []; // {g, fold}
    this.esMarkers = []; // g values where early stopping fired
    this.batchTrace = []; // recent raw train batch losses (current epoch only)
    this.logScale = false;
    this.hoverIdx = null;
    this._map = null;
    this._g = 0;

    canvas.addEventListener("mousemove", (e) => this._onMove(e));
    canvas.addEventListener("mouseleave", () => this._onLeave());
    window.addEventListener("resize", () => this.draw());
  }

  addEpoch({ fold, epoch, trainTotal, valTotal }) {
    const g = this._g++;
    const prev = this.series[this.series.length - 1];
    if (!prev || prev.fold !== fold) this.foldBoundaries.push({ g, fold });
    this.series.push({ g, fold, epoch, trainTotal, valTotal });
    this.batchTrace = [];
    this.draw();
  }

  addBatch(total) {
    this.batchTrace.push(total);
    if (this.batchTrace.length > this.batchTraceMax) this.batchTrace.shift();
    this.draw();
  }

  markEarlyStop() {
    const last = this.series[this.series.length - 1];
    if (last) this.esMarkers.push(last.g);
  }

  toggleLogScale() {
    this.logScale = !this.logScale;
    this.draw();
    return this.logScale;
  }

  static ema(points, key, window) {
    const alpha = 2 / (window + 1);
    let prev = null;
    return points.map((p) => {
      const v = p[key];
      prev = prev == null ? v : v * alpha + prev * (1 - alpha);
      return { g: p.g, v: prev };
    });
  }

  draw() {
    const cv = this.canvas;
    const wrap = cv.parentElement;
    const cssW = wrap.clientWidth;
    const cssH = wrap.clientHeight;
    if (!cssW || !cssH) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = cssW * dpr;
    cv.height = cssH * dpr;
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const S = this.series;
    if (!S.length) return;

    const padL = 42;
    const padR = 12;
    const padT = 14;
    const padB = 22;
    const plotW = cssW - padL - padR;
    const plotH = cssH - padT - padB;
    const wStart = Math.max(0, S.length - this.visibleEpochs);
    const win = S.slice(wStart);
    const gmin = win[0].g;
    const gmax = win[win.length - 1].g;
    const log = this.logScale;

    let lo = Infinity;
    let hi = -Infinity;
    win.forEach((p) => {
      lo = Math.min(lo, p.trainTotal, p.valTotal);
      hi = Math.max(hi, p.trainTotal, p.valTotal);
    });
    const padY = (hi - lo) * 0.12 || 0.1;
    lo -= padY;
    hi += padY;
    const yv = (v) => (log ? Math.log(Math.max(v, 1e-3)) : v);
    const ylo = yv(Math.max(lo, 1e-3));
    const yhi = yv(hi);
    const xOf = (g) => padL + (gmax === gmin ? 0 : (g - gmin) / (gmax - gmin)) * plotW;
    const yOf = (v) => padT + (1 - (yv(v) - ylo) / ((yhi - ylo) || 1)) * plotH;

    ctx.font = "10px ui-monospace, monospace";
    ctx.strokeStyle = "#191919";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const v = lo + (hi - lo) * (i / 4);
      const y = yOf(v);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(cssW - padR, y);
      ctx.stroke();
      ctx.fillStyle = "#555";
      ctx.fillText(v.toFixed(2), 5, y + 3);
    }

    this.foldBoundaries.forEach((b) => {
      if (b.g < gmin || b.g > gmax) return;
      const x = xOf(b.g);
      ctx.strokeStyle = "#2a2a2a";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, cssH - padB);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#5a5a5a";
      ctx.fillText(`F${b.fold}`, x + 3, padT + 9);
    });

    this.esMarkers.forEach((g) => {
      if (g < gmin || g > gmax) return;
      const x = xOf(g);
      ctx.strokeStyle = "rgba(243,156,18,.65)";
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, cssH - padB);
      ctx.stroke();
      ctx.fillStyle = "#f39c12";
      ctx.fillText("ES", x + 3, padT + 20);
    });

    const drawRaw = (key, color) => {
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.16;
      win.forEach((p) => {
        ctx.beginPath();
        ctx.arc(xOf(p.g), yOf(p[key]), 1.3, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    };
    const drawLine = (pts, color, w) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = w;
      ctx.lineJoin = "round";
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = xOf(p.g);
        const y = yOf(p.v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawRaw("trainTotal", "#4a9eff");
    drawRaw("valTotal", "#2ecc71");
    drawLine(LossChart.ema(win, "trainTotal", this.emaWindow), "#4a9eff", 2);
    drawLine(LossChart.ema(win, "valTotal", this.emaWindow), "#2ecc71", 2);

    // Current-epoch raw batch-level train trace: thin, semi-transparent, anchored to
    // the right edge (training-dashboard.md §B "real-time within-epoch visibility").
    if (this.batchTrace.length > 1) {
      const lastG = S[S.length - 1].g;
      const stepPx = 3;
      ctx.strokeStyle = "rgba(74,158,255,.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      this.batchTrace.forEach((v, i) => {
        const x = xOf(lastG) + (i - this.batchTrace.length + 1) * stepPx;
        const y = yOf(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    if (this.hoverIdx != null) {
      const p = S[this.hoverIdx];
      if (p && p.g >= gmin && p.g <= gmax) {
        const x = xOf(p.g);
        ctx.strokeStyle = "rgba(255,255,255,.14)";
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, cssH - padB);
        ctx.stroke();
        ctx.fillStyle = "#4a9eff";
        ctx.beginPath();
        ctx.arc(x, yOf(p.trainTotal), 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#2ecc71";
        ctx.beginPath();
        ctx.arc(x, yOf(p.valTotal), 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    this._map = { padL, plotW, gmin, gmax };
  }

  _onMove(e) {
    const m = this._map;
    if (!m) return;
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const g = m.gmin + ((x - m.padL) / m.plotW) * (m.gmax - m.gmin);
    let best = null;
    let bestDist = Infinity;
    this.series.forEach((p, i) => {
      if (p.g < m.gmin || p.g > m.gmax) return;
      const d = Math.abs(p.g - g);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    });
    if (best == null) return;
    this.hoverIdx = best;
    this.draw();
    const p = this.series[best];
    if (!this.tooltipEl) return;
    this.tooltipEl.hidden = false;
    this.tooltipEl.style.left = `${x + 14}px`;
    this.tooltipEl.style.top = `${Math.max(6, e.clientY - r.top - 10)}px`;
    this.tooltipEl.querySelector("#hover-epoch").textContent = `Fold ${p.fold} · epoch ${p.epoch}`;
    this.tooltipEl.querySelector("#hover-train-val").textContent = p.trainTotal.toFixed(4);
    this.tooltipEl.querySelector("#hover-val-val").textContent = p.valTotal.toFixed(4);
  }

  _onLeave() {
    this.hoverIdx = null;
    if (this.tooltipEl) this.tooltipEl.hidden = true;
    this.draw();
  }
}
