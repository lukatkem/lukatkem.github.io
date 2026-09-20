/* Hydro-1 playground — generation, attention heatmap, next-token brain, loss curve. */
const $ = (s) => document.querySelector(s);
let inspectData = null;
let generating = false;
let genAbort = false;

// ---------- status + training DNA ----------
async function loadStatus() {
  const s = await (await fetch("/api/status")).json();
  if (!s.trained) {
    $("#app").innerHTML = `<div class="notrained"><h1>No checkpoint yet</h1>
      <p>Train the model first: <code>python -m hydro1.train --smoke</code> (1 min proof)<br>
      then the real run: <code>python -m hydro1.train</code></p></div>`;
    return false;
  }
  $("#stats").innerHTML = [
    [fmtNum(s.params), "parameters"],
    [s.layers, "layers × " + (s.history.config?.d_model || "—") + "d"],
    [s.vocab, "char vocab"],
    [s.device.toUpperCase(), "trained on this Mac"],
  ].map(([n, l]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
  drawLoss(s.history);
  return true;
}

function fmtNum(n) {
  return n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "K" : String(n);
}

function drawLoss(h) {
  const c = $("#loss"), ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 34;
  ctx.clearRect(0, 0, W, H);
  const pts = [...h.loss, ...h.val_loss];
  if (!pts.length) return;
  const maxStep = Math.max(...pts.map((p) => p[0]));
  const maxL = Math.max(...pts.map((p) => p[1]));
  const minL = Math.min(...pts.map((p) => p[1])) * 0.95;
  const X = (s) => pad + (s / maxStep) * (W - pad - 10);
  const Y = (l) => pad + (1 - (l - minL) / (maxL - minL)) * (H - pad - 16);
  // grid
  ctx.strokeStyle = "#232838"; ctx.fillStyle = "#8b93a7"; ctx.font = "10px monospace";
  for (let i = 0; i <= 3; i++) {
    const l = minL + ((maxL - minL) * i) / 3;
    ctx.beginPath(); ctx.moveTo(pad, Y(l)); ctx.lineTo(W - 10, Y(l)); ctx.stroke();
    ctx.fillText(l.toFixed(2), 2, Y(l) + 3);
  }
  const line = (arr, color) => {
    if (!arr.length) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    arr.forEach(([s, l], i) => i ? ctx.lineTo(X(s), Y(l)) : ctx.moveTo(X(s), Y(l)));
    ctx.stroke();
  };
  line(h.loss, "#f5a623");
  line(h.val_loss, "#9b6bff");
  $("#steps-label").textContent = `${maxStep} steps`;
}

// ---------- generation ----------
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function generate() {
  if (generating) return;
  generating = true; genAbort = false;
  $("#gen").disabled = true;
  const out = $("#output");
  out.innerHTML = `<span class="muted">thinking with ${$("#tempv").textContent} temperature…</span>`;
  try {
    const r = await fetch("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("#prompt").value, temperature: +$("#temp").value,
        top_k: +$("#topk").value, max_new_tokens: +$("#len").value,
      }),
    });
    const d = await r.json();
    // typewriter reveal of the model's raw output
    out.innerHTML = `<span class="muted">${escapeHtml($("#prompt").value)}</span><span class="cursor"></span>`;
    const text = d.text.slice($("#prompt").value.length);
    let shown = "";
    for (let i = 0; i < text.length; i += 2) {
      if (genAbort) break;
      shown = text.slice(0, i + 2);
      out.innerHTML = `<span class="muted">${escapeHtml($("#prompt").value)}</span>${escapeHtml(shown)}<span class="cursor"></span>`;
      await sleep(12);
    }
    out.innerHTML = `<span class="muted">${escapeHtml($("#prompt").value)}</span>${escapeHtml(shown || text)}`;
  } catch (e) {
    out.textContent = "error: " + e.message;
  } finally { generating = false; $("#gen").disabled = false; }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- inspect ----------
async function analyze() {
  $("#inspect-status").textContent = "running the X-ray…";
  const r = await fetch("/api/inspect", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: $("#inspect-text").value, top_k: 8 }),
  });
  inspectData = await r.json();
  $("#inspect-status").textContent = `${inspectData.layers} layers × ${inspectData.heads} heads analyzed`;
  fillSelectors();
  drawHeatmap();
  drawCharstrip();
}

function fillSelectors() {
  const l = $("#layer"), h = $("#head");
  l.innerHTML = Array.from({ length: inspectData.layers }, (_, i) => `<option value=${i}>${i + 1}</option>`).join("");
  h.innerHTML = Array.from({ length: inspectData.heads }, (_, i) => `<option value=${i}>${i + 1}</option>`).join("");
  l.value = String(inspectData.layers - 1);
}

function drawHeatmap() {
  if (!inspectData) return;
  const li = +$("#layer").value, hi = +$("#head").value;
  const T = inspectData.attention[li][hi].length;
  const cell = Math.max(4, Math.min(18, Math.floor(1150 / T)));
  const c = $("#heat");
  c.width = Math.max(1200, T * cell); c.height = T * cell + 4;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const mat = inspectData.attention[li][hi];
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const a = mat[i][j];
      // black → amber → white ramp
      const r = Math.min(255, a * 480), g = Math.min(255, a * 330), b = a > 0.4 ? (a - 0.4) * 500 : 0;
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.fillRect(j * cell, i * cell, cell - 0.5, cell - 0.5);
    }
  }
  c.onmousemove = (e) => {
    const rect = c.getBoundingClientRect();
    const i = Math.floor((e.clientY - rect.top) / (cell * rect.width / c.width));
    const j = Math.floor((e.clientX - rect.left) / (cell * rect.width / c.width));
    if (i >= 0 && i < T && j >= 0 && j < T) {
      c.title = `"${inspectData.chars[i]}" → "${inspectData.chars[j]}": ${(mat[i][j] * 100).toFixed(1)}%`;
    }
  };
}

function drawCharstrip() {
  const el = $("#charstrip");
  const lossMax = Math.max(...inspectData.per_char_loss, 1);
  el.innerHTML = inspectData.chars
    .map((ch, i) => {
      const surprise = inspectData.per_char_loss[i] / lossMax;
      const bg = `rgba(255,107,107,${(surprise * 0.55).toFixed(3)})`;
      const vis = ch === "\n" ? "⏎" : ch === " " ? "␣" : ch;
      return `<span data-i="${i}" style="background:${bg}">${escapeHtml(vis)}</span>`;
    })
    .join("");
  el.querySelectorAll("span").forEach((s) => {
    s.onclick = () => showPop(s);
  });
}

function showPop(span) {
  const i = +span.dataset.i;
  const pop = $("#pop");
  const preds = inspectData.predictions[i] || [];
  pop.innerHTML = `<div class="muted small">after <b style="color:var(--amber)">${escapeHtml(repr(inspectData.chars[i]))}</b> the model predicts:</div>` +
    preds.map((p) => {
      const w = Math.max(2, p.p * 200);
      return `<div class="probbar"><span class="lbl">${escapeHtml(repr(p.char))}</span>
        <span class="bar" style="width:${w}px"></span><span class="p">${(p.p * 100).toFixed(1)}%</span></div>`;
    }).join("");
  const rect = span.getBoundingClientRect();
  pop.style.display = "block";
  pop.style.left = Math.min(rect.left, window.innerWidth - 250) + "px";
  pop.style.top = rect.bottom + 8 + "px";
}

function repr(ch) {
  return ch === "\n" ? "⏎" : ch === " " ? "␣" : ch;
}

document.addEventListener("click", (e) => {
  if (!e.target.closest("#charstrip span") && !e.target.closest("#pop")) $("#pop").style.display = "none";
});

// ---------- arena ----------
async function runArena() {
  const results = $("#arena-results");
  results.innerHTML = `<div class="muted">running the same prompt through every brain…</div>`;
  try {
    const r = await fetch("/api/arena", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: $("#arena-prompt").value, max_new_tokens: 160 }),
    });
    const d = await r.json();
    results.innerHTML = d.contenders.map((c) => {
      const badge = c.origin === "local-born" ? "amber" : c.origin === "cloud" ? "violet" : "muted";
      const lat = c.latency_ms ? `${c.latency_ms} ms` : "";
      return `<div class="panel" style="margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <b><span class="swatch" style="background:var(--${badge === "muted" ? "border" : badge})"></span>${escapeHtml(c.label)}</b>
          <span class="muted small">${escapeHtml(c.origin)} · ${lat}</span>
        </div>
        <div class="output" style="${c.origin === "local-born" ? "border-color:var(--amber)" : ""}">${escapeHtml(c.text)}</div>
      </div>`;
    }).join("");
  } catch (e) {
    results.innerHTML = `<div class="muted">arena error: ${escapeHtml(e.message)}</div>`;
  }
}

async function loadTeachers() {
  const t = await (await fetch("/api/teachers")).json();
  const el = $("#arena-setup");
  if (t.cloud_configured) {
    el.innerHTML = `☁️ Cloud teachers active: ${t.teachers.filter((x) => x.tier !== "local").map((x) => x.label).join(", ")}`;
  } else {
    el.innerHTML = `ℹ️ No cloud key yet — open the <b>Academy</b> tab to paste one (GLM 5.3 / GLM 5.3 Flash). Until then the arena runs the local fallback teacher. Nothing ever downloads.`;
  }
}
$("#arena-go").onclick = runArena;

// ---------- academy: one-click teachers, distill, retrain ----------
let jobWasActive = false;

async function academyRefresh() {
  try {
    const s = await (await fetch("/api/academy")).json();
    $("#ac-key-status").textContent = s.key_set ? "key saved ✓ — test it or start distilling below" : "no key yet";
    $("#ac-remove").style.display = s.key_set ? "inline-block" : "none";
    if (!$("#ac-base").value || document.activeElement !== $("#ac-base")) $("#ac-base").value = s.base_url || "";
    $("#ac-corpus").textContent = s.distill_ready ? "textbook ready ✓" : "no textbook yet";
    $("#ac-roster").textContent = s.distilled_model
      ? "Active student: DISTILLED (trained on the teachers' textbook) — the Arena already uses it."
      : "Active student: BASE (the original from-scratch run).";
    renderJob(s.job);
  } catch (e) { /* server momentarily away — next poll retries */ }
}

function renderJob(job) {
  const wrap = $("#ac-progresswrap");
  if (!job) { wrap.style.display = "none"; return; }
  if (job.active) {
    wrap.style.display = "block";
    const pct = job.total ? Math.round((job.done / job.total) * 100) : 0;
    $("#ac-bar").style.width = pct + "%";
    $("#ac-stage").textContent =
      `${job.kind} · ${job.stage} · ${job.done}${job.total ? "/" + job.total : ""} (${pct}%) · ${job.elapsed_s}s elapsed`;
    $("#ac-log").textContent = (job.log_tail || []).join("\n");
    $("#ac-log").scrollTop = $("#ac-log").scrollHeight;
  } else if (job.kind && job.exit_code !== null && job.exit_code !== undefined && job.log_tail && job.log_tail.length) {
    // finished — keep the outcome visible until the next run starts
    wrap.style.display = "block";
    $("#ac-bar").style.width = "100%";
    $("#ac-stage").textContent = `${job.kind} finished · exit code ${job.exit_code}`;
    $("#ac-log").textContent = (job.log_tail || []).join("\n");
    $("#ac-log").scrollTop = $("#ac-log").scrollHeight;
  }
}

async function academyTick() {
  try {
    const job = await (await fetch("/api/academy/job")).json();
    renderJob(job);
    if (job.active !== jobWasActive) {
      jobWasActive = job.active;
      if (!job.active) academyRefresh(); // a job just finished — refresh roster/corpus labels
    }
  } catch (e) { /* ignore poll errors */ }
}
setInterval(() => {
  const tabVisible = $("#tab-academy").style.display !== "none";
  if (jobWasActive || tabVisible) academyTick();
}, 2500);

$("#ac-save").onclick = async () => {
  const key = $("#ac-key").value.trim();
  const st = $("#ac-key-status");
  if (!key) { st.textContent = "paste a key first"; return; }
  st.textContent = "testing against the provider…";
  try {
    const r = await (await fetch("/api/academy/key", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, base_url: $("#ac-base").value.trim() || null }),
    })).json();
    st.textContent = r.ok ? "key works ✓ — cloud teachers are live" : "saved, but the test call failed — check the key / provider credits";
    $("#ac-probes").innerHTML = r.probes.map((p) =>
      `<div>${p.ok ? "🟢" : "🔴"} ${escapeHtml(p.label)} ${p.ok ? "· " + p.latency_ms + " ms" : "· " + escapeHtml(p.error || "unreachable")}</div>`).join("");
    $("#ac-key").value = "";
    loadTeachers();
  } catch (e) { st.textContent = "error: " + e.message; }
};

$("#ac-remove").onclick = async () => {
  await fetch("/api/academy/key", { method: "DELETE" });
  $("#ac-probes").innerHTML = "";
  $("#ac-key-status").textContent = "key removed";
  $("#ac-remove").style.display = "none";
  loadTeachers();
};

$("#ac-distill").onclick = async () => {
  try {
    const r = await fetch("/api/academy/distill", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stories: +$("#ac-stories").value || 200 }),
    });
    const d = await r.json();
    if (!r.ok) $("#ac-stage").textContent = d.detail || "could not start";
    else { jobWasActive = true; academyTick(); }
  } catch (e) { $("#ac-stage").textContent = "error: " + e.message; }
};

$("#ac-train").onclick = async () => {
  try {
    const r = await fetch("/api/academy/train", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ steps: 8000 }),
    });
    const d = await r.json();
    if (!r.ok) $("#ac-stage").textContent = d.detail || "could not start";
    else { jobWasActive = true; academyTick(); }
  } catch (e) { $("#ac-stage").textContent = "error: " + e.message; }
};

$("#ac-stop").onclick = async () => {
  await fetch("/api/academy/stop", { method: "POST" });
  academyTick();
};

$("#ac-reset").onclick = async () => {
  const r = await fetch("/api/academy/reset", { method: "POST" });
  const d = await r.json();
  $("#ac-roster").textContent = r.ok
    ? `Base student restored (removed: ${d.removed.join(", ") || "nothing"}).`
    : (d.detail || "could not reset");
  academyRefresh();
};

// ---------- wiring ----------
$("#temp").oninput = () => ($("#tempv").textContent = (+$("#temp").value).toFixed(2));
$("#topk").oninput = () => ($("#topkv").textContent = $("#topk").value);
$("#len").oninput = () => ($("#lenv").textContent = $("#len").value);
$("#gen").onclick = generate;
$("#stop").onclick = () => (genAbort = true);
$("#analyze").onclick = analyze;
$("#layer").onchange = drawHeatmap;
$("#head").onchange = drawHeatmap;
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    document.querySelectorAll("[id^='tab-']").forEach((sec) => {
      sec.style.display = sec.id === "tab-" + t.dataset.tab ? "block" : "none";
    });
    if (t.dataset.tab === "academy") academyRefresh();
  };
});
$("#netlink").href = "https://jalammar.github.io/illustrated-gpt2/";

$("#prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); generate(); }
});

loadStatus();
loadTeachers();
