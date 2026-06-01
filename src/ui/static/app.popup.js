let heatmapThreshold = 2.0;
let csvStateThreshold = 0.5;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderBusySummary(node, message) {
  if (!node) return;
  node.innerHTML = `<div class="muted" style="display:flex;align-items:center;gap:8px;"><span aria-hidden="true" style="width:14px;height:14px;border:2px solid #c5cfda;border-top-color:#2d7d4f;border-radius:50%;display:inline-block;animation:spin 0.8s linear infinite;"></span><span>${escapeHtml(String(message || "Running ..."))}</span></div>`;
}

function renderResultInWindow(studyWindow, result) {
  if (!studyWindow || studyWindow.closed) return;
  const doc = studyWindow.document;
  doc.open();
  doc.write(`<!doctype html><html><head><meta charset="utf-8"/><title>Study Result</title>
  <style>
  body{font-family:Segoe UI,Arial,sans-serif;margin:12px;color:#17202a}
  .muted{color:#617080}.ok{color:#2d7d4f}.error{color:#a83d3d}
  .box{border:1px solid #d7dde4;border-radius:6px;padding:10px;margin:10px 0}
  .scroll{overflow:auto;max-height:520px;border:1px solid #d7dde4;border-radius:6px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:6px 8px;border-bottom:1px solid #d7dde4;text-align:left}
  th{position:sticky;top:0;background:#edf2f4}
  canvas{display:block}
  a{display:block;margin:4px 0}
  .status-row{display:flex;align-items:center;gap:8px}
  .spinner{width:14px;height:14px;border:2px solid #c5cfda;border-top-color:#4d6f8f;border-radius:50%;display:inline-block;animation:spin 0.8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  </style></head><body>
  <h2>Study Result</h2>
  <div id="summary" class="box"></div>
  <div id="hidden" class="box"></div>
  <div id="csv" class="box"></div>
  <div id="artifacts" class="box"></div>
  </body></html>`);
  doc.close();
  ensurePopupRenderer(studyWindow);
  const payload = JSON.parse(JSON.stringify(result || {}));
  if (typeof studyWindow.__renderPopupResult === "function") {
    studyWindow.__renderPopupResult(payload);
  }
}

function openAndRunInPopup(request) {
  const studyWindow = window.open("", "_blank");
  if (!studyWindow) return null;
  const doc = studyWindow.document;
  doc.open();
  doc.write(`<!doctype html><html><head><meta charset="utf-8"/><title>Study Result</title>
  <style>
  body{font-family:Segoe UI,Arial,sans-serif;margin:12px;color:#17202a}
  .muted{color:#617080}.ok{color:#2d7d4f}.error{color:#a83d3d}
  .box{border:1px solid #d7dde4;border-radius:6px;padding:10px;margin:10px 0}
  .scroll{overflow:auto;max-height:520px;border:1px solid #d7dde4;border-radius:6px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:6px 8px;border-bottom:1px solid #d7dde4;text-align:left}
  th{position:sticky;top:0;background:#edf2f4}
  canvas{display:block}
  a{display:block;margin:4px 0}
  .status-row{display:flex;align-items:center;gap:8px}
  .spinner{width:14px;height:14px;border:2px solid #c5cfda;border-top-color:#4d6f8f;border-radius:50%;display:inline-block;animation:spin 0.8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  </style></head><body>
  <h2>Study Result</h2>
  <div id="summary" class="box"></div>
  <div id="hidden" class="box"></div>
  <div id="csv" class="box"></div>
  <div id="artifacts" class="box"></div>
  </body></html>`);
  doc.close();
  ensurePopupRenderer(studyWindow);
  const payload = JSON.parse(JSON.stringify(request || {}));
  if (typeof studyWindow.__runPopupRequest === "function") {
    studyWindow.__runPopupRequest(payload);
  }
  return studyWindow;
}

function renderResultInWindowPopup(result) {
  const doc = document;
  const summary = doc.getElementById("summary");
  const hidden = doc.getElementById("hidden");
  const csv = doc.getElementById("csv");
  const artifacts = doc.getElementById("artifacts");
  if (!summary || !hidden || !csv || !artifacts) return;

  const ok = result && result.status === "ok";
  const busy = result && result.status === "busy";
  summary.innerHTML = `
    <div class="${ok ? "ok" : busy ? "muted" : "error"}">Status: ${escapeHtml((result && result.status) || "unknown")}</div>
    <div>Return code: ${escapeHtml(String(result && result.return_code != null ? result.return_code : ""))}</div>
    <div>Artifacts: ${result && Array.isArray(result.artifacts) ? result.artifacts.length : 0}</div>
  `;
  if (result && result.stderr) {
    const pre = doc.createElement("pre");
    pre.textContent = String(result.stderr);
    summary.appendChild(pre);
  }

  try {
    renderStudyMetaIntoDoc(doc, summary, result && result.hidden_state_heatmap ? result.hidden_state_heatmap : null);
  } catch (_metaErr) {
    // Keep rendering resilient.
  }

  try {
    renderHeatmapIntoDoc(doc, hidden, result && result.hidden_state_heatmap ? result.hidden_state_heatmap : null);
  } catch (heatErr) {
    hidden.innerHTML = `<h3>Hidden State</h3><div class="error">Heatmap render crashed: ${escapeHtml(String(heatErr && heatErr.message ? heatErr.message : heatErr))}</div>`;
  }
  renderCsvIntoDoc(doc, csv, result ? result.csv_preview : null);
  renderArtifactsIntoDoc(doc, artifacts, result && Array.isArray(result.artifacts) ? result.artifacts : []);
}

function ensurePopupRenderer(studyWindow) {
  if (!studyWindow || studyWindow.closed) return;
  const doc = studyWindow.document;
  if (!doc) return;
  if (typeof studyWindow.__renderPopupResult === "function") return;

  const fnList = [
    escapeHtml,
    heatmapColorFromValue,
    toDisplayProtocol,
    runPopupRequest,
    waitPopupTask,
    renderStudyMetaIntoDoc,
    renderCsvIntoDoc,
    renderArtifactsIntoDoc,
    renderHeatmapIntoDoc,
    renderTextOutputIntoDoc,
    renderOneHeatmapIntoDoc,
    renderTopLogitsTableIntoDoc,
    renderNeuronLogitsTableIntoDoc,
    renderResultInWindowPopup,
  ];
  const source = [
    "let heatmapThreshold = 2.0;",
    "let csvStateThreshold = 0.5;",
    ...fnList.map((fn) => fn.toString()),
    "window.__renderPopupResult = renderResultInWindowPopup;",
    "window.__runPopupRequest = runPopupRequest;",
  ].join("\n\n");

  const script = doc.createElement("script");
  script.type = "text/javascript";
  script.text = source;
  (doc.head || doc.documentElement).appendChild(script);
}

async function runPopupRequest(request) {
  const summary = document.getElementById("summary");
  const kind = String((request && request.kind) || "execute");
  const actionId = String((request && request.action_id) || "");
  renderBusySummary(summary, `Running ${actionId || kind} ...`);
  try {
    if (kind === "history") {
      const selectedName = String((request && request.selected_name) || "").trim();
      const url = selectedName
        ? `/api/history/layer-ffn-neuron/item?name=${encodeURIComponent(selectedName)}`
        : "/api/history/layer-ffn-neuron/latest";
      const resp = await fetch(url);
      const payload = await resp.json();
      renderResultInWindowPopup(payload);
      return;
    }

    const params = (request && request.params) || {};
    const response = await fetch("/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action_id: actionId, params }),
    });
    let payload = await response.json();
    if (payload && payload.status === "accepted" && payload.task_id) {
      payload = await waitPopupTask(payload.task_id, summary);
    }
    renderResultInWindowPopup(payload);
  } catch (err) {
    renderResultInWindowPopup({
      status: "error",
      return_code: null,
      stderr: String(err && err.message ? err.message : err),
      artifacts: [],
    });
  }
}

function waitPopupTask(taskId, summaryNode) {
  const tid = String(taskId || "").trim();
  if (!tid) {
    return Promise.resolve({
      status: "error",
      return_code: null,
      stderr: "Empty task id",
      artifacts: [],
    });
  }
  if (typeof EventSource === "undefined") {
    return (async () => {
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const resp = await fetch(`/api/tasks/${encodeURIComponent(tid)}`);
        const payload = await resp.json();
        if (payload && payload.status === "running") {
          const running = Number(payload.running_for_seconds || 0);
          const remain = Number(payload.estimated_remaining_seconds || 0);
          renderBusySummary(summaryNode, `Task running (${tid}): running=${running.toFixed(1)}s, remaining~${remain.toFixed(1)}s`);
          continue;
        }
        return payload;
      }
    })();
  }
  return new Promise((resolve) => {
    const es = new EventSource(`/api/tasks/${encodeURIComponent(tid)}/events`);
    es.onmessage = (evt) => {
      let payload = null;
      try {
        payload = JSON.parse(String(evt.data || "{}"));
      } catch (_err) {
        return;
      }
      if (!payload) return;
      if (payload.status === "running") {
        const running = Number(payload.running_for_seconds || 0);
        const remain = Number(payload.estimated_remaining_seconds || 0);
        renderBusySummary(summaryNode, `Task running (${tid}): running=${running.toFixed(1)}s, remaining~${remain.toFixed(1)}s`);
        return;
      }
      try { es.close(); } catch (_e) {}
      resolve(payload);
    };
    es.onerror = async () => {
      try { es.close(); } catch (_e) {}
      for (;;) {
        await new Promise((resolveDelay) => setTimeout(resolveDelay, 1000));
        const resp = await fetch(`/api/tasks/${encodeURIComponent(tid)}`);
        const payload = await resp.json();
        if (payload && payload.status === "running") continue;
        resolve(payload);
        return;
      }
    };
  });
}

function renderStudyMetaIntoDoc(doc, container, heatmap) {
  if (!heatmap || typeof heatmap !== "object") return;
  const omit = new Set([
    "matrix",
    "heatmaps",
    "top_logits",
    "top_logits_top100",
    "neuron_logits_rows",
    "neuron_logits_batches",
    "ui_tasks",
    "all_token_hidden_by_layer",
    "last_token_attention_by_layer",
  ]);
  const meta = {};
  Object.keys(heatmap).forEach((k) => {
    if (omit.has(k)) return;
    const v = heatmap[k];
    if (v === undefined) return;
    if (Array.isArray(v) && v.length > 50) return;
    if (typeof v === "object" && v !== null) return;
    meta[k] = v;
  });
  if (!Object.prototype.hasOwnProperty.call(meta, "study")) {
    meta.study = String(heatmap.study || "unknown");
  }
  const title = doc.createElement("div");
  title.style.marginTop = "8px";
  title.style.fontWeight = "600";
  title.textContent = "Study Meta (JSON)";
  container.appendChild(title);

  const pre = doc.createElement("pre");
  pre.className = "scroll";
  pre.style.maxHeight = "220px";
  pre.style.padding = "10px";
  pre.textContent = JSON.stringify(meta);
  container.appendChild(pre);
}

function renderArtifactsIntoDoc(doc, container, artifacts) {
  container.innerHTML = "<h3>Artifacts</h3>";
  artifacts.forEach((artifact) => {
    const link = doc.createElement("a");
    link.href = artifact.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = `${artifact.type.toUpperCase()} ${artifact.relative_path}`;
    container.appendChild(link);
  });
}

function loadStartupRequestFromUrl() {
  try {
    const url = new URL(window.location.href);
    const key = String(url.searchParams.get("req_key") || "").trim();
    if (!key) return null;
    const raw = localStorage.getItem(key);
    if (raw === null) return null;
    localStorage.removeItem(key);
    const payload = JSON.parse(raw);
    return payload && typeof payload === "object" ? payload : null;
  } catch (_err) {
    return null;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  // Expose aliases for compatibility with legacy opener/injection paths.
  window.__renderPopupResult = renderResultInWindowPopup;
  window.__runPopupRequest = runPopupRequest;
  const startupRequest = loadStartupRequestFromUrl();
  if (!startupRequest) {
    const summary = document.getElementById("summary");
    if (summary) {
      summary.innerHTML = `<div class="muted">Popup ready. No request payload was provided.</div>`;
    }
    return;
  }
  runPopupRequest(startupRequest);
});


