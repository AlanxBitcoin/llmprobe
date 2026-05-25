let actions = [];
let selectedAction = null;
let lastResult = null;
let heatmapThreshold = 2.0;
let busyCountdownTimer = null;

const actionsList = document.getElementById("actionsList");
const paramsForm = document.getElementById("paramsForm");
const actionTitle = document.getElementById("actionTitle");
const actionDescription = document.getElementById("actionDescription");
const runButton = document.getElementById("runButton");
const commandPreview = document.getElementById("commandPreview");
const serverStatus = document.getElementById("serverStatus");
const resultSummary = document.getElementById("resultSummary");
const csvContainer = document.getElementById("csvContainer");
const hiddenStateContainer = document.getElementById("hiddenStateContainer");
const artifactList = document.getElementById("artifactList");
const showCsvButton = document.getElementById("showCsvButton");
const histogramButton = document.getElementById("histogramButton");
const colorMapButton = document.getElementById("colorMapButton");
const chartsContainer = document.getElementById("chartsContainer");

async function loadActions() {
  setStatus("Loading");
  const response = await fetch("/api/actions");
  const payload = await response.json();
  actions = payload.actions || [];
  renderActions();
  if (actions.length > 0) {
    selectAction(actions[0].id);
  }
  setStatus("Ready");
}

function renderActions() {
  actionsList.innerHTML = "";
  actions.forEach((action) => {
    const button = document.createElement("button");
    button.className = "action-button";
    button.type = "button";
    button.dataset.actionId = action.id;
    button.innerHTML = `${escapeHtml(action.label)}<span>${escapeHtml(action.kind)}</span>`;
    button.addEventListener("click", () => selectAction(action.id));
    actionsList.appendChild(button);
  });
}

function selectAction(actionId) {
  selectedAction = actions.find((action) => action.id === actionId);
  if (!selectedAction) return;
  document.querySelectorAll(".action-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.actionId === actionId);
  });
  actionTitle.textContent = selectedAction.label;
  actionDescription.textContent = selectedAction.description || "";
  renderForm(selectedAction.fields || []);
  runButton.disabled = false;
  updateCommandPreview();
}

function renderForm(fields) {
  paramsForm.innerHTML = "";
  fields.forEach((field) => {
    const wrap = document.createElement("div");
    wrap.className = "field";
    wrap.dataset.fieldName = field.name || "";
    const label = document.createElement("label");
    label.textContent = field.label || field.name;
    wrap.appendChild(label);
    if (field.type === "display") {
      const display = document.createElement("div");
      display.className = "display-field";
      display.textContent = field.default || "";
      wrap.appendChild(display);
    } else {
      const input = document.createElement("input");
      input.name = field.name;
      input.type = field.type || "text";
      if (input.type === "checkbox") {
        input.checked = field.default !== false;
      } else {
        input.value = field.default ?? "";
      }
      if (field.min !== undefined) input.min = field.min;
      if (field.required) input.required = true;
      input.addEventListener(input.type === "checkbox" ? "change" : "input", updateCommandPreview);
      wrap.appendChild(input);
    }
    paramsForm.appendChild(wrap);
  });
  updateBosAssistantVisibility();
}

function collectParams() {
  const params = {};
  (selectedAction.fields || []).forEach((field) => {
    if (field.type === "display") return;
    const element = paramsForm.elements.namedItem(field.name);
    if (!element) return;
    if (field.type === "checkbox") {
      params[field.name] = Boolean(element.checked);
      return;
    }
    const raw = element.value;
    params[field.name] = field.type === "number" ? Number(raw) : raw;
  });
  return params;
}

function toDisplayProtocol(heatmap) {
  if (heatmap && typeof heatmap.protocol === "string" && heatmap.protocol.length > 0) {
    return heatmap.protocol;
  }
  if (heatmap && typeof heatmap.include_bos === "boolean") {
    const bos = Boolean(heatmap.include_bos);
    const assistant = Boolean(heatmap.include_assistant) && bos;
    if (!bos) return "bos0_assistant0";
    return assistant ? "bos1_assistant1" : "bos1_assistant0";
  }
  return "unknown";
}

function updateCommandPreview() {
  updateBosAssistantVisibility();
  if (!selectedAction) {
    commandPreview.textContent = "";
    return;
  }
  const params = collectParams();
  commandPreview.textContent = JSON.stringify(
    { action_id: selectedAction.id, params },
    null,
    2,
  );
}

function updateBosAssistantVisibility() {
  const bosInput = paramsForm.elements.namedItem("include_bos");
  const assistantInput = paramsForm.elements.namedItem("include_assistant");
  if (!bosInput || !assistantInput) return;

  const assistantWrap = assistantInput.closest(".field");
  if (!assistantWrap) return;

  const bosEnabled = Boolean(bosInput.checked);
  if (!bosEnabled) {
    assistantInput.checked = false;
    assistantWrap.style.display = "none";
    return;
  }
  assistantWrap.style.display = "";
}

async function runSelectedAction() {
  if (!selectedAction) return;
  const params = collectParams();
  const studyWindow = window.open("", "_blank");
  setStatus("Running");
  runButton.disabled = true;
  resultSummary.textContent = "Running study. Large model calls can take a while.";
  csvContainer.innerHTML = "";
  hiddenStateContainer.innerHTML = "";
  artifactList.innerHTML = "";
  try {
    const response = await fetch("/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action_id: selectedAction.id, params }),
    });
    lastResult = await response.json();
    renderResult(lastResult);
    renderResultInWindow(studyWindow, lastResult);
  } catch (error) {
    lastResult = null;
    resultSummary.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
    if (studyWindow && !studyWindow.closed) {
      studyWindow.document.body.innerHTML = `<pre style="padding:12px;color:#a83d3d;">${escapeHtml(error.message)}</pre>`;
    }
  } finally {
    runButton.disabled = false;
    setStatus("Ready");
  }
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
  </style></head><body>
  <h2>Study Result</h2>
  <div id="summary" class="box"></div>
  <div id="hidden" class="box"></div>
  <div id="csv" class="box"></div>
  <div id="artifacts" class="box"></div>
  </body></html>`);
  doc.close();

  const summary = doc.getElementById("summary");
  const hidden = doc.getElementById("hidden");
  const csv = doc.getElementById("csv");
  const artifacts = doc.getElementById("artifacts");
  if (!summary || !hidden || !csv || !artifacts) return;

  const ok = result.status === "ok";
  const busy = result.status === "busy";
  summary.innerHTML = `
    <div class="${ok ? "ok" : busy ? "muted" : "error"}">Status: ${escapeHtml(result.status || "unknown")}</div>
    <div>Return code: ${escapeHtml(String(result.return_code ?? ""))}</div>
    <div>Artifacts: ${result.artifacts ? result.artifacts.length : 0}</div>
  `;
  if (result.stderr) {
    const pre = doc.createElement("pre");
    pre.textContent = result.stderr;
    summary.appendChild(pre);
  }

  renderHeatmapIntoDoc(doc, hidden, result.hidden_state_heatmap);
  renderCsvIntoDoc(doc, csv, result.csv_preview);
  renderArtifactsIntoDoc(doc, artifacts, result.artifacts || []);
}

function renderCsvIntoDoc(doc, container, csvPreview) {
  container.innerHTML = "<h3>CSV Preview</h3>";
  if (!csvPreview || !csvPreview.headers || csvPreview.headers.length === 0) {
    container.append("No CSV preview available yet.");
    return;
  }
  const wrap = doc.createElement("div");
  wrap.className = "scroll";
  const table = doc.createElement("table");
  const thead = doc.createElement("thead");
  const hr = doc.createElement("tr");
  csvPreview.headers.forEach((h) => { const th = doc.createElement("th"); th.textContent = h; hr.appendChild(th); });
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = doc.createElement("tbody");
  csvPreview.rows.forEach((row) => {
    const tr = doc.createElement("tr");
    csvPreview.headers.forEach((h) => { const td = doc.createElement("td"); td.textContent = row[h] ?? ""; tr.appendChild(td); });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
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

function renderHeatmapIntoDoc(doc, container, heatmap) {
  container.innerHTML = "<h3>Hidden State</h3>";
  if (!heatmap || heatmap.ok === false || !Array.isArray(heatmap.matrix) || heatmap.matrix.length === 0) {
    const msg = doc.createElement("div");
    msg.className = "error";
    msg.textContent = "No heatmap result.";
    container.appendChild(msg);
    renderTopLogitsTableIntoDoc(
      doc,
      container,
      heatmap && heatmap.top_logits ? heatmap.top_logits : [],
      heatmap,
      "Top 15 Logits (with cosine similarity)",
      "logits_source",
      "logits_error",
    );
    renderTopLogitsTableIntoDoc(
      doc,
      container,
      heatmap && heatmap.top_logits_top100 ? heatmap.top_logits_top100 : [],
      heatmap,
      "Top 15 Logits (Penultimate Top-100 Intervention)",
      "top_logits_top100_source",
      "top_logits_top100_error",
    );
    return;
  }
  const rows = Number(heatmap.rows || heatmap.matrix.length);
  const cols = Number(heatmap.cols || (heatmap.matrix[0] ? heatmap.matrix[0].length : 0));
  const cell = 10;
  const info = doc.createElement("div");
  info.className = "muted";
  info.textContent = `word=${heatmap.word || ""}, source=${heatmap.cache_source || "unknown"}, protocol=${toDisplayProtocol(heatmap)}, logits=${heatmap.logits_source || "unknown"}`;
  container.appendChild(info);
  const hoverMeta = doc.createElement("div");
  hoverMeta.className = "muted";
  hoverMeta.textContent = "Hover: X=-, Y=-";
  container.appendChild(hoverMeta);
  const controls = doc.createElement("div");
  controls.className = "muted";
  const thresholdLabel = doc.createElement("span");
  thresholdLabel.textContent = "Threshold ";
  const thresholdSlider = doc.createElement("input");
  thresholdSlider.type = "range";
  thresholdSlider.min = "0.01";
  thresholdSlider.max = "10";
  thresholdSlider.step = "0.01";
  thresholdSlider.value = String(heatmapThreshold);
  const thresholdValue = doc.createElement("span");
  thresholdValue.textContent = Number(heatmapThreshold).toFixed(2);
  controls.appendChild(thresholdLabel);
  controls.appendChild(thresholdSlider);
  controls.appendChild(thresholdValue);
  container.appendChild(controls);
  const wrap = doc.createElement("div");
  wrap.className = "scroll";
  const canvas = doc.createElement("canvas");
  canvas.width = cols * cell;
  canvas.height = rows * cell;
  wrap.appendChild(canvas);
  container.appendChild(wrap);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  function drawWithThreshold(threshold) {
    const safeThreshold = Math.max(0.000001, Number(threshold) || 1);
    for (let r = 0; r < rows; r += 1) {
      const row = heatmap.matrix[r] || [];
      for (let c = 0; c < cols; c += 1) {
        const v = Number(row[c] || 0);
        const intensity = Math.min(1, Math.abs(v) / safeThreshold);
        const red = v > 0 ? Math.round(255 * intensity) : 0;
        const blue = v < 0 ? Math.round(255 * intensity) : 0;
        ctx.fillStyle = `rgb(${red},0,${blue})`;
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
  }

  thresholdSlider.addEventListener("input", () => {
    heatmapThreshold = Number(thresholdSlider.value);
    thresholdValue.textContent = heatmapThreshold.toFixed(2);
    drawWithThreshold(heatmapThreshold);
  });

  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / cell);
    const y = Math.floor((event.clientY - rect.top) / cell);
    if (x < 0 || x >= cols || y < 0 || y >= rows) {
      hoverMeta.textContent = "Hover: X=-, Y=-";
      return;
    }
    const row = heatmap.matrix[y] || [];
    const value = Number(row[x] || 0);
    hoverMeta.textContent = `Hover: X=${x} (neuron), Y=${y} (layer), value=${value.toFixed(6)}`;
  });
  canvas.addEventListener("mouseleave", () => {
    hoverMeta.textContent = "Hover: X=-, Y=-";
  });

  drawWithThreshold(heatmapThreshold);

  renderTopLogitsTableIntoDoc(
    doc,
    container,
    heatmap.top_logits || [],
    heatmap,
    "Top 15 Logits (with cosine similarity)",
    "logits_source",
    "logits_error",
  );
  renderTopLogitsTableIntoDoc(
    doc,
    container,
    heatmap.top_logits_top100 || [],
    heatmap,
    "Top 15 Logits (Penultimate Top-100 Intervention)",
    "top_logits_top100_source",
    "top_logits_top100_error",
  );
}

function renderTopLogitsTableIntoDoc(doc, container, rows, heatmap, titleText, sourceKey, errorKey) {
  if (!Array.isArray(rows) && !heatmap) return;
  const title = doc.createElement("h3");
  title.textContent = titleText || "Top Logits";
  container.appendChild(title);
  const info = doc.createElement("div");
  info.className = "muted";
  const source = heatmap && heatmap[sourceKey] ? heatmap[sourceKey] : "unknown";
  const err = heatmap && heatmap[errorKey] ? `, error=${heatmap[errorKey]}` : "";
  info.textContent = `logits_source=${source}${err}`;
  container.appendChild(info);
  if (sourceKey === "top_logits_top100_source") {
    const req = heatmap && heatmap.top100_request ? heatmap.top100_request : {};
    const meta = heatmap && heatmap.top100_intervention ? heatmap.top100_intervention : {};
    const topK = req.top_k_neurons ?? meta.keep_k;
    const layer = req.intervention_layer ?? meta.injection_layer_index;
    const params = doc.createElement("div");
    params.className = "muted";
    params.textContent = `top_k_neurons=${topK ?? "-"}, intervention_layer=${layer ?? "-"}`;
    container.appendChild(params);
  }
  if (!Array.isArray(rows) || rows.length === 0) {
    const empty = doc.createElement("div");
    empty.className = "muted";
    empty.textContent = "No logits rows returned.";
    container.appendChild(empty);
    return;
  }

  const wrap = doc.createElement("div");
  wrap.className = "scroll";
  const table = doc.createElement("table");
  const thead = doc.createElement("thead");
  const headerRow = doc.createElement("tr");
  ["rank", "token_id", "token", "text", "logit", "cosine_similarity"].forEach((header) => {
    const th = doc.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = doc.createElement("tbody");
  rows.forEach((row) => {
    const tr = doc.createElement("tr");
    ["rank", "token_id", "token", "text", "logit", "cosine_similarity"].forEach((key) => {
      const td = doc.createElement("td");
      const value = row[key];
      if (typeof value === "number" && (key === "logit" || key === "cosine_similarity")) {
        td.textContent = value.toFixed(6);
      } else {
        td.textContent = value ?? "";
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}

function renderResult(result) {
  if (busyCountdownTimer) {
    clearInterval(busyCountdownTimer);
    busyCountdownTimer = null;
  }
  const ok = result.status === "ok";
  const busy = result.status === "busy";
  resultSummary.innerHTML = `
    <div class="${ok ? "ok" : busy ? "muted" : "error"}">Status: ${escapeHtml(result.status || "unknown")}</div>
    <div>Return code: ${escapeHtml(String(result.return_code ?? ""))}</div>
    <div>Artifacts: ${result.artifacts ? result.artifacts.length : 0}</div>
  `;
  if (busy) {
    const note = document.createElement("div");
    const running = Number(result.running_for_seconds || 0);
    let remaining = Number(result.estimated_remaining_seconds || 0);
    note.textContent = `Task busy: running=${running.toFixed(1)}s, remaining≈${remaining.toFixed(1)}s`;
    resultSummary.appendChild(note);
    busyCountdownTimer = setInterval(() => {
      remaining = Math.max(0, remaining - 1);
      note.textContent = `Task busy: running=${running.toFixed(1)}s, remaining≈${remaining.toFixed(1)}s`;
      if (remaining <= 0) {
        clearInterval(busyCountdownTimer);
        busyCountdownTimer = null;
      }
    }, 1000);
  }
  if (result.stderr) {
    const pre = document.createElement("pre");
    pre.className = "command-preview";
    pre.textContent = result.stderr;
    resultSummary.appendChild(pre);
  }
  renderCsv(result.csv_preview);
  renderHiddenStateHeatmap(result.hidden_state_heatmap);
  renderArtifacts(result.artifacts || []);
  const hasCsv = !!(result.csv_preview && result.csv_preview.rows && result.csv_preview.rows.length);
  showCsvButton.disabled = !hasCsv;
  histogramButton.disabled = !hasCsv;
  colorMapButton.disabled = !hasCsv;
}

function renderHiddenStateHeatmap(heatmap) {
  hiddenStateContainer.innerHTML = "";
  if (heatmap && heatmap.ok === false) {
    const msg = document.createElement("div");
    msg.className = "error";
    msg.textContent = `Hidden-state heatmap requires a single-token word. token_count=${Number(heatmap.token_count || 0)}`;
    hiddenStateContainer.appendChild(msg);
    renderTopLogitsTable(
      heatmap.top_logits || [],
      heatmap,
      "Top 15 Logits (with cosine similarity)",
      "logits_source",
      "logits_error",
    );
    renderTopLogitsTable(
      heatmap.top_logits_top100 || [],
      heatmap,
      "Top 15 Logits (Penultimate Top-100 Intervention)",
      "top_logits_top100_source",
      "top_logits_top100_error",
    );
    return;
  }
  if (!heatmap || !Array.isArray(heatmap.matrix) || heatmap.matrix.length === 0) {
    return;
  }
  const rows = Number(heatmap.rows || heatmap.matrix.length);
  const cols = Number(heatmap.cols || (heatmap.matrix[0] ? heatmap.matrix[0].length : 0));
  if (!rows || !cols) return;

  const cell = 10;
  const canvas = document.createElement("canvas");
  canvas.className = "heatmap-canvas";
  canvas.width = cols * cell;
  canvas.height = rows * cell;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let maxAbs = 0;
  for (let r = 0; r < rows; r += 1) {
    const row = heatmap.matrix[r] || [];
    for (let c = 0; c < cols; c += 1) {
      const v = Number(row[c] || 0);
      const a = Math.abs(v);
      if (a > maxAbs) maxAbs = a;
    }
  }
  if (!Number.isFinite(maxAbs) || maxAbs <= 0) maxAbs = 1;

  function drawWithThreshold(threshold) {
    const safeThreshold = Math.max(0.000001, Number(threshold) || 1);
    for (let r = 0; r < rows; r += 1) {
      const row = heatmap.matrix[r] || [];
      for (let c = 0; c < cols; c += 1) {
        const v = Number(row[c] || 0);
        const intensity = Math.min(1, Math.abs(v) / safeThreshold);
        let red = 0;
        let blue = 0;
        if (v > 0) red = Math.round(255 * intensity);
        else if (v < 0) blue = Math.round(255 * intensity);
        ctx.fillStyle = `rgb(${red},0,${blue})`;
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
  }

  const meta = document.createElement("p");
  meta.className = "heatmap-meta";
  meta.textContent = `Heatmap: ${rows} x ${cols}, cell=${cell}x${cell}, word=${heatmap.word || ""}, source=${heatmap.cache_source || "unknown"}, logits=${heatmap.logits_source || "unknown"}, zero=black, positive=red, negative=blue`;
  hiddenStateContainer.appendChild(meta);
  if (heatmap.logits_source === "cache_miss_no_compute") {
    const tip = document.createElement("div");
    tip.className = "muted";
    tip.textContent = "Logits cache miss: skipped compute to keep this request GPU-free.";
    hiddenStateContainer.appendChild(tip);
  }
  if (heatmap.logits_error) {
    const logitsErr = document.createElement("div");
    logitsErr.className = "error";
    logitsErr.textContent = `Logits skipped: ${heatmap.logits_error}`;
    hiddenStateContainer.appendChild(logitsErr);
  }

  const hoverMeta = document.createElement("p");
  hoverMeta.className = "heatmap-meta";
  hoverMeta.textContent = "Hover: X=-, Y=-";
  hiddenStateContainer.appendChild(hoverMeta);

  const controls = document.createElement("div");
  controls.className = "heatmap-controls";
  const thresholdLabel = document.createElement("span");
  thresholdLabel.textContent = "Threshold";
  const thresholdSlider = document.createElement("input");
  thresholdSlider.type = "range";
  thresholdSlider.min = "0.01";
  thresholdSlider.max = "10";
  thresholdSlider.step = "0.01";
  thresholdSlider.value = String(heatmapThreshold);
  const thresholdValue = document.createElement("span");
  thresholdValue.textContent = heatmapThreshold.toFixed(2);
  thresholdSlider.addEventListener("input", () => {
    heatmapThreshold = Number(thresholdSlider.value);
    thresholdValue.textContent = heatmapThreshold.toFixed(2);
    drawWithThreshold(heatmapThreshold);
  });
  controls.appendChild(thresholdLabel);
  controls.appendChild(thresholdSlider);
  controls.appendChild(thresholdValue);
  hiddenStateContainer.appendChild(controls);

  const wrap = document.createElement("div");
  wrap.className = "heatmap-scroll";
  wrap.appendChild(canvas);
  hiddenStateContainer.appendChild(wrap);

  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / cell);
    const y = Math.floor((event.clientY - rect.top) / cell);
    if (x < 0 || x >= cols || y < 0 || y >= rows) {
      hoverMeta.textContent = "Hover: X=-, Y=-";
      return;
    }
    const row = heatmap.matrix[y] || [];
    const value = Number(row[x] || 0);
    hoverMeta.textContent = `Hover: X=${x} (neuron), Y=${y} (layer), value=${value.toFixed(6)}`;
  });
  canvas.addEventListener("mouseleave", () => {
    hoverMeta.textContent = "Hover: X=-, Y=-";
  });

  drawWithThreshold(heatmapThreshold);
  renderTopLogitsTable(
    heatmap.top_logits || [],
    heatmap,
    "Top 15 Logits (with cosine similarity)",
    "logits_source",
    "logits_error",
  );
  renderTopLogitsTable(
    heatmap.top_logits_top100 || [],
    heatmap,
    "Top 15 Logits (Penultimate Top-100 Intervention)",
    "top_logits_top100_source",
    "top_logits_top100_error",
  );
}

function renderTopLogitsTable(rows, heatmap, titleText, sourceKey, errorKey) {
  if (!Array.isArray(rows) && !heatmap) return;
  const title = document.createElement("p");
  title.className = "heatmap-meta";
  title.textContent = titleText || "Top Logits";
  hiddenStateContainer.appendChild(title);
  const info = document.createElement("p");
  info.className = "heatmap-meta";
  const source = heatmap && heatmap[sourceKey] ? heatmap[sourceKey] : "unknown";
  const err = heatmap && heatmap[errorKey] ? `, error=${heatmap[errorKey]}` : "";
  info.textContent = `logits_source=${source}${err}`;
  hiddenStateContainer.appendChild(info);
  if (sourceKey === "top_logits_top100_source") {
    const req = heatmap && heatmap.top100_request ? heatmap.top100_request : {};
    const meta = heatmap && heatmap.top100_intervention ? heatmap.top100_intervention : {};
    const topK = req.top_k_neurons ?? meta.keep_k;
    const layer = req.intervention_layer ?? meta.injection_layer_index;
    const params = document.createElement("p");
    params.className = "heatmap-meta";
    params.textContent = `top_k_neurons=${topK ?? "-"}, intervention_layer=${layer ?? "-"}`;
    hiddenStateContainer.appendChild(params);
  }
  if (!Array.isArray(rows) || rows.length === 0) {
    const empty = document.createElement("p");
    empty.className = "heatmap-meta";
    empty.textContent = "No logits rows returned.";
    hiddenStateContainer.appendChild(empty);
    return;
  }

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["rank", "token_id", "token", "text", "logit", "cosine_similarity"].forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    ["rank", "token_id", "token", "text", "logit", "cosine_similarity"].forEach((key) => {
      const td = document.createElement("td");
      const value = row[key];
      if (typeof value === "number" && (key === "logit" || key === "cosine_similarity")) {
        td.textContent = value.toFixed(6);
      } else {
        td.textContent = value ?? "";
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  const wrap = document.createElement("div");
  wrap.className = "csv-container";
  wrap.appendChild(table);
  hiddenStateContainer.appendChild(wrap);
}

function renderCsv(csvPreview) {
  csvContainer.innerHTML = "";
  if (!csvPreview || !csvPreview.headers || csvPreview.headers.length === 0) {
    csvContainer.textContent = "No CSV preview available yet.";
    return;
  }
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  csvPreview.headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  csvPreview.rows.forEach((row) => {
    const tr = document.createElement("tr");
    csvPreview.headers.forEach((header) => {
      const td = document.createElement("td");
      td.textContent = row[header] ?? "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  csvContainer.appendChild(table);
}

function renderArtifacts(artifacts) {
  artifactList.innerHTML = "";
  artifacts.forEach((artifact) => {
    const link = document.createElement("a");
    link.href = artifact.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = `${artifact.type.toUpperCase()} · ${artifact.relative_path}`;
    artifactList.appendChild(link);
  });
}

function addHistogram() {
  const csv = lastResult && lastResult.csv_preview;
  if (!csv || !csv.rows || csv.rows.length === 0) return;
  const numericHeader = csv.headers.find((header) =>
    csv.rows.some((row) => Number.isFinite(Number(row[header]))),
  );
  if (!numericHeader) return;
  const values = csv.rows.map((row) => Number(row[numericHeader])).filter(Number.isFinite).slice(0, 40);
  const max = Math.max(...values, 1);
  const width = Math.max(values.length * 24, 360);
  const height = 220;
  const bars = values.map((value, index) => {
    const barHeight = Math.max(2, (value / max) * 170);
    const x = 12 + index * 24;
    const y = height - barHeight - 24;
    return `<rect x="${x}" y="${y}" width="16" height="${barHeight}" fill="#256f75"></rect>`;
  }).join("");
  addChart(`Histogram · ${numericHeader}`, `<svg width="${width}" height="${height}" role="img">${bars}</svg>`);
}

function addColorMap() {
  const csv = lastResult && lastResult.csv_preview;
  if (!csv || !csv.rows || csv.rows.length === 0) return;
  const headers = csv.headers.slice(0, 8);
  const rows = csv.rows.slice(0, 24);
  const cell = 24;
  const headerOffset = 120;
  const width = headerOffset + headers.length * cell + 20;
  const height = 40 + rows.length * cell;
  const cells = [];
  rows.forEach((row, r) => {
    headers.forEach((header, c) => {
      const raw = row[header] ?? "";
      const score = hashString(String(raw)) % 100;
      const color = `hsl(${170 + score}, 42%, ${42 + (score % 24)}%)`;
      cells.push(`<rect x="${headerOffset + c * cell}" y="${30 + r * cell}" width="${cell}" height="${cell}" fill="${color}"></rect>`);
    });
  });
  addChart("Color Map", `<svg width="${width}" height="${height}" role="img">${cells.join("")}</svg>`);
}

function addChart(title, html) {
  const card = document.createElement("div");
  card.className = "chart-card";
  card.innerHTML = `<h3>${escapeHtml(title)}</h3>${html}`;
  chartsContainer.prepend(card);
}

function setStatus(value) {
  serverStatus.textContent = value;
}

function hashString(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

runButton.addEventListener("click", runSelectedAction);
showCsvButton.addEventListener("click", () => lastResult && renderCsv(lastResult.csv_preview));
histogramButton.addEventListener("click", addHistogram);
colorMapButton.addEventListener("click", addColorMap);

loadActions().catch((error) => {
  setStatus("Error");
  resultSummary.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
});
