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
const inlineParamResult = document.createElement("div");
inlineParamResult.id = "inlineParamResult";
inlineParamResult.className = "command-preview";
inlineParamResult.style.marginTop = "8px";
inlineParamResult.style.display = "none";
inlineParamResult.style.maxHeight = "none";
inlineParamResult.style.overflow = "visible";
commandPreview.insertAdjacentElement("afterend", inlineParamResult);
const serverStatus = document.getElementById("serverStatus");
const resultSummary = document.getElementById("resultSummary");
const csvContainer = document.getElementById("csvContainer");
const hiddenStateContainer = document.getElementById("hiddenStateContainer");
const artifactList = document.getElementById("artifactList");
const showCsvButton = document.getElementById("showCsvButton");
const histogramButton = document.getElementById("histogramButton");
const colorMapButton = document.getElementById("colorMapButton");
const chartsContainer = document.getElementById("chartsContainer");
const chatTranscript = document.getElementById("chatTranscript");
const chatInput = document.getElementById("chatInput");
const chatSendButton = document.getElementById("chatSendButton");
const chatClearButton = document.getElementById("chatClearButton");
const chatTemperature = document.getElementById("chatTemperature");
const chatMaxTokens = document.getElementById("chatMaxTokens");
const chatIncludeAssistantMarker = document.getElementById("chatIncludeAssistantMarker");
const chatLayerNeuronEnabled = document.getElementById("chatLayerNeuronEnabled");
const chatLayerNeuronLayer = document.getElementById("chatLayerNeuronLayer");
const chatLayerNeuronId = document.getElementById("chatLayerNeuronId");
const chatLayerNeuronValue = document.getElementById("chatLayerNeuronValue");
const chatLayerNeuronToken = document.getElementById("chatLayerNeuronToken");
const chatFfnNeuronEnabled = document.getElementById("chatFfnNeuronEnabled");
const chatFfnNeuronLayer = document.getElementById("chatFfnNeuronLayer");
const chatFfnNeuronId = document.getElementById("chatFfnNeuronId");
const chatFfnNeuronValue = document.getElementById("chatFfnNeuronValue");
const openHistoryButton = document.createElement("button");
openHistoryButton.type = "button";
openHistoryButton.textContent = "Open History Result";
openHistoryButton.style.marginLeft = "8px";
openHistoryButton.style.display = "none";
runButton.insertAdjacentElement("afterend", openHistoryButton);
const historySelect = document.createElement("select");
historySelect.style.marginLeft = "8px";
historySelect.style.display = "none";
historySelect.style.minWidth = "260px";
runButton.insertAdjacentElement("afterend", historySelect);

let chatMessages = [];

// Common grammar/function words.
const GRAMMAR_WORD_SET = new Set([
  "the","a","an","this","that","these","those","some","any","each","every","all","both","either","neither","no",
  "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their",
  "mine","yours","hers","ours","theirs","myself","yourself","himself","herself","itself","ourselves","themselves",
  "is","am","are","was","were","be","being","been","do","does","did","have","has","had",
  "can","could","may","might","must","shall","should","will","would",
  "and","or","but","so","if","then","than","as","because","while","when","where","who","whom","which","what","why","how",
  "to","of","in","on","at","for","from","with","by","about","into","over","after","before","between","under","through","during",
  "not","n't"
]);

// High-frequency symbols / separators / chat-template tokens.
const GRAMMAR_SYMBOL_SET = new Set([
  " ", "\t", "\n", "\r", "\r\n",
  ",", ".", ":", ";", "!", "?", "'", "\"", "`",
  "-", "--", "—", "_", "/", "\\", "|", "~",
  "(", ")", "[", "]", "{", "}",
  "<", ">", "=", "+", "*", "&", "%", "$", "#", "@",
  "<0x0A>", "<0x0D>", "\\n", "\\r", "\\r\\n",
  "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
  "assistant", "user", "system"
]);

function normalizeRawTokenForSymbolFilter(value) {
  let t = String(value ?? "");
  // Remove BPE boundary marks only; keep punctuation/newline semantics.
  t = t.replace(/[Ġ▁]/g, "");
  return t.toLowerCase().trim();
}

function normalizeWordTokenForGrammarFilter(value) {
  let t = String(value ?? "").toLowerCase();
  t = t.replace(/[Ġ▁\s\r\n\t]/g, "");
  // Keep apostrophe for tokens like n't.
  t = t.replace(/[^a-z']/g, "");
  return t;
}

function isGrammarTokenLike(item) {
  if (!item) return false;
  const rawCandidates = [
    normalizeRawTokenForSymbolFilter(item.text),
    normalizeRawTokenForSymbolFilter(item.token),
  ];
  if (rawCandidates.some((c) => c && GRAMMAR_SYMBOL_SET.has(c))) {
    return true;
  }
  const wordCandidates = [
    normalizeWordTokenForGrammarFilter(item.text),
    normalizeWordTokenForGrammarFilter(item.token),
  ];
  return wordCandidates.some((c) => c && GRAMMAR_WORD_SET.has(c));
}

function heatmapColorFromValue(value, threshold) {
  const safeThreshold = Math.max(0.000001, Number(threshold) || 1);
  const intensity = Math.min(1, Math.abs(Number(value) || 0) / safeThreshold);
  const red = value > 0 ? Math.round(255 * intensity) : 0;
  const blue = value < 0 ? Math.round(255 * intensity) : 0;
  if (value > 0 || value < 0) return `rgb(${red},0,${blue})`;
  return "rgb(0,0,0)";
}

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
  updateHistoryButtonVisibility();
  if (selectedAction && selectedAction.id === "study_layer_ffn_neuron_logits_table") {
    loadFfnHistoryList();
  }
  updateCommandPreview();
  clearInlineParamResult();
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
      if (field.max !== undefined) input.max = field.max;
      if (field.step !== undefined) input.step = field.step;
      if (field.required) input.required = true;
      input.addEventListener(input.type === "checkbox" ? "change" : "input", updateCommandPreview);
      wrap.appendChild(input);
    }
    paramsForm.appendChild(wrap);
  });
  updateBosAssistantVisibility();
  updatePrefixContextVisibility();
  updateBatchNameDropdown();
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
  updatePrefixContextVisibility();
  updateBatchNameDropdownVisibility();
  updateHistoryButtonVisibility();
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

function updatePrefixContextVisibility() {
  const usePrefixInput = paramsForm.elements.namedItem("use_prefix_context");
  const prefixTextInput = paramsForm.elements.namedItem("prefix_text");
  if (!usePrefixInput || !prefixTextInput) return;
  const prefixWrap = prefixTextInput.closest(".field");
  if (!prefixWrap) return;
  prefixWrap.style.display = Boolean(usePrefixInput.checked) ? "" : "none";
}

function updateHistoryButtonVisibility() {
  if (!openHistoryButton || !historySelect) return;
  const visible = Boolean(selectedAction && selectedAction.id === "study_layer_ffn_neuron_logits_table");
  openHistoryButton.style.display = visible ? "inline-block" : "none";
  historySelect.style.display = visible ? "inline-block" : "none";
}

async function loadFfnHistoryList() {
  if (!historySelect) return;
  historySelect.innerHTML = "";
  const loading = document.createElement("option");
  loading.value = "";
  loading.textContent = "Loading history...";
  historySelect.appendChild(loading);
  try {
    const resp = await fetch("/api/history/layer-ffn-neuron/list");
    const payload = await resp.json();
    const items = Array.isArray(payload && payload.items) ? payload.items : [];
    historySelect.innerHTML = "";
    const first = document.createElement("option");
    first.value = "";
    first.textContent = items.length > 0 ? "Latest" : "No history";
    historySelect.appendChild(first);
    items.forEach((item) => {
      const name = String((item && item.name) || "").trim();
      if (!name) return;
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      historySelect.appendChild(opt);
    });
  } catch (_err) {
    historySelect.innerHTML = "";
    const failed = document.createElement("option");
    failed.value = "";
    failed.textContent = "History load failed";
    historySelect.appendChild(failed);
  }
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

function updateBatchNameDropdownVisibility() {
  const batchSelect = paramsForm.querySelector("select[name='batch_name_picker']");
  const batchInput = paramsForm.elements.namedItem("batch_name");
  if (!batchSelect || !batchInput) return;
  const wrap = batchSelect.closest(".field-inline");
  if (!wrap) return;
  wrap.style.display = selectedAction && selectedAction.id === "study_single_word_hidden_state_batch_average" ? "" : "none";
}

async function updateBatchNameDropdown() {
  if (!selectedAction || selectedAction.id !== "study_single_word_hidden_state_batch_average") return;
  const batchInput = paramsForm.elements.namedItem("batch_name");
  const wordsInput = paramsForm.elements.namedItem("words_csv");
  const batchField = paramsForm.querySelector(".field[data-field-name='batch_name']");
  if (!batchInput || !wordsInput || !batchField) return;
  if (batchField.querySelector("select[name='batch_name_picker']")) {
    updateBatchNameDropdownVisibility();
    return;
  }

  const inline = document.createElement("div");
  inline.className = "field-inline";
  const picker = document.createElement("select");
  picker.name = "batch_name_picker";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Choose saved batch";
  picker.appendChild(placeholder);
  inline.appendChild(picker);
  batchField.appendChild(inline);

  try {
    const resp = await fetch("/api/batches");
    const payload = await resp.json();
    const batches = Array.isArray(payload.batches) ? payload.batches : [];
    batches.forEach((item) => {
      const name = String(item.name || "").trim();
      if (!name) return;
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      option.dataset.wordsCsv = String(item.words_csv || "");
      picker.appendChild(option);
    });
  } catch (_error) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Failed to load batches";
    picker.appendChild(option);
  }

  picker.addEventListener("change", () => {
    const opt = picker.selectedOptions[0];
    const name = opt ? String(opt.value || "") : "";
    const wordsCsv = opt ? String(opt.dataset.wordsCsv || "") : "";
    if (name) batchInput.value = name;
    if (wordsCsv) wordsInput.value = wordsCsv;
    updateCommandPreview();
  });
  updateBatchNameDropdownVisibility();
}

async function runSelectedAction() {
  if (!selectedAction) return;
  const params = collectParams();
  const inlineOnly = selectedAction.id === "study_sentence_next_word";
  clearInlineParamResult();
  const studyWindow = inlineOnly ? null : window.open("", "_blank");
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
    if (lastResult && lastResult.status === "accepted" && lastResult.task_id) {
      resultSummary.textContent = `Task accepted. task_id=${lastResult.task_id}. Waiting for server push...`;
      await streamTaskUntilDone(lastResult.task_id, studyWindow);
      return;
    }
    if (inlineOnly) {
      renderResult(lastResult);
    } else {
      renderResultInWindow(studyWindow, lastResult);
    }
    if (lastResult && lastResult.status === "ok") {
      resultSummary.textContent = inlineOnly
        ? "Study finished. Results are shown below."
        : "Study finished. Results are shown in the popup window.";
    } else if (lastResult && lastResult.status === "busy") {
      resultSummary.textContent = "Study is busy. Please wait and retry.";
    } else {
      resultSummary.textContent = "Study failed. Check the popup window for details.";
    }
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

function clearInlineParamResult() {
  inlineParamResult.innerHTML = "";
  inlineParamResult.style.display = "none";
}

function renderSentenceNextWordInParams(result) {
  if (!selectedAction || selectedAction.id !== "study_sentence_next_word") return;
  const heatmap = result && result.hidden_state_heatmap ? result.hidden_state_heatmap : {};
  const rows = Array.isArray(heatmap.top_logits) ? heatmap.top_logits : [];
  const sentence = String(heatmap.sentence || "");
  const source = String(heatmap.logits_source || "unknown");
  const err = heatmap.logits_error ? String(heatmap.logits_error) : "";

  const parts = [];
  parts.push(`<div><strong>Sentence Next Word Result</strong></div>`);
  if (sentence) parts.push(`<div>sentence=${escapeHtml(sentence)}</div>`);
  parts.push(`<div>logits_source=${escapeHtml(source)}${err ? `, error=${escapeHtml(err)}` : ""}</div>`);
  if (!rows.length) {
    parts.push(`<div>No logits rows returned.</div>`);
    inlineParamResult.innerHTML = parts.join("");
    inlineParamResult.style.display = "";
    return;
  }

  const tableRows = rows.map((row) => {
    const rank = Number(row.rank || 0);
    const tokenId = Number(row.token_id || 0);
    const text = escapeHtml(String(row.text ?? ""));
    const logit = Number(row.logit || 0);
    return `<tr><td>${rank}</td><td>${tokenId}</td><td>${text}</td><td>${logit.toFixed(6)}</td></tr>`;
  }).join("");

  parts.push(
    `<div style="margin-top:6px;">` +
      `<table style="width:100%;border-collapse:collapse;">` +
        `<thead><tr><th style="text-align:left;">rank</th><th style="text-align:left;">token_id</th><th style="text-align:left;">text</th><th style="text-align:left;">logit</th></tr></thead>` +
        `<tbody>${tableRows}</tbody>` +
      `</table>` +
    `</div>`,
  );
  inlineParamResult.innerHTML = parts.join("");
  inlineParamResult.style.display = "";
}


async function openLatestFfnHistory() {
  const studyWindow = window.open("", "_blank");
  setStatus("Loading History");
  try {
    const selectedName = historySelect ? String(historySelect.value || "").trim() : "";
    const url = selectedName
      ? `/api/history/layer-ffn-neuron/item?name=${encodeURIComponent(selectedName)}`
      : "/api/history/layer-ffn-neuron/latest";
    const resp = await fetch(url);
    const payload = await resp.json();
    if (!payload || payload.status !== "ok") {
      const msg = payload && payload.stderr ? String(payload.stderr) : "No history found.";
      if (studyWindow && !studyWindow.closed) {
        studyWindow.document.body.innerHTML = `<pre style="padding:12px;color:#a83d3d;">${escapeHtml(msg)}</pre>`;
      }
      resultSummary.textContent = msg;
      return;
    }
    renderResultInWindow(studyWindow, payload);
    resultSummary.textContent = "Opened latest FFN history result.";
  } catch (error) {
    const msg = String(error && error.message ? error.message : error);
    if (studyWindow && !studyWindow.closed) {
      studyWindow.document.body.innerHTML = `<pre style="padding:12px;color:#a83d3d;">${escapeHtml(msg)}</pre>`;
    }
    resultSummary.textContent = `Open history failed: ${msg}`;
  } finally {
    setStatus("Ready");
  }
}

async function pollTaskUntilDone(taskId, studyWindow) {
  const tid = String(taskId || "").trim();
  if (!tid) return;
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const resp = await fetch(`/api/tasks/${encodeURIComponent(tid)}`);
    const payload = await resp.json();
    if (!payload) continue;
    if (payload.status === "running") {
      const running = Number(payload.running_for_seconds || 0);
      const remain = Number(payload.estimated_remaining_seconds || 0);
      resultSummary.textContent = `Task running (${tid}) ... running=${running.toFixed(1)}s, remaining≈${remain.toFixed(1)}s`;
      continue;
    }
    lastResult = payload;
    if (studyWindow) {
      renderResultInWindow(studyWindow, payload);
    } else {
      renderResult(payload);
    }
    if (payload.status === "ok") {
      resultSummary.textContent = "Study finished. Results are shown in the popup window.";
    } else {
      resultSummary.textContent = "Study failed. Check the popup window for details.";
    }
    return;
  }
}

function streamTaskUntilDone(taskId, studyWindow) {
  const tid = String(taskId || "").trim();
  if (!tid || typeof EventSource === "undefined") {
    return pollTaskUntilDone(taskId, studyWindow);
  }
  return new Promise((resolve) => {
    let done = false;
    const closeAndResolve = () => {
      if (done) return;
      done = true;
      try { es.close(); } catch (_e) {}
      resolve();
    };
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
        resultSummary.textContent = `Task running (${tid}) ... running=${running.toFixed(1)}s, remaining≈${remain.toFixed(1)}s`;
        return;
      }
      lastResult = payload;
      if (studyWindow) {
        renderResultInWindow(studyWindow, payload);
      } else {
        renderResult(payload);
      }
      if (payload.status === "ok") {
        resultSummary.textContent = "Study finished. Results are shown in the popup window.";
      } else {
        resultSummary.textContent = "Study failed. Check the popup window for details.";
      }
      closeAndResolve();
    };
    es.onerror = () => {
      // Fallback to polling if SSE stream is interrupted/unavailable.
      if (done) return;
      try { es.close(); } catch (_e) {}
      pollTaskUntilDone(tid, studyWindow).finally(closeAndResolve);
    };
  });
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
  const heatmapSyncGroup = { wraps: [], syncing: false };
  const hasMatrix = Boolean(Array.isArray(heatmap && heatmap.matrix) && heatmap.matrix.length > 0);
  const hasHeatmaps = Boolean(Array.isArray(heatmap && heatmap.heatmaps) && heatmap.heatmaps.length > 0);
  const hasTasks = Boolean(Array.isArray(heatmap && heatmap.ui_tasks) && heatmap.ui_tasks.length > 0);
  if (!heatmap || heatmap.ok === false || (!hasMatrix && !hasHeatmaps && !hasTasks)) {
    const msg = doc.createElement("div");
    msg.className = "error";
    if (heatmap && heatmap.ok === false) {
      const reason = String(heatmap.reason || "unknown");
      const tokenCount = Number(heatmap.token_count || 0);
      const tokenNote = Number.isFinite(tokenCount) && tokenCount > 0 ? ` (token_count=${tokenCount})` : "";
      msg.textContent = `Heatmap failed: ${reason}${tokenNote}`;
    } else {
      msg.textContent = "No heatmap result.";
    }
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
  const info = doc.createElement("div");
  info.className = "muted";
  info.textContent = `word=${heatmap.word || ""}, source=${heatmap.cache_source || "unknown"}, protocol=${toDisplayProtocol(heatmap)}, logits=${heatmap.logits_source || "unknown"}`;
  container.appendChild(info);

  const tasks = Array.isArray(heatmap.ui_tasks) ? heatmap.ui_tasks : [];
  if (tasks.length > 0) {
    tasks.forEach((task) => {
      try {
        const name = String((task && task.name) || "");
        const valueKey = String((task && task.value_key) || "");
        const value = valueKey ? heatmap[valueKey] : undefined;
        if (name === "render_heatmap") {
          // Support:
          // 1) value_key -> 2D matrix
          // 2) value_key -> [{title,matrix}, ...]
          // 3) fallback to heatmap.heatmaps / heatmap.matrix
          if (Array.isArray(value) && value.length > 0 && Array.isArray(value[0]) && Array.isArray(value[0][0]) === false) {
            renderOneHeatmapIntoDoc(doc, container, value, "Hidden State Heatmap", heatmapSyncGroup);
            return;
          }
          if (Array.isArray(value) && value.length > 0 && value[0] && typeof value[0] === "object" && Array.isArray(value[0].matrix)) {
            value.forEach((hm, idx) => {
              renderOneHeatmapIntoDoc(
                doc,
                container,
                hm.matrix,
                String(hm.title || `Heatmap ${idx + 1}`),
                heatmapSyncGroup,
              );
            });
            return;
          }
          const fallbackHeatmaps = Array.isArray(heatmap.heatmaps) && heatmap.heatmaps.length > 0
            ? heatmap.heatmaps
            : [{ title: "Hidden State Heatmap", matrix: heatmap.matrix }];
          fallbackHeatmaps.forEach((hm, idx) => {
            renderOneHeatmapIntoDoc(
              doc,
              container,
              hm && Array.isArray(hm.matrix) ? hm.matrix : [],
              String((hm && hm.title) || `Heatmap ${idx + 1}`),
              heatmapSyncGroup,
            );
          });
          return;
        }
        if (name === "render_logits") {
          renderTopLogitsTableIntoDoc(
            doc,
            container,
            Array.isArray(value) ? value : (heatmap.top_logits || []),
            heatmap,
            "Top 15 Logits (with cosine similarity)",
            "logits_source",
            "logits_error",
          );
          return;
        }
        if (name === "render_logits_top100") {
          renderTopLogitsTableIntoDoc(
            doc,
            container,
            Array.isArray(value) ? value : (heatmap.top_logits_top100 || []),
            heatmap,
            "Top 15 Logits (Penultimate Top-100 Intervention)",
            "top_logits_top100_source",
            "top_logits_top100_error",
          );
          return;
        }
        if (name === "render_neuron_logits_table") {
          renderNeuronLogitsTableIntoDoc(
            doc,
            container,
            Array.isArray(value) ? value : [],
            heatmap,
          );
        }
      } catch (taskErr) {
        const warn = doc.createElement("div");
        warn.className = "error";
        warn.textContent = `Task render failed: ${String(taskErr && taskErr.message ? taskErr.message : taskErr)}`;
        container.appendChild(warn);
      }
    });
    return;
  }

  // Backward-compatible fallback for older payloads without ui_tasks.
  const fallbackHeatmaps = Array.isArray(heatmap.heatmaps) && heatmap.heatmaps.length > 0
    ? heatmap.heatmaps
    : [{ key: "default", title: "Hidden State Heatmap", matrix: heatmap.matrix }];
  fallbackHeatmaps.forEach((hm, hmIdx) => {
    renderOneHeatmapIntoDoc(
      doc,
      container,
      hm && Array.isArray(hm.matrix) ? hm.matrix : [],
      String((hm && hm.title) || `Heatmap ${hmIdx + 1}`),
      heatmapSyncGroup,
    );
  });
  renderTopLogitsTableIntoDoc(doc, container, heatmap.top_logits || [], heatmap, "Top 15 Logits (with cosine similarity)", "logits_source", "logits_error");
  renderTopLogitsTableIntoDoc(doc, container, heatmap.top_logits_top100 || [], heatmap, "Top 15 Logits (Penultimate Top-100 Intervention)", "top_logits_top100_source", "top_logits_top100_error");
}

function renderOneHeatmapIntoDoc(doc, container, matrix, titleText, syncGroup) {
  const rows = Number(Array.isArray(matrix) ? matrix.length : 0);
  const cols = Number(rows > 0 && Array.isArray(matrix[0]) ? matrix[0].length : 0);
  if (!rows || !cols) return;

  const title = doc.createElement("h4");
  title.textContent = String(titleText || "Heatmap");
  container.appendChild(title);

  const cell = 10;
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
  canvas.style.cursor = "crosshair";
  wrap.appendChild(canvas);
  container.appendChild(wrap);

  if (syncGroup && Array.isArray(syncGroup.wraps)) {
    syncGroup.wraps.push(wrap);
    wrap.addEventListener("scroll", () => {
      if (syncGroup.syncing) return;
      syncGroup.syncing = true;
      try {
        const left = wrap.scrollLeft;
        const top = wrap.scrollTop;
        syncGroup.wraps.forEach((other) => {
          if (!other || other === wrap) return;
          other.scrollLeft = left;
          other.scrollTop = top;
        });
      } finally {
        syncGroup.syncing = false;
      }
    });
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  function drawWithThreshold(threshold) {
    for (let r = 0; r < rows; r += 1) {
      const row = matrix[r] || [];
      for (let c = 0; c < cols; c += 1) {
        const v = Number(row[c] || 0);
        ctx.fillStyle = heatmapColorFromValue(v, threshold);
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
    const sx = rect.width > 0 ? (canvas.width / rect.width) : 1;
    const sy = rect.height > 0 ? (canvas.height / rect.height) : 1;
    const x = Math.floor(((event.clientX - rect.left) * sx) / cell);
    const y = Math.floor(((event.clientY - rect.top) * sy) / cell);
    if (x < 0 || x >= cols || y < 0 || y >= rows) {
      hoverMeta.textContent = "Hover: X=-, Y=-";
      return;
    }
    const row = matrix[y] || [];
    const value = Number(row[x] || 0);
    hoverMeta.textContent = `Hover: X=${x} (neuron), Y=${y} (layer), value=${value.toFixed(6)}`;
  });
  canvas.addEventListener("mouseleave", () => {
    hoverMeta.textContent = "Hover: X=-, Y=-";
  });

  drawWithThreshold(heatmapThreshold);
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

function renderNeuronLogitsTableIntoDoc(doc, container, rows, payload) {
  const title = doc.createElement("h3");
  title.textContent = "Neuron -> Top 15 Logits Table";
  container.appendChild(title);

  const layer = Number(payload && payload.intervention_layer);
  const activation = Number(payload && payload.activation_value);
  const topK = Number(payload && payload.top_k);
  const hiddenDim = Number(payload && payload.hidden_dim);
  let threshold = Number(payload && payload.threshold);
  if (!Number.isFinite(threshold)) threshold = 15.0;
  const returnedRows = Number(payload && payload.returned_rows);
  const filteredRows = Number(payload && payload.filtered_out_rows);
  const meta = doc.createElement("div");
  meta.className = "muted";
  meta.textContent = `layer=${Number.isFinite(layer) ? layer : "-"}, activation=${Number.isFinite(activation) ? activation : "-"}, threshold=${threshold.toFixed(3)}, top_k=${Number.isFinite(topK) ? topK : "-"}, hidden_dim=${Number.isFinite(hiddenDim) ? hiddenDim : "-"}, returned=${Number.isFinite(returnedRows) ? returnedRows : "-"}, filtered=${Number.isFinite(filteredRows) ? filteredRows : "-"}`;
  container.appendChild(meta);

  const isBatched = Array.isArray(rows) && rows.length > 0 && rows[0] && typeof rows[0] === "object" && Array.isArray(rows[0].rows);
  if (!Array.isArray(rows) || rows.length === 0) {
    const empty = doc.createElement("div");
    empty.className = "muted";
    empty.textContent = "No neuron logits rows returned.";
    container.appendChild(empty);
    return;
  }

  function rowPassesThreshold(row) {
    const top = Array.isArray(row && row.top_logits) ? row.top_logits : [];
    if (!top.length) return false;
    const v = Number(top[0] && top[0].logit);
    return Number.isFinite(v) && v >= threshold;
  }

  function buildTableForRows(tableRows) {
    const safeRows = Array.isArray(tableRows) ? tableRows : [];
    const filteredRowsNow = safeRows.filter((r) => rowPassesThreshold(r));
    const effectiveTopK = Number.isFinite(topK) && topK > 0
      ? Math.floor(topK)
      : Math.max(...filteredRowsNow.map((r) => Array.isArray(r && r.top_logits) ? r.top_logits.length : 0), 0);

    const wrap = doc.createElement("div");
    wrap.className = "scroll";
    const table = doc.createElement("table");
    table.style.tableLayout = "fixed";
    table.style.width = "100%";
    const thead = doc.createElement("thead");
    const headerRow = doc.createElement("tr");
    const first = doc.createElement("th");
    first.textContent = "neuron_id";
    first.style.width = "8ch";
    headerRow.appendChild(first);
    for (let rank = 1; rank <= effectiveTopK; rank += 1) {
      const thText = doc.createElement("th");
      thText.textContent = `r${rank}_text`;
      thText.style.width = "20ch";
      headerRow.appendChild(thText);
      const thLogit = doc.createElement("th");
      thLogit.textContent = `r${rank}_logit`;
      thLogit.style.width = "12ch";
      headerRow.appendChild(thLogit);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = doc.createElement("tbody");
    if (filteredRowsNow.length === 0) {
      const tr = doc.createElement("tr");
      const td = doc.createElement("td");
      td.colSpan = Math.max(1, 1 + effectiveTopK * 2);
      td.className = "muted";
      td.textContent = `No rows pass threshold ${threshold.toFixed(3)} in this batch.`;
      tr.appendChild(td);
      tbody.appendChild(tr);
    } else {
      filteredRowsNow.forEach((row) => {
        const tr = doc.createElement("tr");
        const neuronCell = doc.createElement("td");
        neuronCell.textContent = String((row && row.neuron_id) ?? "");
        tr.appendChild(neuronCell);
        const top = Array.isArray(row && row.top_logits) ? row.top_logits : [];
        for (let idx = 0; idx < effectiveTopK; idx += 1) {
          const item = top[idx] || {};
          const tdText = doc.createElement("td");
          tdText.textContent = String(item.text ?? "");
          tdText.title = String(item.text ?? "");
          tdText.style.maxWidth = "20ch";
          tdText.style.whiteSpace = "nowrap";
          tdText.style.overflow = "hidden";
          tdText.style.textOverflow = "ellipsis";
          tr.appendChild(tdText);
          const tdLogit = doc.createElement("td");
          const lv = item.logit;
          tdLogit.textContent = typeof lv === "number" ? lv.toFixed(6) : "";
          tr.appendChild(tdLogit);
        }
        tbody.appendChild(tr);
      });
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  if (isBatched) {
    const batchHost = doc.createElement("div");
    container.appendChild(batchHost);
    rows.forEach((batch, idx) => {
      const batchRows = Array.isArray(batch && batch.rows) ? batch.rows : [];
      const startId = Number(batch && batch.start_neuron_id);
      const endId = Number(batch && batch.end_neuron_id);
      const head = doc.createElement("div");
      head.className = "muted";
      head.textContent = `Batch ${idx}: neuron ${Number.isFinite(startId) ? startId : "-"} - ${Number.isFinite(endId) ? endId : "-"}, rows=${batchRows.length}`;
      batchHost.appendChild(head);
      batchHost.appendChild(buildTableForRows(batchRows));
    });
    return;
  }

  container.appendChild(buildTableForRows(rows));
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
  renderSentenceNextWordInParams(result);
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
    const reason = String(heatmap.reason || "unknown");
    const tokenCount = Number(heatmap.token_count || 0);
    const tokenNote = Number.isFinite(tokenCount) && tokenCount > 0 ? ` (token_count=${tokenCount})` : "";
    msg.textContent = `Hidden-state heatmap failed: ${reason}${tokenNote}`;
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
    renderTopLogitsTable(
      (heatmap && heatmap.top_logits) || [],
      heatmap || {},
      "Top 15 Logits (with cosine similarity)",
      "logits_source",
      "logits_error",
    );
    renderTopLogitsTable(
      (heatmap && heatmap.top_logits_top100) || [],
      heatmap || {},
      "Top 15 Logits (Penultimate Top-100 Intervention)",
      "top_logits_top100_source",
      "top_logits_top100_error",
    );
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
    for (let r = 0; r < rows; r += 1) {
      const row = heatmap.matrix[r] || [];
      for (let c = 0; c < cols; c += 1) {
        const v = Number(row[c] || 0);
        ctx.fillStyle = heatmapColorFromValue(v, threshold);
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

function initChat() {
  chatMessages = [];
  renderChatTranscript();
  appendChatMeta("Chat is ready. Press Enter to send, Shift+Enter for newline.");
}

function appendChatMeta(text) {
  const node = document.createElement("div");
  node.className = "chat-bubble meta";
  node.textContent = text;
  chatTranscript.appendChild(node);
  chatTranscript.scrollTop = chatTranscript.scrollHeight;
}

function renderChatTranscript() {
  chatTranscript.innerHTML = "";
  chatMessages.forEach((msg) => {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${msg.role === "user" ? "user" : "assistant"}`;
    bubble.textContent = msg.content;
    chatTranscript.appendChild(bubble);
  });
  chatTranscript.scrollTop = chatTranscript.scrollHeight;
}

async function sendChatMessage() {
  const text = (chatInput.value || "").trim();
  if (!text) return;
  const temp = Number(chatTemperature.value);
  const maxTokens = Number(chatMaxTokens.value);
  const includeAssistantMarker = Boolean(chatIncludeAssistantMarker && chatIncludeAssistantMarker.checked);
  const neuronEnabled = Boolean(chatLayerNeuronEnabled && chatLayerNeuronEnabled.checked);
  const neuronLayer = Number(chatLayerNeuronLayer && chatLayerNeuronLayer.value);
  const neuronId = Number(chatLayerNeuronId && chatLayerNeuronId.value);
  const neuronValue = Number(chatLayerNeuronValue && chatLayerNeuronValue.value);
  const neuronToken = String((chatLayerNeuronToken && chatLayerNeuronToken.value) || "").trim();
  const ffnEnabled = Boolean(chatFfnNeuronEnabled && chatFfnNeuronEnabled.checked);
  const ffnLayer = Number(chatFfnNeuronLayer && chatFfnNeuronLayer.value);
  const ffnNeuronId = Number(chatFfnNeuronId && chatFfnNeuronId.value);
  const ffnNeuronValue = Number(chatFfnNeuronValue && chatFfnNeuronValue.value);
  const layerNeuronChange = neuronEnabled
    ? {
        enabled: true,
        layer: Number.isFinite(neuronLayer) ? Math.trunc(neuronLayer) : 0,
        neuron: Number.isFinite(neuronId) ? Math.trunc(neuronId) : 0,
        value: Number.isFinite(neuronValue) ? neuronValue : 0,
        token: neuronToken,
      }
    : { enabled: false };
  const ffnNeuronChange = ffnEnabled
    ? {
        enabled: true,
        layer: Number.isFinite(ffnLayer) ? Math.trunc(ffnLayer) : 0,
        neuron: Number.isFinite(ffnNeuronId) ? Math.trunc(ffnNeuronId) : 0,
        value: Number.isFinite(ffnNeuronValue) ? ffnNeuronValue : 0,
      }
    : { enabled: false };

  chatMessages.push({ role: "user", content: text });
  renderChatTranscript();
  chatInput.value = "";
  chatSendButton.disabled = true;
  setStatus("Chatting");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chatMessages,
        temperature: Number.isFinite(temp) ? temp : 0.7,
        max_new_tokens: Number.isFinite(maxTokens) ? maxTokens : 128,
        top_p: 0.9,
        include_assistant_marker: includeAssistantMarker,
        layer_neuron_change: layerNeuronChange,
        ffn_neuron_change: ffnNeuronChange,
      }),
    });
    const payload = await response.json();
    if (!response.ok || payload.status !== "ok") {
      const detail = payload.error || payload.stderr || `HTTP ${response.status}`;
      appendChatMeta(`Error: ${detail}`);
      return;
    }
    const assistantText = String(payload.assistant_message || "").trim();
    chatMessages.push({
      role: "assistant",
      content: assistantText || "[empty response]",
    });
    renderChatTranscript();
    const lnc = payload.layer_neuron_change || { enabled: false };
    if (lnc.enabled) {
      appendChatMeta(
        `Layer neuron change active: layer=${lnc.layer}, neuron=${lnc.neuron}, value=${Number(lnc.value).toFixed(4)}, token=${String(lnc.token || "")}, applied_count=${Number(lnc.applied_count || 0)}`,
      );
    }
    const fnc = payload.ffn_neuron_change || { enabled: false };
    if (fnc.enabled) {
      appendChatMeta(
        `FFN neuron change active: layer=${fnc.layer}, neuron=${fnc.neuron}, value=${Number(fnc.value).toFixed(4)}`,
      );
    }
    if (payload.prompt_mode) {
      appendChatMeta(`Prompt mode: ${String(payload.prompt_mode)}`);
    }
  } catch (error) {
    appendChatMeta(`Error: ${error.message}`);
  } finally {
    chatSendButton.disabled = false;
    setStatus("Ready");
    chatInput.focus();
  }
}

function updateChatNeuronControls() {
  const enabled = Boolean(chatLayerNeuronEnabled && chatLayerNeuronEnabled.checked);
  if (chatLayerNeuronLayer) chatLayerNeuronLayer.disabled = !enabled;
  if (chatLayerNeuronId) chatLayerNeuronId.disabled = !enabled;
  if (chatLayerNeuronValue) chatLayerNeuronValue.disabled = !enabled;
  if (chatLayerNeuronToken) chatLayerNeuronToken.disabled = !enabled;
  const ffnEnabled = Boolean(chatFfnNeuronEnabled && chatFfnNeuronEnabled.checked);
  if (chatFfnNeuronLayer) chatFfnNeuronLayer.disabled = !ffnEnabled;
  if (chatFfnNeuronId) chatFfnNeuronId.disabled = !ffnEnabled;
  if (chatFfnNeuronValue) chatFfnNeuronValue.disabled = !ffnEnabled;
}

function clearChat() {
  chatMessages = [];
  renderChatTranscript();
  appendChatMeta("Chat history cleared.");
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
openHistoryButton.addEventListener("click", openLatestFfnHistory);
showCsvButton.addEventListener("click", () => lastResult && renderCsv(lastResult.csv_preview));
histogramButton.addEventListener("click", addHistogram);
colorMapButton.addEventListener("click", addColorMap);
paramsForm.addEventListener("submit", (event) => {
  event.preventDefault();
});
chatSendButton.addEventListener("click", sendChatMessage);
chatClearButton.addEventListener("click", clearChat);
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChatMessage();
  }
});
if (chatLayerNeuronEnabled) {
  chatLayerNeuronEnabled.addEventListener("change", updateChatNeuronControls);
}
if (chatFfnNeuronEnabled) {
  chatFfnNeuronEnabled.addEventListener("change", updateChatNeuronControls);
}

loadActions().catch((error) => {
  setStatus("Error");
  resultSummary.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
});
initChat();
updateChatNeuronControls();
