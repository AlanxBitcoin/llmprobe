let actions = [];
let selectedAction = null;
const actionParamsCache = new Map();
const LEGACY_STUDY_IDS = new Set([
  "study_single_word",
  "study_color_words",
  "study_single_batch",
  "study_multi_batch",
  "study_linear_probe",
  "study_attribute_probe",
]);

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

const resultToolbar = document.querySelector(".result-toolbar");
const csvContainer = document.getElementById("csvContainer");
const hiddenStateContainer = document.getElementById("hiddenStateContainer");
const chartsSection = document.querySelector(".charts-section");
if (resultToolbar) resultToolbar.style.display = "none";
if (csvContainer) csvContainer.style.display = "none";
if (hiddenStateContainer) hiddenStateContainer.style.display = "none";
if (chartsSection) chartsSection.style.display = "none";

let layerNeuronsFullPayload = null;
let attributeGroupsFullPayload = null;

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
  "-", "--", "_", "/", "\\", "|", "~",
  "(", ")", "[", "]", "{", "}",
  "<", ">", "=", "+", "*", "&", "%", "$", "#", "@",
  "<0x0A>", "<0x0D>", "\\n", "\\r", "\\r\\n",
  "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
  "assistant", "user", "system"
]);

function normalizeRawTokenForSymbolFilter(value) {
  let t = String(value ?? "");
  // Remove BPE boundary marks only; keep punctuation/newline semantics.
  t = t.replace(/[_\u2581\u0120]/g, "");
  return t.toLowerCase().trim();
}

function normalizeWordTokenForGrammarFilter(value) {
  let t = String(value ?? "").toLowerCase();
  t = t.replace(/[_\u2581\u0120\s\r\n\t]/g, "");
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
    if (LEGACY_STUDY_IDS.has(String(action.id || ""))) {
      button.classList.add("action-button-legacy");
    }
    button.type = "button";
    button.dataset.actionId = action.id;
    button.innerHTML = `${escapeHtml(action.label)}<span>${escapeHtml(action.kind)}</span>`;
    button.addEventListener("click", () => selectAction(action.id));
    actionsList.appendChild(button);
  });
}

function selectAction(actionId) {
  if (selectedAction && selectedAction.id) {
    actionParamsCache.set(selectedAction.id, collectParams());
  }
  selectedAction = actions.find((action) => action.id === actionId);
  if (!selectedAction) return;
  document.querySelectorAll(".action-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.actionId === actionId);
  });
  actionTitle.textContent = selectedAction.label;
  actionDescription.textContent = selectedAction.description || "";
  const cachedParams = actionParamsCache.get(selectedAction.id) || null;
  renderForm(selectedAction.fields || [], cachedParams);
  if (selectedAction && selectedAction.id === "study_layer_neurons") {
    loadLayerNeuronsJsonIntoTextbox();
  }
  if (selectedAction && selectedAction.id === "study_attribute_group_neurons") {
    loadAttributeGroupsJsonIntoTextbox();
  }
  runButton.disabled = false;
  updateHistoryButtonVisibility();
  if (selectedAction && selectedAction.id === "study_layer_ffn_neuron_logits_table") {
    loadFfnHistoryList();
  }
  updateCommandPreview();
  clearInlineParamResult();
}

async function loadLayerNeuronsJsonIntoTextbox() {
  if (!selectedAction || selectedAction.id !== "study_layer_neurons") return;
  const textArea = paramsForm.elements.namedItem("layer_neuron_list_json");
  const selectedInput = paramsForm.elements.namedItem("selected_list_name");
  if (!textArea) return;
  try {
    const resp = await fetch("/api/layer-neurons/list-json");
    const payload = await resp.json();
    if (!payload || payload.status !== "ok") return;
    layerNeuronsFullPayload = normalizeLayerNeuronsPayload(safeJsonParse(payload.json_text, {}));
    if (selectedInput && !String(selectedInput.value || "").trim()) {
      const names = getLayerNeuronListNamesFromPayload(layerNeuronsFullPayload);
      if (names.length > 0) selectedInput.value = names[0];
    }
    renderSelectedLayerNeuronEntry();
    updateLayerNeuronsListPicker();
    updateCommandPreview();
  } catch (_err) {
    // Keep current textarea value if file-load fails.
  }
}

async function loadAttributeGroupsJsonIntoTextbox() {
  if (!window.AppHomepageAttributeGroups) return;
  return window.AppHomepageAttributeGroups.loadAttributeGroupsJsonIntoTextbox();
}

function renderForm(fields, initialParams = null) {
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
      if ((field.type || "text") === "textarea") {
        const area = document.createElement("textarea");
        area.name = field.name;
        const hasCached = initialParams && Object.prototype.hasOwnProperty.call(initialParams, field.name);
        area.value = hasCached ? String(initialParams[field.name] ?? "") : (field.default ?? "");
        area.rows = Number(field.rows || 10);
        if (field.required) area.required = true;
        area.addEventListener("input", updateCommandPreview);
        wrap.appendChild(area);
      } else {
        input.name = field.name;
        input.type = field.type || "text";
        const hasCached = initialParams && Object.prototype.hasOwnProperty.call(initialParams, field.name);
        if (input.type === "checkbox") {
          input.checked = hasCached ? Boolean(initialParams[field.name]) : (field.default !== false);
        } else {
          input.value = hasCached ? String(initialParams[field.name] ?? "") : (field.default ?? "");
        }
        if (field.min !== undefined) input.min = field.min;
        if (field.max !== undefined) input.max = field.max;
        if (field.step !== undefined) input.step = field.step;
        if (field.required) input.required = true;
        input.addEventListener(input.type === "checkbox" ? "change" : "input", updateCommandPreview);
        wrap.appendChild(input);
      }
    }
    paramsForm.appendChild(wrap);
  });
  updateBosAssistantVisibility();
  updatePrefixContextVisibility();
  updateBatchNameDropdown();
  updateLayerNeuronsListPicker();
  updateAttributeGroupsPicker();
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
  if (selectedAction && selectedAction.id === "study_layer_neurons") {
    params.layer_neuron_list_json = JSON.stringify(normalizeLayerNeuronsPayload(layerNeuronsFullPayload || {}), null, 0);
  }
  if (selectedAction && selectedAction.id === "study_attribute_group_neurons") {
    params.attribute_groups_json = JSON.stringify(normalizeAttributeGroupsPayload(attributeGroupsFullPayload || {}), null, 0);
  }
  return params;
}


function updateCommandPreview() {
  updateBosAssistantVisibility();
  updatePrefixContextVisibility();
  updateBatchNameDropdownVisibility();
  updateLayerNeuronsListPickerVisibility();
  updateAttributeGroupsPickerVisibility();
  updateHistoryButtonVisibility();
  if (!selectedAction) {
    commandPreview.textContent = "";
    return;
  }
  const params = collectParams();
  commandPreview.textContent = JSON.stringify({ action_id: selectedAction.id, params });
}

function updatePrefixContextVisibility() {
  const usePrefixInput = paramsForm.elements.namedItem("use_prefix_context");
  const prefixTextInput = paramsForm.elements.namedItem("prefix_text");
  if (!usePrefixInput || !prefixTextInput) return;
  const prefixWrap = prefixTextInput.closest(".field");
  if (!prefixWrap) return;
  const prefixEnabled = Boolean(usePrefixInput.checked);
  prefixWrap.style.display = prefixEnabled ? "" : "none";

  // Layer-neurons only: 1000-token baseline option is meaningful only when no prefix is used.
  const baselineInput = paramsForm.elements.namedItem("use_random1000_baseline_no_prefix");
  if (baselineInput) {
    const baselineWrap = baselineInput.closest(".field");
    if (baselineWrap) {
      baselineWrap.style.display = prefixEnabled ? "none" : "";
    }
  }
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

function refreshBatchPickerOptions(picker, items, batchInput) {
  picker.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = items.length > 0 ? "Choose saved batch" : "No saved batch";
  picker.appendChild(placeholder);
  items.forEach((item) => {
    const name = String(item.name || "").trim();
    if (!name) return;
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.dataset.wordsCsv = String(item.words_csv || "");
    picker.appendChild(option);
  });
  const current = String((batchInput && batchInput.value) || "").trim();
  if (current && items.some((x) => String(x.name || "").trim() === current)) {
    picker.value = current;
  } else {
    picker.value = "";
  }
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
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.textContent = "Save Entry";
  saveBtn.title = "Save current batch_name + words into batch cache";
  saveBtn.style.marginLeft = "8px";
  const newBtn = document.createElement("button");
  newBtn.type = "button";
  newBtn.textContent = "New Entry";
  newBtn.title = "Create a new batch draft";
  newBtn.style.marginLeft = "8px";
  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.textContent = "Delete";
  deleteBtn.title = "Delete selected batch from batch cache";
  deleteBtn.style.marginLeft = "8px";
  inline.appendChild(picker);
  inline.appendChild(saveBtn);
  inline.appendChild(newBtn);
  inline.appendChild(deleteBtn);
  batchField.appendChild(inline);

  let batchItems = [];

  try {
    const resp = await fetch("/api/batches");
    const payload = await resp.json();
    batchItems = Array.isArray(payload.batches) ? payload.batches : [];
    refreshBatchPickerOptions(picker, batchItems, batchInput);
  } catch (_error) {
    refreshBatchPickerOptions(picker, [], batchInput);
  }

  picker.addEventListener("change", () => {
    const opt = picker.selectedOptions[0];
    const name = opt ? String(opt.value || "") : "";
    const wordsCsv = opt ? String(opt.dataset.wordsCsv || "") : "";
    if (name) batchInput.value = name;
    if (wordsCsv) wordsInput.value = wordsCsv;
    updateCommandPreview();
  });

  newBtn.addEventListener("click", () => {
    const names = batchItems.map((x) => String(x.name || "").trim()).filter(Boolean);
    let i = 1;
    let candidate = `new_batch_${i}`;
    while (names.includes(candidate)) {
      i += 1;
      candidate = `new_batch_${i}`;
    }
    batchInput.value = candidate;
    wordsInput.value = "";
    picker.value = "";
    updateCommandPreview();
  });

  saveBtn.addEventListener("click", async () => {
    const name = String(batchInput.value || "").trim();
    const words = String(wordsInput.value || "").trim();
    if (!name || !words) return;
    try {
      const resp = await fetch("/api/batches/upsert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_name: name, words_csv: words }),
      });
      const payload = await resp.json();
      batchItems = Array.isArray(payload && payload.batches) ? payload.batches : batchItems;
      refreshBatchPickerOptions(picker, batchItems, batchInput);
      picker.value = name;
      updateCommandPreview();
    } catch (_err) {
      // keep local fields unchanged on save failure
    }
  });

  deleteBtn.addEventListener("click", async () => {
    const selectedName = String((picker.value || batchInput.value || "")).trim();
    if (!selectedName) return;
    try {
      const resp = await fetch("/api/batches/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_name: selectedName }),
      });
      const payload = await resp.json();
      batchItems = Array.isArray(payload && payload.batches) ? payload.batches : [];
      refreshBatchPickerOptions(picker, batchItems, batchInput);
      const names = batchItems.map((x) => String(x.name || "").trim()).filter(Boolean);
      if (names.length > 0) {
        const next = names[0];
        const nextItem = batchItems.find((x) => String(x.name || "").trim() === next) || null;
        batchInput.value = next;
        wordsInput.value = nextItem ? String(nextItem.words_csv || "") : "";
      } else {
        batchInput.value = "";
        wordsInput.value = "";
      }
      picker.value = "";
      updateCommandPreview();
    } catch (_err) {
      // keep local fields unchanged on delete failure
    }
  });
  updateBatchNameDropdownVisibility();
}

function extractLayerNeuronListNames(raw) {
  try {
    const payload = JSON.parse(String(raw || "").trim() || "{}");
    let lists = [];
    if (Array.isArray(payload)) {
      lists = payload;
    } else if (payload && Array.isArray(payload.lists)) {
      lists = payload.lists;
    } else if (payload && typeof payload === "object") {
      lists = [payload];
    }
    const out = [];
    lists.forEach((x, idx) => {
      if (!x || typeof x !== "object") return;
      const neurons = Array.isArray(x.neurons) ? x.neurons : [];
      // Lightweight strict validation for picker refresh:
      // each neuron row must be object-like or 2-item pair-like.
      const malformed = neurons.some((row) => {
        if (row && typeof row === "object" && !Array.isArray(row)) return false;
        if (Array.isArray(row) && row.length === 2) return false;
        return true;
      });
      if (malformed) return;
      const explicit = typeof x.list_name === "string" ? String(x.list_name).trim() : "";
      if (explicit) {
        out.push(explicit);
        return;
      }
      const hasLayer = Number.isFinite(Number(x.nLayer));
      const hasNeurons = Array.isArray(x.neurons);
      if (hasLayer && hasNeurons) {
        out.push(`list_${idx + 1}`);
      }
    });
    return out;
  } catch (_err) {
    return [];
  }
}

function safeJsonParse(raw, fallback) {
  try {
    return JSON.parse(String(raw || "").trim() || "{}");
  } catch (_err) {
    return fallback;
  }
}

function strictJsonParse(raw, label) {
  try {
    return JSON.parse(String(raw || "").trim() || "{}");
  } catch (err) {
    const msg = String(err && err.message ? err.message : err);
    throw new Error(`${label} JSON invalid: ${msg}`);
  }
}

function validateLayerNeuronEntryFromEditor() {
  const textArea = paramsForm.elements.namedItem("layer_neuron_list_json");
  const selectedInput = paramsForm.elements.namedItem("selected_list_name");
  if (!textArea) return null;
  let entry = null;
  try {
    entry = strictJsonParse(textArea.value, "Layer neuron entry");
  } catch (err) {
    return String(err && err.message ? err.message : err);
  }
  const normalized = normalizeLayerNeuronsPayload(entry);
  const one = (normalized.lists || [])[0] || null;
  if (!one) return "Layer neuron entry JSON invalid: empty entry.";
  const selected = String((selectedInput && selectedInput.value) || "").trim();
  const listName = String(selected || one.list_name || "").trim();
  if (!listName) return "Layer neuron entry JSON invalid: list_name is required.";
  const nLayer = Number(one.nLayer);
  if (!Number.isFinite(nLayer)) return "Layer neuron entry JSON invalid: nLayer must be numeric.";
  const neurons = Array.isArray(one.neurons) ? one.neurons : [];
  if (!neurons.length) return "Layer neuron entry JSON invalid: neurons must be a non-empty array.";
  for (const row of neurons) {
    if (Array.isArray(row) && row.length === 2) {
      const nid = Number(row[0]);
      const val = Number(row[1]);
      if (!Number.isFinite(nid) || !Number.isFinite(val)) {
        return "Layer neuron entry JSON invalid: each neuron row must be [neuron_id, value] numbers.";
      }
      continue;
    }
    return "Layer neuron entry JSON invalid: each neuron row must be [neuron_id, value].";
  }
  return null;
}

function validateAttributeGroupEntryFromEditor() {
  if (!window.AppHomepageAttributeGroups) return "Attribute group entry JSON invalid: module missing.";
  return window.AppHomepageAttributeGroups.validateAttributeGroupEntryFromEditor();
}

function showJsonValidationError(message) {
  const msg = String(message || "JSON validation failed.");
  setStatus("Input Error");
  inlineParamResult.innerHTML = `<span class="error">${escapeHtml(msg)}</span>`;
  inlineParamResult.style.display = "";
}

function normalizeLayerNeuronsPayload(payload) {
  let lists = [];
  if (Array.isArray(payload)) lists = payload;
  else if (payload && Array.isArray(payload.lists)) lists = payload.lists;
  else if (payload && typeof payload === "object") lists = [payload];
  const out = [];
  lists.forEach((x, idx) => {
    if (!x || typeof x !== "object") return;
    const name = String(x.list_name || `list_${idx + 1}`).trim();
    const nLayer = Number.isFinite(Number(x.nLayer)) ? Number(x.nLayer) : 30;
    const neurons = Array.isArray(x.neurons) ? x.neurons : [];
    out.push({ list_name: name, nLayer, neurons });
  });
  return { lists: out };
}

function normalizeAttributeGroupsPayload(payload) {
  if (!window.AppHomepageAttributeGroups) return { groups: [] };
  return window.AppHomepageAttributeGroups.normalizeAttributeGroupsPayload(payload);
}

function getLayerNeuronListNamesFromPayload(payload) {
  const lists = Array.isArray(payload && payload.lists) ? payload.lists : [];
  return lists.map((x) => String(x.list_name || "").trim()).filter(Boolean);
}

function getAttributeGroupNamesFromPayload(payload) {
  if (!window.AppHomepageAttributeGroups) return [];
  return window.AppHomepageAttributeGroups.getAttributeGroupNamesFromPayload(payload);
}

function renderSelectedLayerNeuronEntry() {
  const textArea = paramsForm.elements.namedItem("layer_neuron_list_json");
  const selectedInput = paramsForm.elements.namedItem("selected_list_name");
  if (!textArea || !selectedInput) return;
  const payload = normalizeLayerNeuronsPayload(layerNeuronsFullPayload || {});
  const selected = String(selectedInput.value || "").trim();
  const hit = (payload.lists || []).find((x) => String(x.list_name || "") === selected);
  if (hit) {
    textArea.value = JSON.stringify(hit);
    return;
  }
  const draft = { list_name: selected || "new_list", nLayer: 30, neurons: [[45, 20.0]] };
  textArea.value = JSON.stringify(draft);
}

function renderSelectedAttributeGroupEntry() {
  if (!window.AppHomepageAttributeGroups) return;
  return window.AppHomepageAttributeGroups.renderSelectedAttributeGroupEntry();
}

function buildLayerNeuronsPayloadJsonFromUi() {
  const textArea = paramsForm.elements.namedItem("layer_neuron_list_json");
  const selectedInput = paramsForm.elements.namedItem("selected_list_name");
  const selected = String((selectedInput && selectedInput.value) || "").trim();
  const entry = safeJsonParse(textArea ? textArea.value : "{}", {});
  const payload = normalizeLayerNeuronsPayload(layerNeuronsFullPayload || {});
  const lists = Array.isArray(payload.lists) ? payload.lists.slice() : [];
  const normalized = normalizeLayerNeuronsPayload(entry);
  let one = (normalized.lists || [])[0] || {};
  one = { ...one, list_name: selected || String(one.list_name || "new_list") };
  const idx = lists.findIndex((x) => String(x.list_name || "") === String(one.list_name || ""));
  if (idx >= 0) lists[idx] = one;
  else lists.push(one);
  layerNeuronsFullPayload = { lists };
  return JSON.stringify(layerNeuronsFullPayload, null, 0);
}

function buildAttributeGroupsPayloadJsonFromUi(persist = false) {
  if (!window.AppHomepageAttributeGroups) return JSON.stringify({ groups: [] }, null, 0);
  return window.AppHomepageAttributeGroups.buildAttributeGroupsPayloadJsonFromUi(persist);
}

function updateLayerNeuronsListPickerVisibility() {
  const picker = paramsForm.querySelector("select[name='selected_list_name_picker']");
  const selectedInput = paramsForm.elements.namedItem("selected_list_name");
  if (!picker || !selectedInput) return;
  const wrap = picker.closest(".field-inline");
  if (!wrap) return;
  wrap.style.display = selectedAction && selectedAction.id === "study_layer_neurons" ? "" : "none";
}

function extractAttributeGroupNames(raw) {
  try {
    const payload = JSON.parse(String(raw || "").trim() || "{}");
    let groups = [];
    if (Array.isArray(payload)) {
      groups = payload;
    } else if (payload && Array.isArray(payload.groups)) {
      groups = payload.groups;
    } else if (payload && typeof payload === "object") {
      groups = [payload];
    }
    const out = [];
    groups.forEach((g, idx) => {
      if (!g || typeof g !== "object") return;
      const name = String(g.group_name || g.name || "").trim();
      if (name) {
        out.push(name);
        return;
      }
      const tokens = Array.isArray(g.tokens) ? g.tokens : [];
      if (tokens.length > 0) {
        out.push(`group_${idx + 1}`);
      }
    });
    return out;
  } catch (_err) {
    return [];
  }
}

function updateAttributeGroupsPickerVisibility() {
  if (!window.AppHomepageAttributeGroups) return;
  return window.AppHomepageAttributeGroups.updateAttributeGroupsPickerVisibility();
}

function refreshAttributeGroupsPickerOptions(picker, jsonInput, selectedInput) {
  if (!window.AppHomepageAttributeGroups) return;
  return window.AppHomepageAttributeGroups.refreshAttributeGroupsPickerOptions(picker, jsonInput, selectedInput);
}

function updateAttributeGroupsPicker() {
  if (!window.AppHomepageAttributeGroups) return;
  return window.AppHomepageAttributeGroups.updateAttributeGroupsPicker();
}

function refreshLayerNeuronsListPickerOptions(picker, jsonInput, selectedInput) {
  const names = getLayerNeuronListNamesFromPayload(layerNeuronsFullPayload || {});
  picker.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = names.length > 0 ? "Choose list_name" : "No valid list_name";
  picker.appendChild(placeholder);
  names.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    picker.appendChild(opt);
  });
  const current = String((selectedInput.value || "").trim());
  if (current && names.includes(current)) {
    picker.value = current;
  } else if (!current && names.length === 1) {
    picker.value = names[0];
    selectedInput.value = names[0];
  } else {
    picker.value = "";
  }
}

function updateLayerNeuronsListPicker() {
  if (!selectedAction || selectedAction.id !== "study_layer_neurons") return;
  const selectedInput = paramsForm.elements.namedItem("selected_list_name");
  const jsonInput = paramsForm.elements.namedItem("layer_neuron_list_json");
  const selectedField = paramsForm.querySelector(".field[data-field-name='selected_list_name']");
  if (!selectedInput || !jsonInput || !selectedField) return;
  let picker = selectedField.querySelector("select[name='selected_list_name_picker']");
  let refreshBtn = selectedField.querySelector("button[name='selected_list_name_refresh']");
  let newBtn = selectedField.querySelector("button[name='selected_list_name_new']");
  let deleteBtn = selectedField.querySelector("button[name='selected_list_name_delete']");
  if (!picker) {
    const inline = document.createElement("div");
    inline.className = "field-inline";
    picker = document.createElement("select");
    picker.name = "selected_list_name_picker";
    refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.name = "selected_list_name_refresh";
    refreshBtn.textContent = "Save Entry";
    refreshBtn.title = "Save current list entry into lists JSON";
    refreshBtn.style.marginLeft = "8px";
    newBtn = document.createElement("button");
    newBtn.type = "button";
    newBtn.name = "selected_list_name_new";
    newBtn.textContent = "New Entry";
    newBtn.title = "Create a new list draft in editor";
    newBtn.style.marginLeft = "8px";
    deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.name = "selected_list_name_delete";
    deleteBtn.textContent = "Delete";
    deleteBtn.title = "Delete selected list entry from lists JSON";
    deleteBtn.style.marginLeft = "8px";
    inline.appendChild(picker);
    inline.appendChild(refreshBtn);
    inline.appendChild(newBtn);
    inline.appendChild(deleteBtn);
    selectedField.appendChild(inline);
    picker.addEventListener("change", () => {
      selectedInput.value = String(picker.value || "");
      renderSelectedLayerNeuronEntry();
      updateCommandPreview();
    });
    const refreshFromJson = () => {
      const err = validateLayerNeuronEntryFromEditor();
      if (err) {
        showJsonValidationError(err);
        return;
      }
      clearInlineParamResult();
      buildLayerNeuronsPayloadJsonFromUi();
      refreshLayerNeuronsListPickerOptions(picker, jsonInput, selectedInput);
      renderSelectedLayerNeuronEntry();
      updateCommandPreview();
    };
    refreshBtn.addEventListener("click", refreshFromJson);
    newBtn.addEventListener("click", () => {
      const names = getLayerNeuronListNamesFromPayload(layerNeuronsFullPayload || {});
      const base = "new_list";
      let i = 1;
      let candidate = `${base}_${i}`;
      while (names.includes(candidate)) {
        i += 1;
        candidate = `${base}_${i}`;
      }
      selectedInput.value = candidate;
      picker.value = "";
      renderSelectedLayerNeuronEntry();
      updateCommandPreview();
    });
    deleteBtn.addEventListener("click", () => {
      const selected = String((selectedInput.value || "").trim());
      if (!selected) return;
      const payload = normalizeLayerNeuronsPayload(layerNeuronsFullPayload || {});
      const lists = Array.isArray(payload.lists) ? payload.lists.slice() : [];
      const filtered = lists.filter((x) => String(x.list_name || "") !== selected);
      layerNeuronsFullPayload = { lists: filtered };
      const names = getLayerNeuronListNamesFromPayload(layerNeuronsFullPayload || {});
      selectedInput.value = names.length > 0 ? names[0] : "";
      refreshLayerNeuronsListPickerOptions(picker, jsonInput, selectedInput);
      renderSelectedLayerNeuronEntry();
      updateCommandPreview();
    });
  }
  refreshLayerNeuronsListPickerOptions(picker, jsonInput, selectedInput);
  renderSelectedLayerNeuronEntry();
  updateLayerNeuronsListPickerVisibility();
}

async function runSelectedAction() {
  if (!selectedAction) return;
  if (selectedAction.id === "study_attribute_group_neurons") {
    const err = validateAttributeGroupEntryFromEditor();
    if (err) {
      showJsonValidationError(err);
      return;
    }
    // Auto-save current editor entry before execute.
    buildAttributeGroupsPayloadJsonFromUi(true);
  }
  if (selectedAction.id === "study_layer_neurons") {
    const err = validateLayerNeuronEntryFromEditor();
    if (err) {
      showJsonValidationError(err);
      return;
    }
    // Auto-save current editor entry before execute.
    buildLayerNeuronsPayloadJsonFromUi();
  }
  clearInlineParamResult();
  const params = collectParams();
  setStatus("Opening Popup");
  runButton.disabled = true;
  try {
    openPopupWithRequest({
      kind: "execute",
      action_id: selectedAction.id,
      params,
    });
  } catch (error) {
    console.error(error);
  } finally {
    runButton.disabled = false;
    setStatus("Ready");
  }
}

function clearInlineParamResult() {
  inlineParamResult.innerHTML = "";
  inlineParamResult.style.display = "none";
}

function openPopupWithRequest(request) {
  const payload = JSON.parse(JSON.stringify(request || {}));
  const key = `popup_req_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  try {
    localStorage.setItem(key, JSON.stringify(payload));
  } catch (_err) {
    // If storage fails, popup page will show an error.
  }
  const url = `/popup?req_key=${encodeURIComponent(key)}`;
  return window.open(url, "_blank");
}


async function openLatestFfnHistory() {
  setStatus("Opening Popup");
  try {
    const selectedName = historySelect ? String(historySelect.value || "").trim() : "";
    openPopupWithRequest({
      kind: "history",
      selected_name: selectedName,
    });
  } catch (error) {
    console.error(error);
  } finally {
    setStatus("Ready");
  }
}

function setStatus(value) {
  serverStatus.textContent = value;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function initAttributeGroupsModule() {
  if (!window.AppHomepageAttributeGroups || typeof window.AppHomepageAttributeGroups.init !== "function") return;
  window.AppHomepageAttributeGroups.init({
    getSelectedAction: () => selectedAction,
    getParamsForm: () => paramsForm,
    getAttributeGroupsFullPayload: () => attributeGroupsFullPayload,
    setAttributeGroupsFullPayload: (value) => {
      attributeGroupsFullPayload = value;
    },
    safeJsonParse,
    updateCommandPreview,
    clearInlineParamResult,
    showJsonValidationError,
  });
}

initAttributeGroupsModule();
runButton.addEventListener("click", runSelectedAction);
openHistoryButton.addEventListener("click", openLatestFfnHistory);
paramsForm.addEventListener("submit", (event) => {
  event.preventDefault();
});

loadActions().catch((error) => {
  setStatus("Error");
  console.error(error);
});
