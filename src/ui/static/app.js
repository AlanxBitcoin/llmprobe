let actions = [];
let selectedAction = null;
let lastResult = null;

const actionsList = document.getElementById("actionsList");
const paramsForm = document.getElementById("paramsForm");
const actionTitle = document.getElementById("actionTitle");
const actionDescription = document.getElementById("actionDescription");
const runButton = document.getElementById("runButton");
const commandPreview = document.getElementById("commandPreview");
const serverStatus = document.getElementById("serverStatus");
const resultSummary = document.getElementById("resultSummary");
const csvContainer = document.getElementById("csvContainer");
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
      input.value = field.default ?? "";
      if (field.min !== undefined) input.min = field.min;
      if (field.required) input.required = true;
      input.addEventListener("input", updateCommandPreview);
      wrap.appendChild(input);
    }
    paramsForm.appendChild(wrap);
  });
}

function collectParams() {
  const params = {};
  const data = new FormData(paramsForm);
  for (const [key, value] of data.entries()) {
    const field = (selectedAction.fields || []).find((item) => item.name === key);
    params[key] = field && field.type === "number" ? Number(value) : value;
  }
  return params;
}

function updateCommandPreview() {
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

async function runSelectedAction() {
  if (!selectedAction) return;
  const params = collectParams();
  setStatus("Running");
  runButton.disabled = true;
  resultSummary.textContent = "Running study. Large model calls can take a while.";
  csvContainer.innerHTML = "";
  artifactList.innerHTML = "";
  try {
    const response = await fetch("/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action_id: selectedAction.id, params }),
    });
    lastResult = await response.json();
    renderResult(lastResult);
  } catch (error) {
    lastResult = null;
    resultSummary.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
  } finally {
    runButton.disabled = false;
    setStatus("Ready");
  }
}

function renderResult(result) {
  const ok = result.status === "ok";
  resultSummary.innerHTML = `
    <div class="${ok ? "ok" : "error"}">Status: ${escapeHtml(result.status || "unknown")}</div>
    <div>Return code: ${escapeHtml(String(result.return_code ?? ""))}</div>
    <div>Artifacts: ${result.artifacts ? result.artifacts.length : 0}</div>
  `;
  if (result.stderr) {
    const pre = document.createElement("pre");
    pre.className = "command-preview";
    pre.textContent = result.stderr;
    resultSummary.appendChild(pre);
  }
  renderCsv(result.csv_preview);
  renderArtifacts(result.artifacts || []);
  const hasCsv = !!(result.csv_preview && result.csv_preview.rows && result.csv_preview.rows.length);
  showCsvButton.disabled = !hasCsv;
  histogramButton.disabled = !hasCsv;
  colorMapButton.disabled = !hasCsv;
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
