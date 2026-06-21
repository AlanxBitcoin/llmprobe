// CSV/Table module boundary:
// - Owns all table-style rendering (CSV preview + task-based tables).
// - Owns table task dispatch entry (renderCsvTasksIntoDoc).
// - Should not own heatmap/canvas rendering (that belongs to app.popup.heatmap.js).

function renderCsvIntoDoc(doc, container, csvPreview) {
  container.innerHTML = "<h3>CSV Preview</h3>";
  if (!csvPreview || !csvPreview.headers || csvPreview.headers.length === 0) {
    container.append("No CSV preview available yet.");
    return;
  }
  const headers = Array.isArray(csvPreview.headers) ? csvPreview.headers : [];
  const rows = Array.isArray(csvPreview.rows) ? csvPreview.rows : [];
  const isLikelyNumericColorColumn = (header, sampleRows) => {
    const key = String(header || "");
    if (/id|rank|layer|count/i.test(key)) return false;
    const values = (sampleRows || []).slice(0, 20).map((r) => Number(r && r[key]));
    const finite = values.filter((v) => Number.isFinite(v));
    if (!finite.length) return false;
    const hasPos = finite.some((v) => v > 0);
    const hasNeg = finite.some((v) => v < 0);
    return hasPos || hasNeg;
  };
  const colorHeaderFlags = headers.map((h) => {
    const key = String(h || "");
    if (/state|activation|logit|score|value|hidden/i.test(key)) return true;
    return isLikelyNumericColorColumn(key, rows);
  });
  const hasColorColumn = colorHeaderFlags.some(Boolean);
  const colorFromValue = (value, threshold) => {
    if (typeof heatmapColorFromValue === "function") {
      return heatmapColorFromValue(value, threshold);
    }
    const n = Number(value);
    if (!Number.isFinite(n)) return "rgb(0,0,0)";
    if (Math.abs(n) < 1e-12) return "rgb(0,0,0)";
    const safeThreshold = Math.max(0.000001, Number(threshold) || 1);
    const intensity = Math.min(1, Math.abs(n) / safeThreshold);
    const channel = Math.max(0, Math.min(255, Math.round(255 * intensity)));
    if (n > 0) return `rgb(${channel},0,0)`;
    if (n < 0) return `rgb(0,0,${channel})`;
    return "rgb(0,0,0)";
  };
  const setStateCellColor = (cell, rawValue, threshold) => {
    const n = Number(rawValue);
    if (!Number.isFinite(n)) {
      cell.style.backgroundColor = "";
      cell.style.color = "";
      return;
    }
    cell.style.backgroundColor = colorFromValue(n, threshold);
    cell.style.color = "#8a949e";
  };

  let localThreshold = Number(csvStateThreshold);
  if (!Number.isFinite(localThreshold) || localThreshold <= 0) localThreshold = 0.5;
  csvStateThreshold = localThreshold;

  if (hasColorColumn) {
    const controls = doc.createElement("div");
    controls.className = "muted";
    controls.style.display = "flex";
    controls.style.alignItems = "center";
    controls.style.gap = "8px";
    controls.style.marginBottom = "8px";
    const label = doc.createElement("span");
    label.textContent = "State Color Threshold";
    const slider = doc.createElement("input");
    slider.type = "range";
    slider.min = "0.05";
    slider.max = "5";
    slider.step = "0.05";
    slider.value = String(localThreshold);
    slider.style.width = "240px";
    const valueText = doc.createElement("span");
    valueText.textContent = localThreshold.toFixed(2);
    controls.appendChild(label);
    controls.appendChild(slider);
    controls.appendChild(valueText);
    container.appendChild(controls);

    slider.addEventListener("input", () => {
      const next = Number(slider.value);
      if (!Number.isFinite(next) || next <= 0) return;
      localThreshold = next;
      csvStateThreshold = next;
      valueText.textContent = next.toFixed(2);
      container.querySelectorAll("td[data-state-value]").forEach((cell) => {
        setStateCellColor(cell, cell.getAttribute("data-state-value"), localThreshold);
      });
    });
  }

  const wrap = doc.createElement("div");
  wrap.className = "scroll";
  wrap.style.maxHeight = "1560px";
  const table = doc.createElement("table");
  table.style.width = "max-content";
  table.style.tableLayout = "auto";
  table.style.fontSize = "12px";
  const thead = doc.createElement("thead");
  const hr = doc.createElement("tr");
  headers.forEach((h) => {
    const th = doc.createElement("th");
    th.textContent = h;
    th.style.padding = "3px 5px";
    th.style.whiteSpace = "nowrap";
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = doc.createElement("tbody");
  const formatCsvValue = (header, rawValue) => {
    if (rawValue === null || rawValue === undefined) return "";
    const text = String(rawValue);
    if (text.trim() === "") return "";
    const num = Number(text);
    if (!Number.isFinite(num)) return text;
    if (/id|count|rank|layer/i.test(String(header || ""))) {
      return String(Math.trunc(num));
    }
    return num.toFixed(2);
  };
  rows.forEach((row) => {
    const tr = doc.createElement("tr");
    headers.forEach((h, idx) => {
      const td = doc.createElement("td");
      const formattedValue = formatCsvValue(h, row[h]);
      td.textContent = formattedValue;
      td.style.padding = "2px 5px";
      td.style.whiteSpace = "nowrap";
      if (formattedValue !== "" && Number.isFinite(Number(formattedValue))) {
        td.style.color = "#8a949e";
      }
      if (colorHeaderFlags[idx]) {
        td.setAttribute("data-state-value", String(row[h] ?? ""));
        setStateCellColor(td, row[h], localThreshold);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}

function renderOutputTokenLogitsTableIntoDoc(doc, container, rows, titleText) {
  const title = doc.createElement("h3");
  title.textContent = String(titleText || "Per-Input-Token Top 15 Logits");
  container.appendChild(title);
  if (!Array.isArray(rows) || rows.length === 0) {
    const empty = doc.createElement("div");
    empty.className = "muted";
    empty.textContent = "No per-token logits rows returned.";
    container.appendChild(empty);
    return;
  }
  const wrap = doc.createElement("div");
  wrap.className = "scroll";
  const table = doc.createElement("table");
  table.style.tableLayout = "auto";
  table.style.width = "max-content";

  const thead = doc.createElement("thead");
  const hr = doc.createElement("tr");
  ["token_step", "token_id", "token", "text"].forEach((h) => {
    const th = doc.createElement("th");
    th.textContent = h;
    if (h === "token" || h === "text") {
      th.style.minWidth = "20ch";
      th.style.width = "20ch";
    } else {
      th.style.width = "10ch";
    }
    hr.appendChild(th);
  });
  for (let i = 1; i <= 15; i += 1) {
    const thText = doc.createElement("th");
    thText.textContent = `r${i}_text`;
    thText.style.minWidth = "20ch";
    thText.style.width = "20ch";
    hr.appendChild(thText);
    const thLogit = doc.createElement("th");
    thLogit.textContent = `r${i}_logit`;
    thLogit.style.width = "12ch";
    hr.appendChild(thLogit);
  }
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = doc.createElement("tbody");
  rows.forEach((row) => {
    const tr = doc.createElement("tr");
    const basic = [
      Number(row && row.step),
      Number(row && row.token_id),
      String((row && row.token) || ""),
      String((row && row.text) || ""),
    ];
    basic.forEach((v, idx) => {
      const td = doc.createElement("td");
      if (idx <= 1 && Number.isFinite(v)) td.textContent = String(Math.trunc(v));
      else td.textContent = String(v || "");
      if (idx === 2 || idx === 3) {
        td.style.minWidth = "20ch";
        td.style.maxWidth = "20ch";
        td.style.whiteSpace = "nowrap";
        td.style.overflow = "hidden";
        td.style.textOverflow = "ellipsis";
      } else {
        td.style.whiteSpace = "nowrap";
      }
      tr.appendChild(td);
    });
    const top = Array.isArray(row && row.top_logits) ? row.top_logits : [];
    for (let i = 0; i < 15; i += 1) {
      const item = top[i] || {};
      const tdText = doc.createElement("td");
      tdText.textContent = String(item.text || "");
      tdText.style.minWidth = "20ch";
      tdText.style.maxWidth = "20ch";
      tdText.style.whiteSpace = "nowrap";
      tdText.style.overflow = "hidden";
      tdText.style.textOverflow = "ellipsis";
      tr.appendChild(tdText);
      const tdLogit = doc.createElement("td");
      const lv = Number(item.logit);
      tdLogit.textContent = Number.isFinite(lv) ? lv.toFixed(6) : "";
      tdLogit.style.whiteSpace = "nowrap";
      tr.appendChild(tdLogit);
    }
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}

function renderCsvTasksIntoDoc(doc, container, heatmap) {
  if (!container || !heatmap || typeof heatmap !== "object") return;
  const tasks = Array.isArray(heatmap.ui_tasks) ? heatmap.ui_tasks : [];
  tasks.forEach((task) => {
    const name = String((task && task.name) || "");
    const valueKey = String((task && task.value_key) || "");
    const value = valueKey ? heatmap[valueKey] : undefined;
    if (name === "render_input_token_logits_table") {
      renderOutputTokenLogitsTableIntoDoc(
        doc,
        container,
        Array.isArray(value) ? value : [],
        "Per-Input-Token Top 15 Logits",
      );
    }
  });
}
