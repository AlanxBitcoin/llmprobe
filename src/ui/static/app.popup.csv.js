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
    const safeThreshold = Math.max(0.000001, Number(threshold) || 1);
    const intensity = Math.min(1, Math.abs(Number(value) || 0) / safeThreshold);
    const red = value > 0 ? Math.round(255 * intensity) : 0;
    const blue = value < 0 ? Math.round(255 * intensity) : 0;
    if (value > 0 || value < 0) return `rgb(${red},0,${blue})`;
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
