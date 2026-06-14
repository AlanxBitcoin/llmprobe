function heatmapColorFromValue(value, threshold) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "rgb(0,0,0)";
  if (Math.abs(n) < 1e-12) return "rgb(0,0,0)";
  const safeThreshold = Math.max(0.000001, Number(threshold) || 1);
  const intensity = Math.min(1, Math.abs(n) / safeThreshold);
  const channel = Math.max(0, Math.min(255, Math.round(255 * intensity)));
  if (n > 0) return `rgb(${channel},0,0)`;
  if (n < 0) return `rgb(0,0,${channel})`;
  return "rgb(0,0,0)";
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

function renderHeatmapIntoDoc(doc, container, heatmap) {
  container.innerHTML = "<h3>Hidden State</h3>";
  const heatmapSyncGroup = { wraps: [], syncing: false, heatmaps: [], hover: null };
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
    let renderedHeatmap = false;
    let renderedAnyTask = false;
    let heatmapTaskSeen = false;
    tasks.forEach((task) => {
      try {
        const name = String((task && task.name) || "");
        const valueKey = String((task && task.value_key) || "");
        const value = valueKey ? heatmap[valueKey] : undefined;
        if (name === "render_heatmap") {
          heatmapTaskSeen = true;
          renderedAnyTask = true;
          // Support:
          // 1) value_key -> 2D matrix
          // 2) value_key -> [{title,matrix}, ...]
          // 3) value_key -> {title,matrix}
          // 3) fallback to heatmap.heatmaps / heatmap.matrix
          if (Array.isArray(value) && value.length > 0 && Array.isArray(value[0]) && Array.isArray(value[0][0]) === false) {
            const ok = renderOneHeatmapIntoDoc(doc, container, value, "Hidden State Heatmap", heatmapSyncGroup);
            renderedHeatmap = renderedHeatmap || ok;
            return;
          }
          if (Array.isArray(value) && value.length > 0 && value[0] && typeof value[0] === "object" && Array.isArray(value[0].matrix)) {
            value.forEach((hm, idx) => {
              const ok = renderOneHeatmapIntoDoc(
                doc,
                container,
                hm.matrix,
                String(hm.title || `Heatmap ${idx + 1}`),
                heatmapSyncGroup,
              );
              renderedHeatmap = renderedHeatmap || ok;
            });
            return;
          }
          if (value && typeof value === "object" && !Array.isArray(value) && Array.isArray(value.matrix)) {
            const ok = renderOneHeatmapIntoDoc(
              doc,
              container,
              value.matrix,
              String(value.title || "Hidden State Heatmap"),
              heatmapSyncGroup,
            );
            renderedHeatmap = renderedHeatmap || ok;
            return;
          }
          const fallbackHeatmaps = Array.isArray(heatmap.heatmaps) && heatmap.heatmaps.length > 0
            ? heatmap.heatmaps
            : [{ title: "Hidden State Heatmap", matrix: heatmap.matrix }];
          fallbackHeatmaps.forEach((hm, idx) => {
            const ok = renderOneHeatmapIntoDoc(
              doc,
              container,
              hm && Array.isArray(hm.matrix) ? hm.matrix : [],
              String((hm && hm.title) || `Heatmap ${idx + 1}`),
              heatmapSyncGroup,
            );
            renderedHeatmap = renderedHeatmap || ok;
          });
          return;
        }
        if (name === "render_logits") {
          renderedAnyTask = true;
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
          renderedAnyTask = true;
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
          renderedAnyTask = true;
          renderNeuronLogitsTableIntoDoc(
            doc,
            container,
            Array.isArray(value) ? value : [],
            heatmap,
          );
          return;
        }
        if (name === "render_text_output") {
          renderedAnyTask = true;
          renderTextOutputIntoDoc(
            doc,
            container,
            String(value ?? ""),
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
    if (!renderedHeatmap) {
      const fallbackHeatmaps = Array.isArray(heatmap.heatmaps) && heatmap.heatmaps.length > 0
        ? heatmap.heatmaps
        : [{ key: "default", title: "Hidden State Heatmap", matrix: heatmap.matrix }];
      fallbackHeatmaps.forEach((hm, hmIdx) => {
        const ok = renderOneHeatmapIntoDoc(
          doc,
          container,
          hm && Array.isArray(hm.matrix) ? hm.matrix : [],
          String((hm && hm.title) || `Heatmap ${hmIdx + 1}`),
          heatmapSyncGroup,
        );
        renderedHeatmap = renderedHeatmap || ok;
      });
    }
    if (heatmapTaskSeen && !renderedHeatmap) {
      const warn = doc.createElement("div");
      warn.className = "error";
      warn.textContent = "Heatmap task exists but no valid matrix was rendered.";
      container.appendChild(warn);
    }
    if (!renderedAnyTask) {
      const note = doc.createElement("div");
      note.className = "muted";
      note.textContent = "No matching ui_tasks renderer found; used fallback rendering.";
      container.appendChild(note);
    }
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

function renderTextOutputIntoDoc(doc, container, text, heatmap) {
  const title = doc.createElement("h4");
  title.style.margin = "10px 0 6px 0";
  const limit = Number((heatmap && heatmap.generated_max_new_tokens) || 256);
  title.textContent = `Generated Text (max_new_tokens=${Number.isFinite(limit) ? limit : 256})`;
  container.appendChild(title);

  const err = heatmap && heatmap.generated_text_error ? String(heatmap.generated_text_error) : "";
  if (err) {
    const errNode = doc.createElement("div");
    errNode.className = "error";
    errNode.textContent = `Generation failed: ${err}`;
    container.appendChild(errNode);
    return;
  }

  const pre = doc.createElement("pre");
  pre.className = "scroll";
  pre.style.whiteSpace = "pre-wrap";
  pre.style.wordBreak = "break-word";
  pre.style.padding = "10px";
  pre.textContent = String(text || "");
  container.appendChild(pre);
}

function renderOneHeatmapIntoDoc(doc, container, matrix, titleText, syncGroup) {
  const rows = Number(Array.isArray(matrix) ? matrix.length : 0);
  const cols = Number(rows > 0 && Array.isArray(matrix[0]) ? matrix[0].length : 0);
  if (!rows || !cols) return false;

  const originalMatrix = new Array(rows);
  for (let r = 0; r < rows; r += 1) {
    const srcRow = Array.isArray(matrix[r]) ? matrix[r] : [];
    const row = new Array(cols);
    for (let c = 0; c < cols; c += 1) {
      row[c] = Number(srcRow[c] || 0);
    }
    originalMatrix[r] = row;
  }

  const normalizedMatrix = new Array(rows);
  for (let r = 0; r < rows; r += 1) {
    const row = originalMatrix[r];
    let sqSum = 0;
    for (let c = 0; c < cols; c += 1) sqSum += row[c] * row[c];
    const rms = Math.sqrt(sqSum / Math.max(1, cols));
    const coeff = Number.isFinite(rms) && rms > 1e-12 ? rms : 1.0;
    const out = new Array(cols);
    for (let c = 0; c < cols; c += 1) out[c] = row[c] / coeff;
    normalizedMatrix[r] = out;
  }

  let normEnabled = false;

  const header = doc.createElement("div");
  header.style.display = "flex";
  header.style.alignItems = "center";
  header.style.gap = "8px";
  const toggleBtn = doc.createElement("button");
  toggleBtn.type = "button";
  toggleBtn.textContent = "Hide";
  toggleBtn.title = "Hide / Show this heatmap";
  header.appendChild(toggleBtn);
  const title = doc.createElement("h4");
  title.style.margin = "0";
  title.textContent = String(titleText || "Heatmap");
  header.appendChild(title);
  container.appendChild(header);

  const body = doc.createElement("div");
  container.appendChild(body);

  const hoverMeta = doc.createElement("div");
  hoverMeta.className = "muted";
  hoverMeta.textContent = "Hover: X=-, Y=-";
  body.appendChild(hoverMeta);

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
  const normLabel = doc.createElement("span");
  normLabel.textContent = " Norm ";
  const normCheckbox = doc.createElement("input");
  normCheckbox.type = "checkbox";
  normCheckbox.checked = false;
  controls.appendChild(thresholdLabel);
  controls.appendChild(thresholdSlider);
  controls.appendChild(thresholdValue);
  controls.appendChild(normLabel);
  controls.appendChild(normCheckbox);
  body.appendChild(controls);

  const requestedCell = 10;
  const CANVAS_MAX_DIM = 32760;
  const safeByCols = Math.floor(CANVAS_MAX_DIM / Math.max(1, cols));
  const safeByRows = Math.floor(CANVAS_MAX_DIM / Math.max(1, rows));
  const cell = Math.max(1, Math.min(requestedCell, safeByCols, safeByRows));
  if (cell !== requestedCell) {
    const dimNote = doc.createElement("div");
    dimNote.className = "muted";
    dimNote.textContent = `Auto-scaled cell from ${requestedCell}px to ${cell}px to fit browser canvas limit.`;
    body.appendChild(dimNote);
  }

  const wrap = doc.createElement("div");
  wrap.className = "scroll";
  const stage = doc.createElement("div");
  stage.style.position = "relative";
  stage.style.width = `${cols * cell}px`;
  stage.style.height = `${rows * cell}px`;
  const canvas = doc.createElement("canvas");
  canvas.width = cols * cell;
  canvas.height = rows * cell;
  canvas.style.position = "absolute";
  canvas.style.left = "0";
  canvas.style.top = "0";
  canvas.style.zIndex = "1";
  const guideCanvas = doc.createElement("canvas");
  guideCanvas.width = cols * cell;
  guideCanvas.height = rows * cell;
  guideCanvas.style.position = "absolute";
  guideCanvas.style.left = "0";
  guideCanvas.style.top = "0";
  guideCanvas.style.zIndex = "2";
  guideCanvas.style.cursor = "crosshair";
  stage.appendChild(canvas);
  stage.appendChild(guideCanvas);
  wrap.appendChild(stage);
  body.appendChild(wrap);

  toggleBtn.addEventListener("click", () => {
    const hidden = body.style.display === "none";
    body.style.display = hidden ? "" : "none";
    toggleBtn.textContent = hidden ? "Hide" : "Show";
  });

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
  const guideCtx = guideCanvas.getContext("2d");
  if (!ctx || !guideCtx) return false;

  function getActiveMatrix() {
    return normEnabled ? normalizedMatrix : originalMatrix;
  }

  function drawBase(threshold) {
    const active = getActiveMatrix();
    for (let r = 0; r < rows; r += 1) {
      const row = active[r];
      for (let c = 0; c < cols; c += 1) {
        const v = Number(row[c] || 0);
        ctx.fillStyle = heatmapColorFromValue(v, threshold);
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
  }

  function drawHoverGuide(x, y) {
    guideCtx.clearRect(0, 0, guideCanvas.width, guideCanvas.height);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    if (x < 0 || x >= cols || y < 0 || y >= rows) return;
    const vx = x * cell + (cell / 2);
    const hy = y * cell + (cell / 2);
    guideCtx.save();
    guideCtx.strokeStyle = "rgba(255,255,255,0.85)";
    guideCtx.lineWidth = 1;
    guideCtx.setLineDash([5, 4]);
    guideCtx.beginPath();
    guideCtx.moveTo(vx, 0);
    guideCtx.lineTo(vx, canvas.height);
    guideCtx.stroke();
    guideCtx.beginPath();
    guideCtx.moveTo(0, hy);
    guideCtx.lineTo(canvas.width, hy);
    guideCtx.stroke();
    guideCtx.restore();
  }

  function setHoverText(x, y) {
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x >= cols || y < 0 || y >= rows) {
      hoverMeta.textContent = "Hover: X=-, Y=-";
      return;
    }
    const active = getActiveMatrix();
    const row = active[y] || [];
    const value = Number(row[x] || 0);
    hoverMeta.textContent = `Hover: X=${x} (neuron), Y=${y} (layer), value=${value.toFixed(6)}`;
  }

  function redrawBase() {
    drawBase(heatmapThreshold);
  }

  function redrawGuide() {
    if (syncGroup && syncGroup.hover) {
      drawHoverGuide(Number(syncGroup.hover.x), Number(syncGroup.hover.y));
      return;
    }
    drawHoverGuide(NaN, NaN);
  }

  if (syncGroup) {
    if (!Array.isArray(syncGroup.heatmaps)) syncGroup.heatmaps = [];
    if (typeof syncGroup.setHover !== "function") {
      syncGroup.setHover = (x, y) => {
        syncGroup.hover = { x: Number(x), y: Number(y) };
        (syncGroup.heatmaps || []).forEach((item) => {
          if (!item) return;
          item.setHoverText(Number(x), Number(y));
          item.redrawGuide();
        });
      };
    }
    if (typeof syncGroup.clearHover !== "function") {
      syncGroup.clearHover = () => {
        syncGroup.hover = null;
        (syncGroup.heatmaps || []).forEach((item) => {
          if (!item) return;
          item.setHoverText(NaN, NaN);
          item.redrawGuide();
        });
      };
    }
    syncGroup.heatmaps.push({ redrawBase, redrawGuide, setHoverText });
  }

  thresholdSlider.addEventListener("input", () => {
    heatmapThreshold = Number(thresholdSlider.value);
    thresholdValue.textContent = heatmapThreshold.toFixed(2);
    redrawBase();
    redrawGuide();
  });

  normCheckbox.addEventListener("change", () => {
    normEnabled = Boolean(normCheckbox.checked);
    redrawBase();
    redrawGuide();
  });

  guideCanvas.addEventListener("mousemove", (event) => {
    const rect = guideCanvas.getBoundingClientRect();
    const sx = rect.width > 0 ? (guideCanvas.width / rect.width) : 1;
    const sy = rect.height > 0 ? (guideCanvas.height / rect.height) : 1;
    const x = Math.floor(((event.clientX - rect.left) * sx) / cell);
    const y = Math.floor(((event.clientY - rect.top) * sy) / cell);
    if (syncGroup && typeof syncGroup.setHover === "function") {
      syncGroup.setHover(x, y);
      return;
    }
    setHoverText(x, y);
    drawHoverGuide(x, y);
  });

  guideCanvas.addEventListener("mouseleave", () => {
    if (syncGroup && typeof syncGroup.clearHover === "function") {
      syncGroup.clearHover();
      return;
    }
    setHoverText(NaN, NaN);
    drawHoverGuide(NaN, NaN);
  });

  redrawBase();
  redrawGuide();
  return true;
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

