(() => {
  let ctx = null;

  function init(deps) {
    ctx = deps || {};
  }

  function getSelectedAction() {
    return ctx && typeof ctx.getSelectedAction === "function" ? ctx.getSelectedAction() : null;
  }

  function getParamsForm() {
    return ctx && typeof ctx.getParamsForm === "function" ? ctx.getParamsForm() : null;
  }

  function getAttributeGroupsFullPayload() {
    return ctx && typeof ctx.getAttributeGroupsFullPayload === "function" ? ctx.getAttributeGroupsFullPayload() : null;
  }

  function setAttributeGroupsFullPayload(value) {
    if (ctx && typeof ctx.setAttributeGroupsFullPayload === "function") {
      ctx.setAttributeGroupsFullPayload(value);
    }
  }

  function safeJsonParse(raw, fallback) {
    if (ctx && typeof ctx.safeJsonParse === "function") return ctx.safeJsonParse(raw, fallback);
    try {
      return JSON.parse(String(raw || "").trim() || "{}");
    } catch (_err) {
      return fallback;
    }
  }

  function updateCommandPreview() {
    if (ctx && typeof ctx.updateCommandPreview === "function") ctx.updateCommandPreview();
  }

  function clearInlineParamResult() {
    if (ctx && typeof ctx.clearInlineParamResult === "function") ctx.clearInlineParamResult();
  }

  function showJsonValidationError(message) {
    if (ctx && typeof ctx.showJsonValidationError === "function") {
      ctx.showJsonValidationError(message);
    }
  }

  function normalizeAttributeGroupsPayload(payload) {
    let groups = [];
    if (Array.isArray(payload)) groups = payload;
    else if (payload && Array.isArray(payload.groups)) groups = payload.groups;
    else if (payload && typeof payload === "object") groups = [payload];
    const out = [];
    groups.forEach((g, idx) => {
      if (!g || typeof g !== "object") return;
      const group_name = String(g.group_name || g.name || `group_${idx + 1}`).trim();
      const tokens = Array.isArray(g.tokens) ? g.tokens : String(g.tokens || "").split(",").map((x) => x.trim()).filter(Boolean);
      if (!group_name) return;
      out.push({ group_name, tokens });
    });
    return { groups: out };
  }

  function getAttributeGroupNamesFromPayload(payload) {
    const groups = Array.isArray(payload && payload.groups) ? payload.groups : [];
    return groups.map((x) => String(x.group_name || "").trim()).filter(Boolean);
  }

  async function loadAttributeGroupsJsonIntoTextbox() {
    const selectedAction = getSelectedAction();
    const paramsForm = getParamsForm();
    if (!selectedAction || selectedAction.id !== "study_attribute_group_neurons") return;
    if (!paramsForm) return;
    const textArea = paramsForm.elements.namedItem("attribute_groups_json");
    const selectedInput = paramsForm.elements.namedItem("selected_attribute_group");
    if (!textArea) return;
    try {
      const resp = await fetch("/api/attribute-groups/json");
      const payload = await resp.json();
      if (!payload || payload.status !== "ok") return;
      setAttributeGroupsFullPayload(normalizeAttributeGroupsPayload(safeJsonParse(payload.json_text, {})));
      if (selectedInput && !String(selectedInput.value || "").trim()) {
        const names = getAttributeGroupNamesFromPayload(getAttributeGroupsFullPayload());
        if (names.length > 0) selectedInput.value = names[0];
      }
      renderSelectedAttributeGroupEntry();
      updateAttributeGroupsPicker();
      updateCommandPreview();
    } catch (_err) {
      // Keep current textarea value if file-load fails.
    }
  }

  function validateAttributeGroupEntryFromEditor() {
    const paramsForm = getParamsForm();
    if (!paramsForm) return "Attribute group entry JSON invalid: form not ready.";
    const textArea = paramsForm.elements.namedItem("attribute_groups_json");
    const selectedInput = paramsForm.elements.namedItem("selected_attribute_group");
    if (!textArea) return null;
    let entry = null;
    try {
      entry = JSON.parse(String(textArea.value || "").trim() || "{}");
    } catch (err) {
      const msg = String(err && err.message ? err.message : err);
      return `Attribute group entry JSON invalid: ${msg}`;
    }
    const normalized = normalizeAttributeGroupsPayload(entry);
    const one = (normalized.groups || [])[0] || null;
    if (!one) return "Attribute group entry JSON invalid: empty entry.";
    const selected = String((selectedInput && selectedInput.value) || "").trim();
    const groupName = String(selected || one.group_name || "").trim();
    if (!groupName) return "Attribute group entry JSON invalid: group_name is required.";
    const tokens = Array.isArray(one.tokens) ? one.tokens : [];
    if (!tokens.length) return "Attribute group entry JSON invalid: tokens must be non-empty.";
    return null;
  }

  function renderSelectedAttributeGroupEntry() {
    const paramsForm = getParamsForm();
    if (!paramsForm) return;
    const textArea = paramsForm.elements.namedItem("attribute_groups_json");
    const selectedInput = paramsForm.elements.namedItem("selected_attribute_group");
    if (!textArea || !selectedInput) return;
    const payload = normalizeAttributeGroupsPayload(getAttributeGroupsFullPayload() || {});
    const selected = String(selectedInput.value || "").trim();
    const hit = (payload.groups || []).find((x) => String(x.group_name || "") === selected);
    if (hit) {
      const tokensCsv = Array.isArray(hit.tokens)
        ? hit.tokens.map((x) => String(x ?? "").trim()).filter(Boolean).join(", ")
        : String(hit.tokens || "").trim();
      const editorValue = {
        group_name: String(hit.group_name || "").trim(),
        tokens: tokensCsv,
      };
      textArea.value = JSON.stringify(editorValue);
      return;
    }
    const draft = { group_name: selected || "new_group", tokens: "red, blue, green" };
    textArea.value = JSON.stringify(draft);
  }

  function buildAttributeGroupsPayloadJsonFromUi(persist = false) {
    const paramsForm = getParamsForm();
    if (!paramsForm) return JSON.stringify({ groups: [] }, null, 0);
    const textArea = paramsForm.elements.namedItem("attribute_groups_json");
    const selectedInput = paramsForm.elements.namedItem("selected_attribute_group");
    const selected = String((selectedInput && selectedInput.value) || "").trim();
    const entry = safeJsonParse(textArea ? textArea.value : "{}", {});
    const payload = normalizeAttributeGroupsPayload(getAttributeGroupsFullPayload() || {});
    const groups = Array.isArray(payload.groups) ? payload.groups.slice() : [];
    const normalized = normalizeAttributeGroupsPayload(entry);
    let one = (normalized.groups || [])[0] || {};
    one = { ...one, group_name: selected || String(one.group_name || "new_group") };
    const idx = groups.findIndex((x) => String(x.group_name || "") === String(one.group_name || ""));
    if (idx >= 0) groups[idx] = one;
    else groups.push(one);
    if (persist) {
      setAttributeGroupsFullPayload({ groups });
    }
    return JSON.stringify({ groups }, null, 0);
  }

  function updateAttributeGroupsPickerVisibility() {
    const paramsForm = getParamsForm();
    const selectedAction = getSelectedAction();
    if (!paramsForm) return;
    const picker = paramsForm.querySelector("select[name='selected_attribute_group_picker']");
    const selectedInput = paramsForm.elements.namedItem("selected_attribute_group");
    if (!picker || !selectedInput) return;
    const wrap = picker.closest(".field-inline");
    if (!wrap) return;
    wrap.style.display = selectedAction && selectedAction.id === "study_attribute_group_neurons" ? "" : "none";
  }

  function refreshAttributeGroupsPickerOptions(picker, jsonInput, selectedInput) {
    const names = getAttributeGroupNamesFromPayload(getAttributeGroupsFullPayload() || {});
    picker.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = names.length > 0 ? "Choose group_name" : "No valid group_name";
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

  function updateAttributeGroupsPicker() {
    const selectedAction = getSelectedAction();
    const paramsForm = getParamsForm();
    if (!selectedAction || selectedAction.id !== "study_attribute_group_neurons") return;
    if (!paramsForm) return;
    const selectedInput = paramsForm.elements.namedItem("selected_attribute_group");
    const jsonInput = paramsForm.elements.namedItem("attribute_groups_json");
    const selectedField = paramsForm.querySelector(".field[data-field-name='selected_attribute_group']");
    if (!selectedInput || !jsonInput || !selectedField) return;
    let picker = selectedField.querySelector("select[name='selected_attribute_group_picker']");
    let refreshBtn = selectedField.querySelector("button[name='selected_attribute_group_refresh']");
    let newBtn = selectedField.querySelector("button[name='selected_attribute_group_new']");
    let deleteBtn = selectedField.querySelector("button[name='selected_attribute_group_delete']");
    if (!picker) {
      const inline = document.createElement("div");
      inline.className = "field-inline";
      picker = document.createElement("select");
      picker.name = "selected_attribute_group_picker";
      refreshBtn = document.createElement("button");
      refreshBtn.type = "button";
      refreshBtn.name = "selected_attribute_group_refresh";
      refreshBtn.textContent = "Save Entry";
      refreshBtn.title = "Save current group entry into groups JSON";
      refreshBtn.style.marginLeft = "8px";
      newBtn = document.createElement("button");
      newBtn.type = "button";
      newBtn.name = "selected_attribute_group_new";
      newBtn.textContent = "New Entry";
      newBtn.title = "Create a new group draft in editor";
      newBtn.style.marginLeft = "8px";
      deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.name = "selected_attribute_group_delete";
      deleteBtn.textContent = "Delete";
      deleteBtn.title = "Delete selected group entry from groups JSON";
      deleteBtn.style.marginLeft = "8px";
      inline.appendChild(picker);
      inline.appendChild(refreshBtn);
      inline.appendChild(newBtn);
      inline.appendChild(deleteBtn);
      selectedField.appendChild(inline);
      picker.addEventListener("change", () => {
        selectedInput.value = String(picker.value || "");
        renderSelectedAttributeGroupEntry();
        updateCommandPreview();
      });
      const saveCurrentEntry = () => {
        const err = validateAttributeGroupEntryFromEditor();
        if (err) {
          showJsonValidationError(err);
          return;
        }
        clearInlineParamResult();
        buildAttributeGroupsPayloadJsonFromUi(true);
        refreshAttributeGroupsPickerOptions(picker, jsonInput, selectedInput);
        renderSelectedAttributeGroupEntry();
        updateCommandPreview();
      };
      refreshBtn.addEventListener("click", saveCurrentEntry);
      newBtn.addEventListener("click", () => {
        const names = getAttributeGroupNamesFromPayload(getAttributeGroupsFullPayload() || {});
        const base = "new_group";
        let i = 1;
        let candidate = `${base}_${i}`;
        while (names.includes(candidate)) {
          i += 1;
          candidate = `${base}_${i}`;
        }
        selectedInput.value = candidate;
        picker.value = "";
        renderSelectedAttributeGroupEntry();
        updateCommandPreview();
      });
      deleteBtn.addEventListener("click", () => {
        const selected = String((selectedInput.value || "").trim());
        if (!selected) return;
        const payload = normalizeAttributeGroupsPayload(getAttributeGroupsFullPayload() || {});
        const groups = Array.isArray(payload.groups) ? payload.groups.slice() : [];
        const filtered = groups.filter((x) => String(x.group_name || "") !== selected);
        setAttributeGroupsFullPayload({ groups: filtered });
        const names = getAttributeGroupNamesFromPayload(getAttributeGroupsFullPayload() || {});
        selectedInput.value = names.length > 0 ? names[0] : "";
        refreshAttributeGroupsPickerOptions(picker, jsonInput, selectedInput);
        renderSelectedAttributeGroupEntry();
        updateCommandPreview();
      });
    }
    refreshAttributeGroupsPickerOptions(picker, jsonInput, selectedInput);
    renderSelectedAttributeGroupEntry();
    updateAttributeGroupsPickerVisibility();
  }

  window.AppHomepageAttributeGroups = {
    init,
    loadAttributeGroupsJsonIntoTextbox,
    validateAttributeGroupEntryFromEditor,
    normalizeAttributeGroupsPayload,
    getAttributeGroupNamesFromPayload,
    renderSelectedAttributeGroupEntry,
    buildAttributeGroupsPayloadJsonFromUi,
    updateAttributeGroupsPickerVisibility,
    refreshAttributeGroupsPickerOptions,
    updateAttributeGroupsPicker,
  };
})();
