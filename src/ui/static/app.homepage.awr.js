(() => {
  let ctx = null;

  function init(deps) {
    ctx = deps || {};
  }

  function getParamsForm() {
    return ctx && typeof ctx.getParamsForm === "function" ? ctx.getParamsForm() : null;
  }

  // Keep these helpers with AWR code for easier maintenance.
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
    t = t.replace(/[_\u2581\u0120]/g, "");
    return t.toLowerCase().trim();
  }

  function normalizeWordTokenForGrammarFilter(value) {
    let t = String(value ?? "").toLowerCase();
    t = t.replace(/[_\u2581\u0120\s\r\n\t]/g, "");
    t = t.replace(/[^a-z']/g, "");
    return t;
  }

  function isGrammarTokenLike(item) {
    if (!item) return false;
    const rawCandidates = [
      normalizeRawTokenForSymbolFilter(item.text),
      normalizeRawTokenForSymbolFilter(item.token),
    ];
    if (rawCandidates.some((c) => c && GRAMMAR_SYMBOL_SET.has(c))) return true;
    const wordCandidates = [
      normalizeWordTokenForGrammarFilter(item.text),
      normalizeWordTokenForGrammarFilter(item.token),
    ];
    return wordCandidates.some((c) => c && GRAMMAR_WORD_SET.has(c));
  }

  function updateAwrLayerJumpVisibility() {
    const paramsForm = getParamsForm();
    if (!paramsForm) return;
    const enableInput =
      paramsForm.elements.namedItem("enable_layer_shortcut") ||
      paramsForm.elements.namedItem("enable_layer_jump");
    const startInput = paramsForm.elements.namedItem("shortcut_start_layer");
    const targetInput = paramsForm.elements.namedItem("shortcut_target_layer");
    if (!enableInput || !startInput || !targetInput) return;
    const startWrap = startInput.closest(".field");
    const targetWrap = targetInput.closest(".field");
    if (!startWrap || !targetWrap) return;
    const enabled = Boolean(enableInput.checked);
    startWrap.style.display = enabled ? "" : "none";
    targetWrap.style.display = enabled ? "" : "none";
  }

  function updateAwrIgnoreTokenVisibility() {
    const paramsForm = getParamsForm();
    if (!paramsForm) return;
    const enableInput = paramsForm.elements.namedItem("enable_ignore_replacement_token");
    const tokenInput = paramsForm.elements.namedItem("ignore_replacement_token");
    if (!enableInput || !tokenInput) return;
    const tokenWrap = tokenInput.closest(".field");
    if (!tokenWrap) return;
    tokenWrap.style.display = Boolean(enableInput.checked) ? "" : "none";
  }

  window.AppHomepageAwr = {
    init,
    updateAwrLayerJumpVisibility,
    updateAwrIgnoreTokenVisibility,
    isGrammarTokenLike,
  };
})();
