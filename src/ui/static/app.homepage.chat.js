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

let chatMessages = [];
const CHAT_MAX_HISTORY_MESSAGES = 12;

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

function buildChatRequestMessages(allMessages) {
  const list = Array.isArray(allMessages) ? allMessages : [];
  if (list.length <= CHAT_MAX_HISTORY_MESSAGES) return list;
  // Keep only the most recent turns for faster prompt construction/inference.
  return list.slice(-CHAT_MAX_HISTORY_MESSAGES);
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
    const outboundMessages = buildChatRequestMessages(chatMessages);
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: outboundMessages,
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
    if (chatMessages.length > outboundMessages.length) {
      appendChatMeta(`Speed mode: sent last ${outboundMessages.length} messages (trimmed context).`);
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

initChat();
updateChatNeuronControls();
