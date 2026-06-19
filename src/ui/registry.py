from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Registry defines available Study/Probe actions for UI buttons.
# - Each action binds command id + form schema with stable metadata.

from dataclasses import dataclass
from typing import Any

from .forms import get_form_schema


@dataclass(frozen=True)
class UIAction:
    id: str
    label: str
    description: str
    kind: str
    command: str
    form_schema: str
    default_output: str = "csv"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "kind": self.kind,
            "command": self.command,
            "form_schema": self.form_schema,
            "fields": get_form_schema(self.form_schema),
            "default_output": self.default_output,
        }


UI_ACTIONS: dict[str, UIAction] = {
    "study_single_word_hidden_state": UIAction(
        id="study_single_word_hidden_state",
        label="Single Word Hidden State",
        description="Run one forward pass and render the full 37x4096 hidden-state heatmap.",
        kind="study",
        command="run-single-word-hidden-state",
        form_schema="single_word_hidden_state_form",
    ),
    "study_single_word_hidden_state_batch_average": UIAction(
        id="study_single_word_hidden_state_batch_average",
        label="Single Word Hidden State Batch Average",
        description="Run comma-separated words, average hidden-state matrices element-wise, then show top15 logits.",
        kind="study",
        command="run-single-word-hidden-state-batch-average",
        form_schema="single_word_hidden_state_batch_average_form",
    ),
    "study_sentence_next_word": UIAction(
        id="study_sentence_next_word",
        label="Sentence Next Word",
        description="Input one sentence and show top15 next-token logits.",
        kind="study",
        command="run-sentence-next-word",
        form_schema="sentence_next_word_form",
    ),
    "study_token_diff": UIAction(
        id="study_token_diff",
        label="Token Diff",
        description="Input two single-token words, render A/B hidden-state heatmaps and A-B diff heatmap.",
        kind="study",
        command="run-token-diff",
        form_schema="token_diff_form",
    ),
    "study_one_on_one_attention": UIAction(
        id="study_one_on_one_attention",
        label="1 on 1 Attention",
        description="Input two single-token words, then render layer x head attention heatmaps for token_b->token_a and token_b->self (with BOS).",
        kind="study",
        command="run-one-on-one-attention",
        form_schema="one_on_one_attention_form",
    ),
    "study_qk_params": UIAction(
        id="study_qk_params",
        label="QK Params",
        description="Render per-layer Q-head vs K-head cosine heatmaps from attention projection parameters.",
        kind="study",
        command="run-qk-params",
        form_schema="qk_params_form",
    ),
    "study_single_word_top_100_neurons": UIAction(
        id="study_single_word_top_100_neurons",
        label="Single Word Top 100 Neurons",
        description="Heatmap + baseline top15 logits + penultimate-layer top100-neuron intervention top15 logits.",
        kind="study",
        command="run-single-word-top-100-neurons",
        form_schema="single_word_top_100_neurons_form",
    ),
    "study_layer_neuron_logits_table": UIAction(
        id="study_layer_neuron_logits_table",
        label="Layer Neuron Logits Table",
        description="For one layer, activate one neuron (value=10) at a time and collect top15 logits.",
        kind="study",
        command="run-layer-neuron-logits-table",
        form_schema="layer_neuron_logits_table_form",
    ),
    "study_layer_neurons": UIAction(
        id="study_layer_neurons",
        label="Layer Neurons",
        description="Load a JSON neuron list, apply all neuron overrides once, and return heatmap + top15 logits.",
        kind="study",
        command="run-layer-neurons",
        form_schema="layer_neurons_logits_table_form",
    ),
    "study_layer_ffn_neuron_logits_table": UIAction(
        id="study_layer_ffn_neuron_logits_table",
        label="Layer FFN Neuron Logits Table",
        description="For one layer, activate one post-SiLU FFN neuron at a time and collect top15 logits; reverse mode compares each neuron's W1 vector with this layer input hidden before ranking logits.",
        kind="study",
        command="run-layer-ffn-neuron-logits-table",
        form_schema="layer_ffn_neuron_logits_table_form",
    ),
    "study_single_word": UIAction(
        id="study_single_word",
        label="Single Word",
        description="Analyze one word and write the usual single-word outputs.",
        kind="study",
        command="run-single-word",
        form_schema="single_word_form",
    ),
    "study_color_words": UIAction(
        id="study_color_words",
        label="Color Words Study",
        description="Run the color-word experiment and collect CSV/chart outputs.",
        kind="study",
        command="run-color-words-experiment",
        form_schema="color_words_form",
    ),
    "study_single_batch": UIAction(
        id="study_single_batch",
        label="Single Batch",
        description="Run the configured word list one word at a time.",
        kind="study",
        command="run-single-batch",
        form_schema="single_batch_form",
    ),
    "study_multi_batch": UIAction(
        id="study_multi_batch",
        label="Multi Batch",
        description="Run the configured word list in fixed-size groups.",
        kind="study",
        command="run-multi-batch",
        form_schema="multi_batch_form",
    ),
    "study_linear_probe": UIAction(
        id="study_linear_probe",
        label="Linear Probe Study",
        description="Train and export the current linear probe outputs.",
        kind="study",
        command="run-probe",
        form_schema="linear_probe_form",
    ),
    "study_attribute_probe": UIAction(
        id="study_attribute_probe",
        label="Attribute Probe Study",
        description="Train attribute-family probes and export CSV summaries.",
        kind="study",
        command="run-attribute-probe",
        form_schema="attribute_probe_form",
    ),
    "study_attribute_group_neurons": UIAction(
        id="study_attribute_group_neurons",
        label="Attribute Group Neurons",
        description="Load one attribute group JSON, analyze cached token hidden states, and export selected neurons to CSV.",
        kind="study",
        command="run-attribute-group-neurons",
        form_schema="attribute_group_neurons_form",
    ),
}


def list_ui_actions() -> list[dict[str, Any]]:
    return [action.to_dict() for action in UI_ACTIONS.values()]


def get_ui_action(action_id: str) -> UIAction:
    try:
        return UI_ACTIONS[action_id]
    except KeyError as exc:
        raise ValueError(f"Unknown UI action: {action_id}") from exc


def build_command_args(action: UIAction, params: dict[str, Any]) -> list[str]:
    if action.command == "run-single-word":
        return [action.command, str(params.get("word") or "apple")]
    if action.command == "run-single-word-hidden-state":
        include_bos = params.get("include_bos")
        include_bos_flag = bool(include_bos) if include_bos is not None else True
        include_assistant = params.get("include_assistant")
        include_assistant_flag = bool(include_assistant) if include_assistant is not None else False
        return [
            action.command,
            str(params.get("word") or "apple"),
            "--include-bos",
            "true" if include_bos_flag else "false",
            "--include-assistant",
            "true" if include_assistant_flag else "false",
        ]
    if action.command == "run-single-word-hidden-state-batch-average":
        include_bos = params.get("include_bos")
        include_bos_flag = bool(include_bos) if include_bos is not None else True
        include_assistant = params.get("include_assistant")
        include_assistant_flag = bool(include_assistant) if include_assistant is not None else False
        return [
            action.command,
            str(params.get("words_csv") or "apple, banana, orange"),
            "--include-bos",
            "true" if include_bos_flag else "false",
            "--include-assistant",
            "true" if include_assistant_flag else "false",
        ]
    if action.command == "run-sentence-next-word":
        return [action.command, str(params.get("sentence") or "The apple is red.")]
    if action.command == "run-token-diff":
        return [
            action.command,
            str(params.get("token_a") or "apple"),
            str(params.get("token_b") or "banana"),
        ]
    if action.command == "run-one-on-one-attention":
        include_assistant = params.get("include_assistant")
        include_assistant_flag = bool(include_assistant) if include_assistant is not None else False
        return [
            action.command,
            str(params.get("token_a") or "apple"),
            str(params.get("token_b") or "banana"),
            "--include-assistant",
            "true" if include_assistant_flag else "false",
        ]
    if action.command == "run-qk-params":
        view_by_layer = params.get("view_by_layer")
        view_by_layer_flag = bool(view_by_layer) if view_by_layer is not None else True
        view_by_head = params.get("view_by_head")
        view_by_head_flag = bool(view_by_head) if view_by_head is not None else False
        selected_layer = params.get("selected_layer")
        return [
            action.command,
            "--view-by-layer",
            "true" if view_by_layer_flag else "false",
            "--view-by-head",
            "true" if view_by_head_flag else "false",
            "--selected-layer",
            str(1 if selected_layer is None else selected_layer),
        ]
    if action.command == "run-single-word-top-100-neurons":
        top_k_neurons = params.get("top_k_neurons")
        intervention_layer = params.get("intervention_layer")
        return [
            action.command,
            str(params.get("word") or "apple"),
            "--top-k-neurons",
            str(100 if top_k_neurons is None else top_k_neurons),
            "--intervention-layer",
            str(30 if intervention_layer is None else intervention_layer),
        ]
    if action.command == "run-layer-neuron-logits-table":
        intervention_layer = params.get("intervention_layer")
        activation_value = params.get("activation_value")
        use_prefix_context = params.get("use_prefix_context")
        use_prefix_context_flag = bool(use_prefix_context) if use_prefix_context is not None else False
        prefix_text = str(params.get("prefix_text") or "The apple is red.")
        return_batch_size = params.get("return_batch_size")
        return [
            action.command,
            "--intervention-layer",
            str(30 if intervention_layer is None else intervention_layer),
            "--activation-value",
            str(10.0 if activation_value is None else activation_value),
            "--use-prefix-context",
            "true" if use_prefix_context_flag else "false",
            "--prefix-text",
            prefix_text,
            "--return-batch-size",
            str(1000 if return_batch_size is None else return_batch_size),
        ]
    if action.command == "run-layer-neurons":
        use_prefix_context = params.get("use_prefix_context")
        use_prefix_context_flag = bool(use_prefix_context) if use_prefix_context is not None else False
        prefix_text = str(params.get("prefix_text") or "The apple is red.")
        use_random1000_baseline_no_prefix = params.get("use_random1000_baseline_no_prefix")
        use_random1000_baseline_no_prefix_flag = (
            bool(use_random1000_baseline_no_prefix)
            if use_random1000_baseline_no_prefix is not None
            else True
        )
        layer_neuron_list_json = str(params.get("layer_neuron_list_json") or "")
        selected_list_name = str(params.get("selected_list_name") or "")
        return [
            action.command,
            "--use-prefix-context",
            "true" if use_prefix_context_flag else "false",
            "--prefix-text",
            prefix_text,
            "--use-random1000-baseline-no-prefix",
            "true" if use_random1000_baseline_no_prefix_flag else "false",
            "--selected-list-name",
            selected_list_name,
            "--layer-neuron-list-json",
            layer_neuron_list_json,
        ]
    if action.command == "run-layer-ffn-neuron-logits-table":
        intervention_layer = params.get("intervention_layer")
        activation_value = params.get("activation_value")
        include_bos = params.get("include_bos")
        include_bos_flag = bool(include_bos) if include_bos is not None else True
        reverse = params.get("reverse")
        reverse_flag = bool(reverse) if reverse is not None else False
        use_prefix_context = params.get("use_prefix_context")
        use_prefix_context_flag = bool(use_prefix_context) if use_prefix_context is not None else False
        prefix_text = str(params.get("prefix_text") or "The apple is red.")
        return_batch_size = params.get("return_batch_size")
        return [
            action.command,
            "--intervention-layer",
            str(30 if intervention_layer is None else intervention_layer),
            "--activation-value",
            str(10.0 if activation_value is None else activation_value),
            "--include-bos",
            "true" if include_bos_flag else "false",
            "--reverse",
            "true" if reverse_flag else "false",
            "--use-prefix-context",
            "true" if use_prefix_context_flag else "false",
            "--prefix-text",
            prefix_text,
            "--return-batch-size",
            str(1000 if return_batch_size is None else return_batch_size),
        ]
    if action.command == "run-color-words-experiment":
        return [
            action.command,
            "--word-file",
            str(params.get("word_file") or "data/color_words.txt"),
            "--run-name",
            str(params.get("run_name") or "color_words"),
        ]
    if action.command == "run-multi-batch":
        return [action.command, "--batch-size", str(params.get("batch_size") or 2)]
    if action.command == "run-probe":
        return [action.command, "--label-file", str(params.get("label_file") or "data/word_labels.csv")]
    if action.command == "run-attribute-probe":
        return [
            action.command,
            "--attribute-file",
            str(params.get("attribute_file") or "data/word_attributes.csv"),
        ]
    if action.command == "run-attribute-group-neurons":
        return [
            action.command,
            "--selected-attribute-group",
            str(params.get("selected_attribute_group") or ""),
            "--attribute-groups-json",
            str(params.get("attribute_groups_json") or ""),
            "--filter-json",
            str(params.get("filter_json") or ""),
        ]
    return [action.command]
