from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Central schema list for UI parameter forms.
# - Form defaults and constraints should match callable action args.

from typing import Any


FORM_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "single_word_form": [
        {"name": "word", "label": "Word", "type": "text", "default": "apple", "required": True},
    ],
    "single_word_hidden_state_form": [
        {"name": "word", "label": "Word", "type": "text", "default": "apple", "required": True},
        {"name": "include_bos", "label": "Include BOS", "type": "checkbox", "default": True},
        {"name": "include_assistant", "label": "Include Assistant", "type": "checkbox", "default": False},
    ],
    "single_word_hidden_state_batch_average_form": [
        {
            "name": "batch_name",
            "label": "Batch Name",
            "type": "text",
            "default": "",
            "required": False,
        },
        {
            "name": "words_csv",
            "label": "Words (comma-separated)",
            "type": "text",
            "default": "apple, banana, orange",
            "required": True,
        },
        {"name": "include_bos", "label": "Include BOS", "type": "checkbox", "default": True},
        {"name": "include_assistant", "label": "Include Assistant", "type": "checkbox", "default": False},
    ],
    "sentence_next_word_form": [
        {"name": "sentence", "label": "Sentence", "type": "text", "default": "The apple is red.", "required": True},
    ],
    "token_diff_form": [
        {"name": "token_a", "label": "Token A", "type": "text", "default": "apple", "required": True},
        {"name": "token_b", "label": "Token B", "type": "text", "default": "banana", "required": True},
    ],
    "one_on_one_attention_form": [
        {"name": "token_a", "label": "Word A (previous)", "type": "text", "default": "apple", "required": True},
        {"name": "token_b", "label": "Word B (current)", "type": "text", "default": "banana", "required": True},
        {"name": "include_assistant", "label": "Include Assistant", "type": "checkbox", "default": False},
    ],
    "qk_params_form": [
        {"name": "view_by_layer", "label": "按层看 (Layer x Dim)", "type": "checkbox", "default": True},
        {"name": "view_by_head", "label": "按头看 (Layer x Head)", "type": "checkbox", "default": False},
        {"name": "selected_layer", "label": "层号 (1-based)", "type": "number", "default": 1, "min": 1, "step": 1},
    ],
    "single_word_top_100_neurons_form": [
        {"name": "word", "label": "Word", "type": "text", "default": "apple", "required": True},
        {"name": "top_k_neurons", "label": "Top K Neurons", "type": "number", "default": 100, "min": 1},
        {"name": "intervention_layer", "label": "Intervention Layer", "type": "number", "default": 30, "min": 0, "step": 1},
    ],
    "layer_neuron_logits_table_form": [
        {"name": "intervention_layer", "label": "Layer", "type": "number", "default": 30, "min": 1, "step": 1},
        {"name": "activation_value", "label": "Activation Value", "type": "number", "default": 10.0},
        {"name": "use_prefix_context", "label": "Use Prefix Context", "type": "checkbox", "default": False},
        {"name": "prefix_text", "label": "Prefix Sentence", "type": "text", "default": "The apple is red.", "required": False},
        {"name": "return_batch_size", "label": "Return Batch Size", "type": "number", "default": 1000, "min": 1},
    ],
    "layer_neurons_logits_table_form": [
        {"name": "use_prefix_context", "label": "Use Prefix Context", "type": "checkbox", "default": False},
        {"name": "prefix_text", "label": "Prefix Sentence", "type": "text", "default": "The apple is red.", "required": False},
        {
            "name": "use_random1000_baseline_no_prefix",
            "label": "Use 1000-Token Baseline (No Prefix)",
            "type": "checkbox",
            "default": True,
        },
        {
            "name": "selected_list_name",
            "label": "Selected List Name",
            "type": "text",
            "default": "",
            "required": False,
        },
        {
            "name": "layer_neuron_list_json",
            "label": "Layer Neuron List JSON",
            "type": "textarea",
            "default": "{\"lists\":[{\"list_name\":\"example_a\",\"nLayer\":30,\"neurons\":[[45,20.0],[1024,-5.0]]},{\"list_name\":\"example_b\",\"nLayer\":31,\"neurons\":[[300,8.0]]}]}",
            "required": True,
            "rows": 12,
        },
    ],
    "layer_ffn_neuron_logits_table_form": [
        {"name": "intervention_layer", "label": "Layer", "type": "number", "default": 30, "min": 0, "step": 1},
        {"name": "activation_value", "label": "Activation Value", "type": "number", "default": 10.0},
        {"name": "include_bos", "label": "Include BOS", "type": "checkbox", "default": True},
        {"name": "reverse", "label": "Reverse (W1 x layer-input hidden)", "type": "checkbox", "default": False},
        {"name": "use_prefix_context", "label": "Use Prefix Context", "type": "checkbox", "default": False},
        {"name": "prefix_text", "label": "Prefix Sentence", "type": "text", "default": "The apple is red.", "required": False},
        {"name": "return_batch_size", "label": "Return Batch Size", "type": "number", "default": 1000, "min": 1},
    ],
    "color_words_form": [
        {"name": "word_file", "label": "Word file", "type": "text", "default": "data/color_words.txt"},
        {"name": "run_name", "label": "Run name", "type": "text", "default": "color_words"},
    ],
    "single_batch_form": [
        {"name": "note", "label": "Uses configured word file", "type": "display", "default": "configs/custom.yaml"},
    ],
    "multi_batch_form": [
        {"name": "batch_size", "label": "Batch size", "type": "number", "default": 2, "min": 2},
    ],
    "linear_probe_form": [
        {"name": "label_file", "label": "Label CSV", "type": "text", "default": "data/word_labels.csv"},
    ],
    "attribute_probe_form": [
        {"name": "attribute_file", "label": "Attribute CSV", "type": "text", "default": "data/word_attributes.csv"},
    ],
    "attribute_group_neurons_form": [
        {
            "name": "selected_attribute_group",
            "label": "Selected Attribute Group",
            "type": "text",
            "default": "",
            "required": False,
        },
        {
            "name": "attribute_groups_json",
            "label": "Attribute Groups JSON",
            "type": "textarea",
            "default": "{\"groups\":[{\"group_name\":\"color_basic\",\"tokens\":\"red, blue, green, yellow\"}]}",
            "required": True,
            "rows": 10,
        },
        {
            "name": "filter_json",
            "label": "Filter Params JSON",
            "type": "textarea",
            "default": "[1]",
            "required": False,
            "rows": 3,
        },
    ],
}


def get_form_schema(schema_id: str) -> list[dict[str, Any]]:
    return FORM_SCHEMAS.get(schema_id, [])
