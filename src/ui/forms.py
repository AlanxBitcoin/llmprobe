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
    "single_word_top_100_neurons_form": [
        {"name": "word", "label": "Word", "type": "text", "default": "apple", "required": True},
        {"name": "top_k_neurons", "label": "Top K Neurons", "type": "number", "default": 100, "min": 1},
        {"name": "intervention_layer", "label": "Intervention Layer", "type": "number", "default": 30, "min": 0, "step": 1},
    ],
    "layer_neuron_logits_table_form": [
        {"name": "intervention_layer", "label": "Layer", "type": "number", "default": 30, "min": 1, "step": 1},
        {"name": "activation_value", "label": "Activation Value", "type": "number", "default": 10.0},
        {"name": "use_prefix_context", "label": "Use Prefix Context", "type": "checkbox", "default": False},
        {"name": "prefix_text", "label": "Prefix Words/Text", "type": "text", "default": "apple", "required": False},
        {"name": "return_batch_size", "label": "Return Batch Size", "type": "number", "default": 1000, "min": 1},
    ],
    "layer_ffn_neuron_logits_table_form": [
        {"name": "intervention_layer", "label": "Layer", "type": "number", "default": 30, "min": 1, "step": 1},
        {"name": "activation_value", "label": "Activation Value", "type": "number", "default": 10.0},
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
}


def get_form_schema(schema_id: str) -> list[dict[str, Any]]:
    return FORM_SCHEMAS.get(schema_id, [])
