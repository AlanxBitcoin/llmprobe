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
    ],
    "single_word_top_100_neurons_form": [
        {"name": "word", "label": "Word", "type": "text", "default": "apple", "required": True},
        {"name": "top_k_neurons", "label": "Top K Neurons", "type": "number", "default": 100, "min": 1},
        {"name": "intervention_layer", "label": "Intervention Layer", "type": "number", "default": 30, "min": 0},
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
