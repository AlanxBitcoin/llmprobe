from __future__ import annotations

from typing import Any


FORM_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "single_word_form": [
        {"name": "word", "label": "Word", "type": "text", "default": "apple", "required": True},
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
