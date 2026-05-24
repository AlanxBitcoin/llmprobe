from __future__ import annotations

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
    return [action.command]
