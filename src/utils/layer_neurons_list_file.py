from __future__ import annotations

from pathlib import Path
import json


_REL_PATH = Path("data/cache/layer_neuron_list.json")


def default_layer_neurons_payload() -> dict:
    return {
        "lists": [
            {
                "list_name": "example_a",
                "nLayer": 30,
                "neurons": [
                    [45, 20.0],
                    [1024, -5.0],
                ],
            },
            {
                "list_name": "example_b",
                "nLayer": 31,
                "neurons": [
                    [300, 8.0],
                ],
            },
        ]
    }


def layer_neurons_list_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return (root / _REL_PATH).resolve()


def ensure_layer_neurons_list_file(project_root: str | Path) -> Path:
    path = layer_neurons_list_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(default_layer_neurons_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return path


def load_layer_neurons_list_text(project_root: str | Path) -> tuple[Path, str]:
    path = ensure_layer_neurons_list_file(project_root)
    text = path.read_text(encoding="utf-8")
    return path, text
