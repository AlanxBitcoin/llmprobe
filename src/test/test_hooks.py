from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Lightweight hooks-layer API surface test.
# - Should not require real model loading for basic contract checks.

from src.utils import hooks


def test_hooks_public_apis_exist():
    assert callable(hooks.extract_layer_hidden_states)
    assert callable(hooks.skip_layers)
    assert callable(hooks.disable_attention_heads)
    assert callable(hooks.extract_neuron_parameters)
    assert callable(hooks.compare_neuron_parameters)
