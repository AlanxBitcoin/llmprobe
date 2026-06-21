from __future__ import annotations


def validate_jump_to_layer_zero_based(*, jump_to_layer: int, layer_count: int) -> tuple[int, int]:
    """Validate 0-based jump target and return (jump_to_layer, start_layer_idx)."""
    num_layers = int(layer_count)
    if num_layers <= 0:
        raise ValueError("layer_count must be positive")
    jump = int(jump_to_layer)
    if not (0 <= jump < num_layers):
        raise ValueError(f"jump_to_layer out of range: {jump}, valid=[0,{num_layers - 1}]")
    # Inject as target layer input -> start from (target_layer - 1) output.
    return int(jump), int(jump - 1)


def validate_jump_to_layer_1based(*, jump_to_layer: int, layer_count: int) -> tuple[int, int]:
    """Backward-compatible wrapper for old 1-based callers."""
    num_layers = int(layer_count)
    jump_1 = int(jump_to_layer)
    if not (1 <= jump_1 <= num_layers):
        raise ValueError(f"jump_to_layer out of range: {jump_1}, valid=[1,{num_layers}]")
    return validate_jump_to_layer_zero_based(
        jump_to_layer=int(jump_1 - 1),
        layer_count=int(num_layers),
    )


def validate_shortcut_layers_zero_based(
    *,
    source_layer: int,
    target_layer: int,
    layer_count: int,
) -> tuple[int, int, int]:
    """
    Validate 0-based shortcut pair and return (source, target, start_layer_idx).

    start_layer_idx is the index fed to `starting_from_middle_layer`, i.e. target-1.
    """
    num_layers = int(layer_count)
    if num_layers <= 0:
        raise ValueError("layer_count must be positive")
    src = int(source_layer)
    tgt = int(target_layer)
    if not (0 <= src < num_layers):
        raise ValueError(f"shortcut_start_layer out of range: {src}, valid=[0,{num_layers - 1}]")
    if not (0 <= tgt < num_layers):
        raise ValueError(f"shortcut_target_layer out of range: {tgt}, valid=[0,{num_layers - 1}]")
    if src >= tgt:
        raise ValueError(
            f"shortcut requires start < target, got start={src}, target={tgt}"
        )
    return int(src), int(tgt), int(tgt - 1)
