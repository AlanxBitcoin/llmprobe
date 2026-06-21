from __future__ import annotations

import torch

from src.study.study_chat_attention_word_replacement import (
    _apply_ignore_token_mode_3,
    _merge_past_key_values,
)


def _build_legacy_cache(*, layer_count: int, seq_len: int, dim: int, fill: float):
    layers = []
    for _ in range(layer_count):
        k = torch.full((1, 1, seq_len, dim), float(fill), dtype=torch.float32)
        v = torch.full((1, 1, seq_len, dim), float(fill), dtype=torch.float32)
        layers.append((k, v))
    return tuple(layers)


def test_mode3_full_replace_all_layers_zero_to_one():
    original = _build_legacy_cache(layer_count=4, seq_len=5, dim=3, fill=0.0)
    replaced = _build_legacy_cache(layer_count=4, seq_len=5, dim=3, fill=1.0)

    out = _merge_past_key_values(
        original=original,
        replaced=replaced,
        replace_layer_indices_zero_based={0, 1, 2, 3},
        replace_k=True,
        kv_replace_mode=3,
        target_token_index=None,
    )

    for k, v in out:
        assert torch.all(k == 1.0)
        assert torch.all(v == 1.0)


def test_mode3_v_only_selected_layers():
    original = _build_legacy_cache(layer_count=3, seq_len=4, dim=2, fill=0.0)
    replaced = _build_legacy_cache(layer_count=3, seq_len=4, dim=2, fill=1.0)

    out = _merge_past_key_values(
        original=original,
        replaced=replaced,
        replace_layer_indices_zero_based={1},
        replace_k=False,
        kv_replace_mode=3,
        target_token_index=None,
    )

    # layer 0 unchanged
    assert torch.all(out[0][0] == 0.0)
    assert torch.all(out[0][1] == 0.0)
    # layer 1: only V replaced
    assert torch.all(out[1][0] == 0.0)
    assert torch.all(out[1][1] == 1.0)
    # layer 2 unchanged
    assert torch.all(out[2][0] == 0.0)
    assert torch.all(out[2][1] == 0.0)


def test_mode2_selected_layers_target_and_after_boundary():
    # mode2 behavior:
    # - selected layers: replace target token position only
    # - layers after max(selected): replace positions target+1 ...
    original = _build_legacy_cache(layer_count=4, seq_len=5, dim=2, fill=0.0)
    replaced = _build_legacy_cache(layer_count=4, seq_len=5, dim=2, fill=1.0)

    out = _merge_past_key_values(
        original=original,
        replaced=replaced,
        replace_layer_indices_zero_based={1, 2},
        replace_k=True,
        kv_replace_mode=2,
        target_token_index=1,
    )

    # layer 0 unchanged
    assert torch.all(out[0][0] == 0.0)
    assert torch.all(out[0][1] == 0.0)

    # layer 1/2: only target index=1 replaced
    for li in (1, 2):
        k = out[li][0]
        v = out[li][1]
        assert torch.all(k[:, :, 1, :] == 1.0)
        assert torch.all(v[:, :, 1, :] == 1.0)
        assert torch.all(k[:, :, 0, :] == 0.0)
        assert torch.all(v[:, :, 0, :] == 0.0)
        assert torch.all(k[:, :, 2:, :] == 0.0)
        assert torch.all(v[:, :, 2:, :] == 0.0)

    # layer 3 (> max selected layer=2): positions >=2 replaced
    k3 = out[3][0]
    v3 = out[3][1]
    assert torch.all(k3[:, :, :2, :] == 0.0)
    assert torch.all(v3[:, :, :2, :] == 0.0)
    assert torch.all(k3[:, :, 2:, :] == 1.0)
    assert torch.all(v3[:, :, 2:, :] == 1.0)


def test_mode3_ignore_restores_positions_to_original():
    original = _build_legacy_cache(layer_count=2, seq_len=6, dim=2, fill=0.0)
    mixed = _build_legacy_cache(layer_count=2, seq_len=6, dim=2, fill=1.0)

    out = _apply_ignore_token_mode_3(
        mixed_past=mixed,
        original_past=original,
        ignore_positions=[1, 4],
        replace_layer_indices_zero_based={0, 1},
        replace_k=True,
    )

    for k, v in out:
        # ignored positions restored to original zeros
        assert torch.all(k[:, :, 1, :] == 0.0)
        assert torch.all(v[:, :, 1, :] == 0.0)
        assert torch.all(k[:, :, 4, :] == 0.0)
        assert torch.all(v[:, :, 4, :] == 0.0)
        # non-ignored positions remain ones
        assert torch.all(k[:, :, 0, :] == 1.0)
        assert torch.all(v[:, :, 0, :] == 1.0)
        assert torch.all(k[:, :, 2, :] == 1.0)
        assert torch.all(v[:, :, 2, :] == 1.0)


def test_mode2_ignore_restore_all_except_target_position():
    # Emulate mode2 final-cache restoration behavior with 0/1 toy caches:
    # target token keeps replaced value (1), ignored positions are restored to original (0).
    layer_count = 3
    seq_len = 7
    target_pos = 3
    original = _build_legacy_cache(layer_count=layer_count, seq_len=seq_len, dim=2, fill=0.0)
    mixed = _build_legacy_cache(layer_count=layer_count, seq_len=seq_len, dim=2, fill=1.0)
    ignore_positions = [0, 1, 2, 4, 5, 6]  # all except target

    out = mixed
    for pos in ignore_positions:
        out = _merge_past_key_values(
            original=out,
            replaced=original,
            replace_layer_indices_zero_based={0, 1, 2},
            replace_k=True,
            kv_replace_mode=1,
            target_token_index=int(pos),
        )

    for k, v in out:
        for pos in range(seq_len):
            expected = 1.0 if pos == target_pos else 0.0
            assert torch.all(k[:, :, pos, :] == expected)
            assert torch.all(v[:, :, pos, :] == expected)
