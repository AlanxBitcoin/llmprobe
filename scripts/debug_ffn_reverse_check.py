from __future__ import annotations

import argparse
from collections import Counter

import torch

from src.config import load_config
from src.runtime_api import RuntimeRequest, start_llama_api
from src.study.study_layer_ffn_neuron_logits_table import (
    _build_ffn_w1_neuron_input_matrix,
    _rank_hidden_batch_by_logits,
    _resolve_bootstrap_token_id,
)


def _top_signature(row: list[dict], top_k: int) -> tuple[tuple[int, float], ...]:
    sig: list[tuple[int, float]] = []
    for item in row[:top_k]:
        tok = int(item.get("token_id", -1))
        val = float(item.get("logit", 0.0))
        sig.append((tok, round(val, 6)))
    return tuple(sig)


def _safe_console_text(value: object) -> str:
    s = str(value)
    return s.encode("ascii", "backslashreplace").decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug FFN reverse neuron logits diversity.")
    parser.add_argument("--config", default="configs/custom.yaml")
    parser.add_argument("--layer", type=int, default=0, help="0-based layer index for reverse mode")
    parser.add_argument("--activation", type=float, default=10.0)
    parser.add_argument("--count", type=int, default=16, help="how many neuron ids from 0 to test")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    api = start_llama_api(cfg)
    bundle = api.execute_model_call(RuntimeRequest(config=cfg, force_reload=False)).bundle
    model = bundle.model
    tokenizer = bundle.tokenizer

    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise RuntimeError("model.layers unavailable")
    num_layers = int(len(layers))
    if not (0 <= int(args.layer) < num_layers):
        raise RuntimeError(f"invalid layer={args.layer}, valid=[0,{num_layers - 1}]")

    layer_idx = int(args.layer)
    count = max(1, int(args.count))
    neuron_ids = list(range(count))

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    bos = _resolve_bootstrap_token_id(tokenizer)
    input_ids = torch.tensor([[int(bos)]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = out.hidden_states
    layer_input = hidden_states[layer_idx].detach().to(device=device, dtype=model_dtype)

    w1_batch, source = _build_ffn_w1_neuron_input_matrix(
        model,
        layer_idx=layer_idx,
        neuron_indices=neuron_ids,
        activation_value=float(args.activation),
    )

    row_norms = torch.linalg.vector_norm(w1_batch.to(dtype=torch.float32), dim=1)
    first = w1_batch[:1, :]
    max_diffs = torch.max(torch.abs(w1_batch - first), dim=1).values.to(dtype=torch.float32)
    distinct_vs_first = int(torch.sum(max_diffs > 1e-8).item())
    print(f"w1_source={source}")
    print(f"w1_shape={tuple(w1_batch.shape)}")
    print(f"w1_norm_min={float(row_norms.min().item()):.6f} w1_norm_max={float(row_norms.max().item()):.6f}")
    print(f"w1_distinct_vs_neuron0={distinct_vs_first}/{count - 1 if count > 1 else 0}")

    layer_input_last_hidden = layer_input[:, -1, :].expand(count, -1)
    compare_hidden_batch = (w1_batch * layer_input_last_hidden).to(dtype=model_dtype)
    rows = _rank_hidden_batch_by_logits(
        model=model,
        tokenizer=tokenizer,
        hidden_batch=compare_hidden_batch,
        top_k=int(args.top_k),
        apply_final_norm=False,
    )

    top1_ids = [int((row[0] if row else {}).get("token_id", -1)) for row in rows]
    top1_logits = [round(float((row[0] if row else {}).get("logit", 0.0)), 6) for row in rows]
    signatures = [_top_signature(row, int(args.top_k)) for row in rows]

    print(f"rows={len(rows)}")
    print(f"unique_top1_token_ids={len(set(top1_ids))}")
    print(f"unique_top1_logits={len(set(top1_logits))}")
    print(f"unique_top{int(args.top_k)}_signatures={len(set(signatures))}")
    print("top1_token_id_counts=", dict(Counter(top1_ids).most_common(10)))

    print("sample_rows:")
    for nid, row in zip(neuron_ids[: min(8, len(rows))], rows[: min(8, len(rows))]):
        first_item = row[0] if row else {}
        print(
            f"neuron={nid} top1_token_id={first_item.get('token_id')} "
            f"top1_text={_safe_console_text(first_item.get('text', ''))} top1_logit={first_item.get('logit')}"
        )


if __name__ == "__main__":
    main()
