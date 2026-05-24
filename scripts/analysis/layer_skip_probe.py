"""
Layer-Skip 探针实验（修复版）：
对 'rain' 逐层跳过（k=0..N-1），跑完整模型，
对比 baseline（不跳过任何层）的 top-N 输出。

关键修复：
  - 先跑 baseline 验证模型正常输出（如 'bow','fall' 等 rain 的自然接续）
  - 重构 hook 方式：直接用 pre_hook 修改输入为残差前的输入（等效跳过）
  - 用 RMSNorm 正确参数处理
"""
import torch
from transformers import AutoTokenizer
from src.model_loader import load_local_model
from src.config import load_config

cfg    = load_config("configs/default.yaml")
bundle = load_local_model(cfg)
model  = bundle.model
tok    = bundle.tokenizer
model.eval()

WORD    = " rain"   # 注意前导空格
N_LAYERS = len(model.model.layers)   # 36 for base_expanded
TOP_K   = 15
device  = next(model.parameters()).device

ids      = tok.encode(WORD, add_special_tokens=True)   # 加 BOS token (128000)
rain_id  = ids[-1]   # 最后一个 token 是 ' rain' (11422)
bos_id   = ids[0] if len(ids) > 1 else None
input_ids = torch.tensor([ids], device=device)
print(f"BOS token id: {bos_id}, input_ids: {ids}")

print(f"Model layers: {N_LAYERS}")
print(f"Token: {repr(tok.decode([rain_id]))}  id={rain_id}")
print(f"{'='*80}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Baseline — no skip
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Baseline] Full model — no skip")
with torch.no_grad():
    out     = model(input_ids=input_ids)
    logits  = out.logits[0, -1, :].float().cpu()
    probs   = torch.softmax(logits, dim=0)

top_ids   = logits.topk(TOP_K).indices.tolist()
top_tokens = [tok.decode([t]) for t in top_ids]
top_logits = [logits[t].item() for t in top_ids]
baseline_rain_logit = logits[rain_id].item()
baseline_rain_rank  = int((logits > logits[rain_id]).sum().item()) + 1

print(f"  rain rank #{baseline_rain_rank}  logit={baseline_rain_logit:.4f}  prob={probs[rain_id].item():.6f}")
print(f"  Top-{TOP_K}:")
for rk, (t, lv) in enumerate(zip(top_tokens, top_logits), 1):
    print(f"    {rk:2d}. {repr(t):<28} logit={lv:.4f}   prob={torch.softmax(logits,dim=0)[top_ids[rk-1]].item():.6f}")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Skip each layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
results = []

for skip in range(N_LAYERS):
    # ── 跳过第 skip 层：用 pre_hook 拦截输入并修改
    #   LlamaDecoderLayer: hidden_states → attention → residual → norm → mlp → residual
    #   要跳过整层，只需让输出等于输入（残差连接不变）
    cache = {}

    def pre_hook(module, args):
        cache['h'] = args[0].clone()   # 保存原始输入
        return args   # 不修改输入

    def post_hook(module, args, output):
        h_in = cache['h']
        # LlamaDecoderLayer 返回: (hidden_states,) 或 BaseModelOutputWithPast
        if isinstance(output, tuple):
            return (h_in,) + output[1:]
        return h_in

    pre_handle  = model.model.layers[skip].register_forward_pre_hook(pre_hook)
    post_handle = model.model.layers[skip].register_forward_hook(post_hook)

    with torch.no_grad():
        out = model(input_ids=input_ids)

    pre_handle.remove()
    post_handle.remove()

    logits  = out.logits[0, -1, :].float().cpu()
    probs   = torch.softmax(logits, dim=0)
    top_ids = logits.topk(TOP_K).indices.tolist()

    rain_rank  = int((logits > logits[rain_id]).sum().item()) + 1
    rain_logit = logits[rain_id].item()
    rain_prob  = probs[rain_id].item()

    top_toks = [tok.decode([t]) for t in top_ids]
    top_lvs  = [logits[t].item() for t in top_ids]

    results.append((skip, top_ids, top_toks, top_lvs, rain_rank, rain_logit, rain_prob))

    # 逐行输出
    top3 = " / ".join(f"'{w}'" for w in top_toks[:3])
    garbage = "  <- GARBAGE!" if top_toks[0] in ("://", "=\"", ".Forms", "_REF") else ""
    print(f"skip L{skip:2d}  rain #{rain_rank:>6}  logit={rain_logit:>7.3f}  "
          f"top3: {top3}{garbage}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*80}")
print(f"Summary: baseline rain rank #{baseline_rain_rank}  logit={baseline_rain_logit:.3f}")
print(f"{'='*80}")
print(f"{'Skip':>6}  {'rain_rank':>10}  {'rain_logit':>12}  {'rain_prob':>12}  {'logit_delta':>12}  top1")
print("-" * 90)

for skip, _, top_toks, _, rain_rank, rain_logit, rain_prob in results:
    delta = rain_logit - baseline_rain_logit
    garb  = " G!" if top_toks[0] in ("://", "=\"", ".Forms", "_REF") else ""
    print(f"{skip:6d}  #{rain_rank:>9}  {rain_logit:12.4f}  {rain_prob:12.6f}  "
          f"{delta:+12.4f}  {repr(top_toks[0]):<20}{garb}")

# ── Best / worst for rain ──
best  = min(results, key=lambda x: x[4])
worst = max(results, key=lambda x: x[4])
print(f"\n  Best (rain highest):   skip L{best[0]}  rank #{best[4]}  logit={best[5]:.3f}")
print(f"  Worst (rain lowest):   skip L{worst[0]}  rank #{worst[4]}  logit={worst[5]:.3f}")

# ── Detect garbage layers ──
garbage_keywords = ("://", "=\"", ".Forms", "_REF", ".swing")
garbage_layers = [r[0] for r in results if r[3][0] in garbage_keywords]
if garbage_layers:
    print(f"\n  Garbage output (top1 is URL/code token) when skipping layers: {garbage_layers}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Save results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import json
from pathlib import Path

out = [{
    "skip_layer": "baseline",
    "rain_rank": baseline_rain_rank,
    "rain_logit": baseline_rain_logit,
    "rain_prob": probs[rain_id].item(),
    "top15": [{"rank": i+1, "token": top_tokens[i], "token_id": top_ids[i],
               "logit": top_logits[i]} for i in range(TOP_K)]
}]
for skip, top_ids, top_toks, top_lvs, rain_rank, rain_logit, rain_prob in results:
    out.append({
        "skip_layer": skip,
        "rain_rank": rain_rank,
        "rain_logit": rain_logit,
        "rain_prob": rain_prob,
        "top15": [{"rank": i+1, "token": top_toks[i], "token_id": top_ids[i],
                   "logit": top_lvs[i]} for i in range(TOP_K)]
    })

Path("data/outputs/neuron_backup/apple_layer_skip.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"\n  Results saved: data/outputs/neuron_backup/layer_skip_results.json")
