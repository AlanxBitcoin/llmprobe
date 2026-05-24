"""
注意力头消融探针：对 "rain" 逐个关闭每个注意力头，观察输出 Top-10 变化
36层 × 32头 = 1152 次前向传播
"""
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "D:/杂项/models/Meta-Llama-3-8B-Instruct"
WORD = "rain"

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
)
model.eval()

num_layers = len(model.model.layers)
num_heads  = model.config.num_attention_heads
head_dim   = model.config.hidden_size // num_heads
print(f"层数={num_layers}  头数={num_heads}  head_dim={head_dim}")

# 构造输入：BOS + rain
token_id = tokenizer.encode(WORD, add_special_tokens=False)[0]
input_ids = torch.tensor([[tokenizer.bos_token_id, token_id]]).to(model.device)
print(f"词: {repr(WORD)}  token_id={token_id}\n")

# ---- 基线（无消融）----
with torch.no_grad():
    baseline_logits = model(input_ids).logits[0, -1].float()
baseline_top_ids = torch.argsort(baseline_logits, descending=True)
baseline_rain_rank = (baseline_top_ids == token_id).nonzero()[0].item() + 1
baseline_top10 = [(tokenizer.decode([i.item()]), baseline_logits[i].item())
                  for i in torch.topk(baseline_logits, 10).indices]
print(f"基线 rain排名=#{baseline_rain_rank}  Top-1={repr(baseline_top10[0][0])}")
print()

results = []

# 每次关闭头编号 head_idx（跨所有36层同时关闭），共32次
for head_idx in range(num_heads):
    start = head_idx * head_dim
    end   = (head_idx + 1) * head_dim

    def make_hook(s, e):
        def hook(module, args):
            x = args[0].clone()
            x[:, :, s:e] = 0
            return (x,)
        return hook

    # 在所有层的 o_proj 上同时注册 hook
    hooks = [
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
            make_hook(start, end)
        )
        for l in range(num_layers)
    ]

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1].float()

    for h in hooks:
        h.remove()

    top_vals, top_ids = torch.topk(logits, 10)
    rain_rank  = (torch.argsort(logits, descending=True) == token_id).nonzero()[0].item() + 1
    rain_logit = logits[token_id].item()
    top10 = [{"token": tokenizer.decode([tid.item()]), "logit": v.item()}
             for v, tid in zip(top_vals, top_ids)]

    results.append({
        "head":       head_idx,
        "rain_rank":  rain_rank,
        "rain_logit": rain_logit,
        "top10":      top10,
    })

    top1 = top10[0]["token"]
    print(f"  关闭头{head_idx:2d}（所有{num_layers}层）  rain#{rain_rank:>6}  "
          f"logit={rain_logit:>7.3f}  Top-1={repr(top1)}")

# ---- 保存 ----
out_path = "data/outputs/neuron_backup/head_ablation_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "word": WORD,
        "token_id": token_id,
        "baseline_rain_rank": baseline_rain_rank,
        "baseline_top10": [{"token": t, "logit": l} for t, l in baseline_top10],
        "note": "每次关闭该头编号在所有层的贡献（共32次前向传播）",
        "ablations": results,
    }, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {out_path}")

# ---- 摘要 ----
print("\n" + "="*60)
print("摘要：rain 排名变化最大的注意力头（跨所有层）")
print("="*60)

by_best = sorted(results, key=lambda x: x["rain_rank"])[:5]
print("\n关闭后 rain 排名最靠前（这些头在抑制 rain）：")
for r in by_best:
    print(f"  头{r['head']:2d}  rain#{r['rain_rank']:>6}  Top-1={repr(r['top10'][0]['token'])}")

by_worst = sorted(results, key=lambda x: x["rain_rank"], reverse=True)[:5]
print("\n关闭后 rain 排名最靠后（这些头在促进 rain）：")
for r in by_worst:
    print(f"  头{r['head']:2d}  rain#{r['rain_rank']:>6}  Top-1={repr(r['top10'][0]['token'])}")
