"""
层间相似度探针：输入"rain"，记录每层与上一层的内积和余弦相似度，输出最终top-10
"""
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "D:/杂项/models/Meta-Llama-3-8B-Instruct"

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
num_layers = len(model.model.layers)
print(f"模型层数: {num_layers}")

# 输入 "rain"
word = "rain"
token_ids = tokenizer.encode(word, add_special_tokens=False)
print(f"词: {repr(word)}  token_id={token_ids}")

# 用 BOS + word 作为输入（更贴近实际推理）
input_ids = torch.tensor([[tokenizer.bos_token_id] + token_ids]).to(model.device)

# ---- 收集所有层的隐藏状态 ----
hidden_states_list = []

hooks = []

def make_hook(layer_idx):
    def hook(module, input, output):
        # output 可能是 tuple 或直接是张量 (batch, seq, hidden)
        hs = output[0] if isinstance(output, tuple) else output
        h = hs[0, -1, :].detach().float()  # 取最后一个token的隐藏状态
        hidden_states_list.append((layer_idx, h))
    return hook

# 注册 hook 在每层输出
for i, layer in enumerate(model.model.layers):
    h = layer.register_forward_hook(make_hook(i))
    hooks.append(h)

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)

for h in hooks:
    h.remove()

# 也取embedding层（第一个隐藏状态）
embedding_h = outputs.hidden_states[0][0, -1, :].detach().float()

# ---- 计算层间内积和余弦相似度 ----
print()
print(f"{'层':>4}  {'与上一层内积':>14}  {'余弦相似度':>12}  {'L2范数':>10}")
print("  " + "-" * 55)

all_hs = [embedding_h] + [h for _, h in sorted(hidden_states_list)]

results = []
for i in range(1, len(all_hs)):
    prev = all_hs[i - 1]
    curr = all_hs[i]
    dot = torch.dot(prev, curr).item()
    cos = torch.nn.functional.cosine_similarity(prev.unsqueeze(0), curr.unsqueeze(0)).item()
    norm_curr = curr.norm().item()
    layer_label = f"层{i-1:2d}" if i > 1 else f"嵌入→层{i-1}"
    # i==1: embedding→layer0, i==2: layer0→layer1, ...
    if i == 1:
        label = f"嵌入 → 层 0"
    else:
        label = f"层{i-2:2d} → 层{i-1:2d}"
    print(f"  {label}   内积={dot:>12.2f}   cos={cos:>8.5f}   ‖h‖={norm_curr:>9.2f}")
    results.append({
        "from": "embedding" if i == 1 else f"layer{i-2}",
        "to": f"layer{i-1}",
        "dot_product": dot,
        "cosine_similarity": cos,
        "l2_norm_to": norm_curr,
    })

# ---- 最终输出 top-10 ----
print()
print("=" * 55)
print("最终输出 Top-10 词（完整推理路径）")
print("=" * 55)

final_logits = outputs.logits[0, -1, :]  # (vocab,)
top_vals, top_ids = torch.topk(final_logits, 10)

# rain 排名
rain_token_id = token_ids[0] if token_ids else tokenizer.encode("rain", add_special_tokens=False)[0]
sorted_ids = torch.argsort(final_logits, descending=True)
rain_rank = (sorted_ids == rain_token_id).nonzero(as_tuple=True)[0].item() + 1
rain_logit = final_logits[rain_token_id].item()

for rank, (val, tid) in enumerate(zip(top_vals.tolist(), top_ids.tolist()), 1):
    tok = tokenizer.decode([tid])
    print(f"  #{rank:2d}  {repr(tok):<25}  logit={val:.4f}")

print()
print(f"  'rain' (token_id={rain_token_id}): 排名 #{rain_rank}  logit={rain_logit:.4f}")

# ---- 统计摘要 ----
print()
print("=" * 55)
print("层间余弦相似度统计")
print("=" * 55)
cos_vals = [r["cosine_similarity"] for r in results]
print(f"  最小 cos: {min(cos_vals):.5f}  (最大变化层: {results[cos_vals.index(min(cos_vals))]['to']})")
print(f"  最大 cos: {max(cos_vals):.5f}  (最小变化层: {results[cos_vals.index(max(cos_vals))]['to']})")
print(f"  平均 cos: {sum(cos_vals)/len(cos_vals):.5f}")

dot_vals = [r["dot_product"] for r in results]
print(f"  最小内积: {min(dot_vals):.2f}  (层: {results[dot_vals.index(min(dot_vals))]['to']})")
print(f"  最大内积: {max(dot_vals):.2f}  (层: {results[dot_vals.index(max(dot_vals))]['to']})")

# 保存结果
out_path = "data/outputs/neuron_backup/layer_similarity_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "word": word,
        "token_id": rain_token_id,
        "rain_final_rank": rain_rank,
        "rain_final_logit": rain_logit,
        "top10": [{"rank": i+1, "token": tokenizer.decode([tid.item()]), "logit": v.item()}
                  for i, (v, tid) in enumerate(zip(top_vals, top_ids))],
        "layer_similarities": results,
    }, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {out_path}")
