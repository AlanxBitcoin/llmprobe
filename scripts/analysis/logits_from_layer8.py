"""
用保存的 'rain' 第8层隐藏状态，乘以 lm_head（反嵌入矩阵），
得到词表 logits，输出 top-10 匹配词。
无需重新加载完整模型。
"""
import torch
import glob, os
from safetensors import safe_open
from transformers import AutoTokenizer
from src.config import load_config

cfg = load_config("configs/default.yaml")
model_path = cfg["model"]["model_name_or_path"]

# ── 1. 加载保存的第8层隐藏状态 ───────────────────────────────────
data = torch.load("data/outputs/neuron_backup/rain_layer8.pt", weights_only=True)
h8 = data["hidden_state"].float()          # [4096]
word = data["word"]
print(f"已加载: '{word}' 第{data['layer']}层隐藏状态, shape={h8.shape}")

# ── 2. 从 safetensors 读取 lm_head.weight ───────────────────────
# Llama-3 的 lm_head 不共享 embedding 权重，单独存储
st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
lm_head_w = None
for f in st_files:
    with safe_open(f, framework="pt", device="cpu") as sf:
        if "lm_head.weight" in sf.keys():
            lm_head_w = sf.get_tensor("lm_head.weight").float()  # [vocab, 4096]
            print(f"lm_head.weight: shape={lm_head_w.shape}, 来自 {os.path.basename(f)}")
            break

if lm_head_w is None:
    # 部分模型 lm_head 与 embed_tokens 共享权重
    for f in st_files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            if "model.embed_tokens.weight" in sf.keys():
                lm_head_w = sf.get_tensor("model.embed_tokens.weight").float()
                print(f"lm_head 共享 embed_tokens: shape={lm_head_w.shape}")
                break

# ── 3. 计算 logits ────────────────────────────────────────────────
# logits[i] = lm_head_w[i] · h8   （内积，维度 [vocab_size]）
logits = lm_head_w @ h8            # [vocab_size]
probs  = torch.softmax(logits, dim=0)

# ── 4. 加载 tokenizer 用于解码 ──────────────────────────────────
tok = AutoTokenizer.from_pretrained(model_path)

# ── 5. Top-10 / Bottom-10 ────────────────────────────────────────
top_k = 10
top_ids    = logits.topk(top_k).indices
bottom_ids = logits.topk(top_k, largest=False).indices

print(f"\n{'='*62}")
print(f"  'rain' 第8层隐藏状态 → lm_head 投影  Top-{top_k} 匹配词")
print(f"{'='*62}")
print(f"  {'排名':<4}  {'词':<20}  {'logit':>10}  {'概率':>12}  {'内积':>10}")
print(f"  {'-'*58}")
for rank, tid in enumerate(top_ids, 1):
    tid = tid.item()
    token_str = repr(tok.decode([tid]))
    logit_val = logits[tid].item()
    prob_val  = probs[tid].item()
    dot_val   = (lm_head_w[tid] * h8).sum().item()
    print(f"  {rank:<4}  {token_str:<20}  {logit_val:>10.4f}  {prob_val:>12.6%}  {dot_val:>10.4f}")

print(f"\n{'='*62}")
print(f"  Bottom-{top_k}（最不匹配）")
print(f"{'='*62}")
print(f"  {'排名':<4}  {'词':<20}  {'logit':>10}  {'内积':>10}")
print(f"  {'-'*48}")
for rank, tid in enumerate(bottom_ids, 1):
    tid = tid.item()
    token_str = repr(tok.decode([tid]))
    logit_val = logits[tid].item()
    dot_val   = (lm_head_w[tid] * h8).sum().item()
    print(f"  {rank:<4}  {token_str:<20}  {logit_val:>10.4f}  {dot_val:>10.4f}")

# ── 6. 查找 'rain' 自身的排名 ────────────────────────────────────
rain_id  = tok.encode("rain", add_special_tokens=False)[0]
rain_rank = (logits > logits[rain_id]).sum().item() + 1
print(f"\n  'rain' (token_id={rain_id}) 自身排名: #{rain_rank}")
print(f"  logit={logits[rain_id].item():.4f}, 概率={probs[rain_id].item():.6%}")
print(f"\n【说明】")
print(f"  此处使用第8层输出直接投影（未经过第9-31层+最终RMSNorm），")
print(f"  反映的是第8层对词义方向的早期编码，非标准的下一词预测。")
