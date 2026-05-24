"""
把 'rain' 的第8层隐藏状态注入第24层入口，
从第24层继续跑模型到末尾，获取最终隐藏状态和 logits top-10。
"""
import torch
import glob, os
from safetensors import safe_open
from transformers import AutoTokenizer
from src.model_loader import load_local_model
from src.config import load_config
from pathlib import Path

cfg = load_config("configs/default.yaml")
bundle = load_local_model(cfg)
model = bundle.model
tok   = bundle.tokenizer
model.eval()

OUT_DIR = Path("data/outputs/neuron_backup")

# ── 1. 加载保存的第8层隐藏状态 ───────────────────────────────────
data = torch.load(OUT_DIR / "rain_layer8.pt", weights_only=True)
h8   = data["hidden_state"].float()          # [4096]
word = data["word"]
print(f"已加载: '{word}' 第{data['layer']}层隐藏状态, shape={h8.shape}")

device = next(model.parameters()).device
h8_4d = h8.unsqueeze(0).unsqueeze(0).to(device)   # [1, 1, 4096]

# ── 2. 注入钩子：在第23层输出处替换为 h8，让第24层接收 ───────────
captured = {}

def replace_after_layer23(module, inp, output):
    """把第23层的输出替换成 h8，使第24层收到 h8 作为输入。"""
    if isinstance(output, tuple):
        return (h8_4d,) + output[1:]
    return h8_4d

def capture_final_norm(module, inp, output):
    """捕获最终 RMSNorm 的输出（进入 lm_head 之前）。"""
    captured["final_hidden"] = output.detach().float().cpu()

hook1 = model.model.layers[23].register_forward_hook(replace_after_layer23)
hook2 = model.model.norm.register_forward_hook(capture_final_norm)

# ── 3. 正常 forward（需要一个虚拟输入触发计算图） ────────────────
# 用 'rain' 自身的 token 作为虚拟输入（实际第0-23层结果被丢弃）
dummy_ids = torch.tensor(
    [tok.encode(word, add_special_tokens=False)], device=device
)

with torch.no_grad():
    out = model(input_ids=dummy_ids)

hook1.remove()
hook2.remove()

# ── 4. 提取结果 ──────────────────────────────────────────────────
# logits: [1, seq_len, vocab_size] → 取最后一个位置
logits_tensor = out.logits[0, -1, :].float().cpu()   # [vocab_size]
final_h = captured["final_hidden"][0, -1, :]          # [4096]

print(f"\n第24层→末尾 最终隐藏状态 shape={final_h.shape}")
print(f"  最大值:       {final_h.max().item():.6f}")
print(f"  最小值:       {final_h.min().item():.6f}")
print(f"  绝对值均值:   {final_h.abs().mean().item():.6f}")
print(f"  标准差:       {final_h.std().item():.6f}")
print(f"  L2范数:       {final_h.norm().item():.4f}")

# ── 5. 保存最终隐藏状态 ──────────────────────────────────────────
torch.save({
    "word": word,
    "inject_at_layer": 24,
    "source_layer": 8,
    "final_hidden": final_h,
    "logits": logits_tensor,
}, OUT_DIR / "rain_layer8_injected_at24_output.pt")
print(f"\n已保存: {OUT_DIR}/rain_layer8_injected_at24_output.pt")

# ── 6. logits Top-10 / Bottom-10 ─────────────────────────────────
probs = torch.softmax(logits_tensor, dim=0)
top_k = 10
top_ids    = logits_tensor.topk(top_k).indices
bottom_ids = logits_tensor.topk(top_k, largest=False).indices

print(f"\n{'='*64}")
print(f"  '{word}' 第8层→注入第24层→模型输出  Top-{top_k} 匹配词")
print(f"{'='*64}")
print(f"  {'排名':<4}  {'词':<22}  {'logit':>10}  {'概率':>12}  {'内积':>10}")
print(f"  {'-'*60}")
for rank, tid in enumerate(top_ids, 1):
    tid_v = tid.item()
    token_str = repr(tok.decode([tid_v]))
    lv = logits_tensor[tid_v].item()
    pv = probs[tid_v].item()
    print(f"  {rank:<4}  {token_str:<22}  {lv:>10.4f}  {pv:>12.6%}  {lv:>10.4f}")

print(f"\n{'='*64}")
print(f"  Bottom-{top_k}（最不匹配）")
print(f"{'='*64}")
print(f"  {'排名':<4}  {'词':<22}  {'logit':>10}")
print(f"  {'-'*40}")
for rank, tid in enumerate(bottom_ids, 1):
    tid_v = tid.item()
    token_str = repr(tok.decode([tid_v]))
    lv = logits_tensor[tid_v].item()
    print(f"  {rank:<4}  {token_str:<22}  {lv:>10.4f}")

# 'rain' 自身排名
rain_id   = tok.encode(word, add_special_tokens=False)[0]
rain_rank = (logits_tensor > logits_tensor[rain_id]).sum().item() + 1
print(f"\n  '{word}' (token_id={rain_id}) 自身排名: #{rain_rank}")
print(f"  logit={logits_tensor[rain_id].item():.4f}, 概率={probs[rain_id].item():.6%}")

# 与原始第8层 Top-1 相比
print(f"\n【与上一实验对比】")
print(f"  上一实验（第8层直接投影）Top-1: '.GraphicsUnit'，logit≈44.6，rain排#124935")
print(f"  本实验（注入第24层→跑完剩余层）rain排#{rain_rank}，结果{'有意义' if rain_rank < 10000 else '仍偏离'}")
