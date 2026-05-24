"""
把 'rain' 输入模型，提取第 8 层隐藏状态，保存到文件，
然后与其 embedding 做内积 / 余弦相似度比较。
"""
import torch
import json
from pathlib import Path
from safetensors import safe_open
import glob, os
from src.model_loader import load_local_model
from src.config import load_config

cfg = load_config("configs/default.yaml")
bundle = load_local_model(cfg)
model = bundle.model
tok   = bundle.tokenizer

WORD = "rain"
TARGET_LAYER = 8
OUT_DIR = Path("data/outputs/neuron_backup")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Tokenize ──────────────────────────────────────────────────
ids = tok.encode(WORD, add_special_tokens=False)
token_id = ids[0]
print(f"'{WORD}'  →  token_id={token_id},  tokens={[tok.decode([i]) for i in ids]}")

input_ids = torch.tensor([ids], device=next(model.parameters()).device)

# ── 2. Forward，钩子提取第8层输出 ────────────────────────────────
hidden_state = {}

def hook_fn(module, input, output):
    # output 可能是 tuple，第0个元素是隐藏状态 [batch, seq, hidden]
    h = output[0] if isinstance(output, tuple) else output
    hidden_state["layer8"] = h.detach().float().cpu()

hook = model.model.layers[TARGET_LAYER].register_forward_hook(hook_fn)

with torch.no_grad():
    model(input_ids=input_ids)

hook.remove()

# 取最后一个（也是唯一的）token 的隐藏状态
h8 = hidden_state["layer8"][0, -1, :]   # shape: [4096]
print(f"\n第 {TARGET_LAYER} 层隐藏状态 shape={h8.shape}")
print(f"  最大值:       {h8.max().item():.6f}")
print(f"  最小值:       {h8.min().item():.6f}")
print(f"  绝对值平均值: {h8.abs().mean().item():.6f}")
print(f"  标准差:       {h8.std().item():.6f}")

# ── 3. 保存第8层隐藏状态 ─────────────────────────────────────────
torch.save({"word": WORD, "token_id": token_id, "layer": TARGET_LAYER,
            "hidden_state": h8}, OUT_DIR / "rain_layer8.pt")
print(f"\n已保存: {OUT_DIR}/rain_layer8.pt")

# ── 4. 读取 embedding 向量 ───────────────────────────────────────
model_path = cfg["model"]["model_name_or_path"]
st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
emb_vec = None
for f in st_files:
    with safe_open(f, framework="pt", device="cpu") as sf:
        if "model.embed_tokens.weight" in sf.keys():
            emb_matrix = sf.get_tensor("model.embed_tokens.weight")
            emb_vec = emb_matrix[token_id].float()
            print(f"\nEmbedding 来自: {os.path.basename(f)}, shape={emb_vec.shape}")
            break

# ── 5. 比较 ──────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"  {'指标':<22}  embedding      layer-8 hidden")
print("="*50)
print(f"  {'最大值':<22}  {emb_vec.max().item():>10.6f}    {h8.max().item():>10.6f}")
print(f"  {'最小值':<22}  {emb_vec.min().item():>10.6f}    {h8.min().item():>10.6f}")
print(f"  {'绝对值均值':<22}  {emb_vec.abs().mean().item():>10.6f}    {h8.abs().mean().item():>10.6f}")
print(f"  {'标准差':<22}  {emb_vec.std().item():>10.6f}    {h8.std().item():>10.6f}")
print(f"  {'L2范数':<22}  {emb_vec.norm().item():>10.4f}    {h8.norm().item():>10.4f}")
print("="*50)

# 内积
dot = torch.dot(emb_vec, h8).item()

# 余弦相似度
cos = torch.nn.functional.cosine_similarity(
    emb_vec.unsqueeze(0), h8.unsqueeze(0)
).item()

# L2 距离
l2_dist = (emb_vec - h8).norm().item()

# 投影比（h8 在 emb 方向上的分量占 h8 总范数的比例）
proj_ratio = dot / (emb_vec.norm().item() * h8.norm().item() + 1e-12)

print(f"\n  内积 (dot product):      {dot:.6f}")
print(f"  余弦相似度 (cosine):     {cos:.6f}")
print(f"  L2 距离:                 {l2_dist:.4f}")
print(f"  L2范数放大倍数:          {h8.norm().item() / emb_vec.norm().item():.2f}x")

print("\n【解读】")
if abs(cos) > 0.9:
    interp = "极高，第8层几乎保留了embedding方向"
elif abs(cos) > 0.7:
    interp = "较高，第8层在embedding基础上有明显变换但方向相似"
elif abs(cos) > 0.4:
    interp = "中等，第8层已发生显著语义变换"
else:
    interp = "较低，第8层与embedding方向差异很大，语义编码已大幅重构"
print(f"  余弦相似度={cos:.4f}，方向相似度{interp}")
print(f"  L2范数从 {emb_vec.norm().item():.4f} → {h8.norm().item():.4f}，"
      f"放大了 {h8.norm().item()/emb_vec.norm().item():.1f} 倍")
