"""
直接用 "rain" 的 token embedding 乘 lm_head，输出 top-10 logits
（不经过任何 transformer 层）
"""
import torch
from safetensors.torch import load_file
import glob, os

MODEL_PATH = "D:/杂项/models/Meta-Llama-3-8B-Instruct"
TOKEN_ID = 30193  # "rain"

# 读取 token embedding（embed_tokens.weight）
print("读取 embedding 权重...")
st_files = sorted(glob.glob(os.path.join(MODEL_PATH, "*.safetensors")))
embed_weight = None
lm_head_weight = None

for f in st_files:
    data = load_file(f, device="cpu")
    if "model.embed_tokens.weight" in data and embed_weight is None:
        embed_weight = data["model.embed_tokens.weight"].float()
        print(f"  embed_tokens: {embed_weight.shape}  来自 {os.path.basename(f)}")
    if "lm_head.weight" in data and lm_head_weight is None:
        lm_head_weight = data["lm_head.weight"].float()
        print(f"  lm_head:      {lm_head_weight.shape}  来自 {os.path.basename(f)}")
    if embed_weight is not None and lm_head_weight is not None:
        break

# 取 rain 的 embedding 向量
rain_emb = embed_weight[TOKEN_ID]  # (hidden_size,)
print(f"\nrain token_id={TOKEN_ID}  embedding shape={rain_emb.shape}")
print(f"embedding 范数: {rain_emb.norm().item():.6f}")
print(f"embedding 最大值: {rain_emb.max().item():.6f}  最小值: {rain_emb.min().item():.6f}")

# 直接乘 lm_head（不经过任何层）
logits = lm_head_weight @ rain_emb  # (vocab_size,)

top_vals, top_ids = torch.topk(logits, 10)

# 读 tokenizer 词表（用 json 直接读）
import json
vocab_file = os.path.join(MODEL_PATH, "tokenizer.json")
with open(vocab_file, encoding="utf-8") as f:
    tok_data = json.load(f)
id2token = {v: k for k, v in tok_data["model"]["vocab"].items()}

print("\n" + "="*50)
print("rain embedding → lm_head 直接投影 Top-10")
print("="*50)
for rank, (val, tid) in enumerate(zip(top_vals.tolist(), top_ids.tolist()), 1):
    tok = id2token.get(tid, f"<id:{tid}>")
    print(f"  #{rank:2d}  {repr(tok):<25}  logit={val:.4f}")

# rain 自身排名
sorted_ids = torch.argsort(logits, descending=True)
rain_rank = (sorted_ids == TOKEN_ID).nonzero(as_tuple=True)[0].item() + 1
rain_logit = logits[TOKEN_ID].item()
print(f"\n  'rain' (id={TOKEN_ID}): 排名 #{rain_rank}  logit={rain_logit:.4f}")
print(f"  词表大小: {logits.shape[0]}")
