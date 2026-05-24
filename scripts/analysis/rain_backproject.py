"""
倒推实验：取 lm_head.weight[rain_id] 作为"输出方向向量"
1. 与 embed_tokens 做余弦相似度，找输入空间最近邻
2. 把该向量当 embedding 注入模型，看前向输出
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "D:/杂项/models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
model.eval()

embed_matrix = model.model.embed_tokens.weight   # [vocab, hidden]
lm_head_matrix = model.lm_head.weight            # [vocab, hidden]
bos_id = tokenizer.bos_token_id

# rain 的 token id
rain_id = tokenizer.encode(" rain", add_special_tokens=False)[-1]
rain_token = tokenizer.decode([rain_id])
print(f"rain token: {repr(rain_token)}, id={rain_id}")

# ── 1. 取 lm_head 中 rain 对应的行（即模型输出 rain 所需的隐藏状态方向）
rain_out_vec = lm_head_matrix[rain_id].detach().float()   # [hidden]
print(f"\nlm_head[rain] 范数: {rain_out_vec.norm().item():.4f}")
print(f"embed_tokens[rain] 范数: {embed_matrix[rain_id].detach().float().norm().item():.4f}")

# 自身余弦相似度
cos_self = F.cosine_similarity(
    rain_out_vec.unsqueeze(0),
    embed_matrix[rain_id].detach().float().unsqueeze(0)
).item()
print(f"cos(lm_head[rain], embed[rain]) = {cos_self:.6f}")

# ── 2. 与整个 embed_tokens 做余弦相似度，找最近邻
print("\n【lm_head[rain] 在输入embedding空间的最近邻 Top-20】")
print("=" * 55)
emb_float = embed_matrix.detach().float()                 # [vocab, hidden]
# 批量余弦相似度（向量归一化后做矩阵乘法）
rain_norm = rain_out_vec / rain_out_vec.norm()
emb_norms = F.normalize(emb_float, dim=1)                 # [vocab, hidden]

# 在CPU上分块计算避免OOM
chunk = 32768
scores = []
for i in range(0, emb_norms.shape[0], chunk):
    s = (emb_norms[i:i+chunk] @ rain_norm).tolist()
    scores.extend(s)

scores_t = torch.tensor(scores)
top_vals, top_ids = torch.topk(scores_t, 20)

for rank, (v, tid) in enumerate(zip(top_vals.tolist(), top_ids.tolist()), 1):
    tok = tokenizer.decode([tid])
    marker = " ← rain自身" if tid == rain_id else ""
    print(f"  #{rank:2d}  {repr(tok):<25} cos={v:.6f}{marker}")

# rain 自身的排名
rain_rank = (scores_t >= scores_t[rain_id]).sum().item()
print(f"\n  rain 自身在此排序中的排名: #{rain_rank}  cos={scores_t[rain_id].item():.6f}")

# ── 3. 把 lm_head[rain] 当 embedding 注入模型，看前向输出
print("\n【将 lm_head[rain] 向量作为 token embedding 注入模型，前向传播 Top-10】")
print("=" * 55)
bos_emb = embed_matrix[bos_id].to(model.device).to(model.dtype).reshape(1,1,-1)
rain_emb_inject = rain_out_vec.to(model.device).to(model.dtype).reshape(1,1,-1)
h = torch.cat([bos_emb, rain_emb_inject], dim=1)

with torch.no_grad():
    logits = model(inputs_embeds=h).logits[0, -1].float()

top_vals2, top_ids2 = torch.topk(logits, 10)
for rank, (v, tid) in enumerate(zip(top_vals2.tolist(), top_ids2.tolist()), 1):
    tok = tokenizer.decode([tid])
    print(f"  #{rank:2d}  {repr(tok):<22}  logit={v:.4f}")

# ── 4. 对比：正常 rain embedding 注入时的输出（复现前值）
print("\n【正常 embed_tokens[rain] 注入模型，前向传播 Top-10】（对比）")
print("=" * 55)
rain_emb_normal = embed_matrix[rain_id].to(model.device).to(model.dtype).reshape(1,1,-1)
h2 = torch.cat([bos_emb, rain_emb_normal], dim=1)

with torch.no_grad():
    logits2 = model(inputs_embeds=h2).logits[0, -1].float()

top_vals3, top_ids3 = torch.topk(logits2, 10)
for rank, (v, tid) in enumerate(zip(top_vals3.tolist(), top_ids3.tolist()), 1):
    tok = tokenizer.decode([tid])
    print(f"  #{rank:2d}  {repr(tok):<22}  logit={v:.4f}")
