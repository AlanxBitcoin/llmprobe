"""
倒推实验 Step 1:
  目标：找到 h_34（第34层输出 = 第35层输入），使得：
    h_34 → MLP_35（带残差）→ final_norm → lm_head → logit[rain] 最大化
  方法：固定点迭代（无需反向传播，CPU 上数秒完成）
    残差结构：h_35_out = h_34 + mlp(norm(h_34))
    → h_34 = h_35_out - mlp(norm(h_34))
    → 迭代：h_{t+1} = target - mlp(norm(h_t))，初始 h_0 = target
  目标向量 target 取 lm_head.weight[rain_id]（令 rain logit 最大的隐藏方向）
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_PATH = "D:/杂项/models/Meta-Llama-3-8B-Instruct"
SAVE_PATH  = "data/outputs/neuron_backup/h34_from_rain_backprop.pt"
os.makedirs("data/outputs/neuron_backup", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

device = next(model.parameters()).device
rain_id = tokenizer.encode(" rain", add_special_tokens=False)[-1]
print(f"rain token: {repr(tokenizer.decode([rain_id]))}, id={rain_id}, device={device}")

# 模型结构
layer_35   = model.model.layers[35]
norm_mlp   = layer_35.post_attention_layernorm   # MLP 前的 LayerNorm
mlp_35     = layer_35.mlp
final_norm = model.model.norm
lm_head    = model.lm_head

# 目标向量：令 rain 排在最前的最优隐藏方向
target = lm_head.weight[rain_id].detach().to(device)   # bfloat16, [4096]
print(f"target (lm_head[rain]) 范数 = {target.float().norm():.4f}")

# ── 固定点迭代
print("\n【固定点迭代：h_{t+1} = target - mlp_35(norm(h_t))】")
print(f"{'iter':>5}  {'‖(h+mlp)−target‖':>18}  {'rain_logit':>12}  {'rain_rank':>11}  top3")
print("-" * 78)

h = target.clone()
alpha = 0.3          # 阻尼因子：h_{t+1} = h_t + α*(target - mlp(h_t) - h_t)

best_logit  = -1e9
best_h34    = None

for it in range(30):
    with torch.no_grad():
        mlp_out = mlp_35(norm_mlp(h))
        h_fp    = target - mlp_out              # 无阻尼固定点目标
        h_new   = h + alpha * (h_fp - h)       # 阻尼更新

        # 验证残差：h_new + mlp(norm(h_new)) 与 target 的差
        h35_check = h_new + mlp_35(norm_mlp(h_new))
        residual  = (h35_check - target).float().norm().item()

        logits  = lm_head(final_norm(h_new)).float().squeeze(0)
        rain_lg = logits[rain_id].item()
        rank    = int((logits >= logits[rain_id]).sum().item())
        top3_t  = [repr(tokenizer.decode([i])) for i in torch.topk(logits, 3).indices.tolist()]

        if rain_lg > best_logit:
            best_logit = rain_lg
            best_h34   = h_new.detach().float().cpu().clone()

        if it % 5 == 0 or it < 5:
            print(f"{it:5d}  {residual:18.6f}  {rain_lg:12.4f}  #{rank:<10d}  {top3_t}")
        h = h_new

print(f"\n  最优 rain_logit = {best_logit:.4f}")
h_34_result = best_h34

# ── 保存（最优 h_34）
torch.save(h_34_result, SAVE_PATH)
print(f"\n✓ h_34 已保存 → {SAVE_PATH}")
print(f"  形状={h_34_result.shape}  范数={h_34_result.norm():.4f}")

# ── h_34 在 embed_tokens 空间的余弦近邻
print("\n【h_34 在 embed_tokens 空间的余弦近邻 Top-15】")
embed = model.model.embed_tokens.weight.detach().float().cpu()
h_n   = F.normalize(h_34_result.unsqueeze(0), dim=1)
e_n   = F.normalize(embed, dim=1)

chunk  = 32768
scores = []
for i in range(0, e_n.shape[0], chunk):
    scores.extend((e_n[i:i+chunk] @ h_n.T).squeeze(1).tolist())
scores_t = torch.tensor(scores)

top_v, top_i = torch.topk(scores_t, 15)
for rank, (v, tid) in enumerate(zip(top_v.tolist(), top_i.tolist()), 1):
    marker = " ← rain自身" if tid == rain_id else ""
    print(f"  #{rank:2d}  {repr(tokenizer.decode([tid])):<25} cos={v:.6f}{marker}")

rain_rank_e = int((scores_t >= scores_t[rain_id]).sum().item())
print(f"\n  rain 在 embed 空间排名: #{rain_rank_e}  cos={scores_t[rain_id].item():.6f}")
