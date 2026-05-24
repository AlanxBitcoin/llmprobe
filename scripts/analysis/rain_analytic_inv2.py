"""
rain 倒推 - 解析反向传播法（无 autograd）
原理：
  正向保存 MLP 中间激活值，反向用 Jacobian-vector product 解析计算梯度
  等价于 backprop，但全手写，不依赖 PyTorch autograd

  Loss = ||h_in + mlp(norm(h_in)) - target||^2
  迭代更新 h_in 直到 rel_error < 1‰
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_PATH = "D:/杂项/models/Meta-Llama-3-8B-Instruct"
SAVE_PATH  = "data/outputs/neuron_backup/h34_analytic_inv.pt"
os.makedirs("data/outputs/neuron_backup", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

device = next(model.parameters()).device
rain_id = tokenizer.encode(" rain", add_special_tokens=False)[-1]
print(f"rain token: {repr(tokenizer.decode([rain_id]))}, id={rain_id}")

layer_35   = model.model.layers[35]
norm_mlp   = layer_35.post_attention_layernorm
mlp_35     = layer_35.mlp
final_norm = model.model.norm
lm_head    = model.lm_head

# 权重（float32, CPU）
W_down = mlp_35.down_proj.weight.detach().float().cpu()   # [4096, 14336]
W_gate = mlp_35.gate_proj.weight.detach().float().cpu()   # [14336, 4096]
W_up   = mlp_35.up_proj.weight.detach().float().cpu()     # [14336, 4096]
norm_g = norm_mlp.weight.detach().float().cpu()           # [4096] RMSNorm gamma

print(f"W_down {W_down.shape}, W_gate {W_gate.shape}, W_up {W_up.shape}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 正向 MLP（保存中间激活，用于解析反向）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mlp_forward(h_in: torch.Tensor):
    """layer 35 MLP 前向（float32, CPU，保存中间值）
    返回: (mlp_out, cache) 其中 cache 包含反向所需的中间激活
    """
    # RMSNorm: y = x / rms(x) * gamma
    rms     = h_in.pow(2).mean().sqrt().clamp(min=1e-8)
    h_norm  = (h_in / rms) * norm_g                      # [4096]

    # gate & up projections
    gate    = W_gate @ h_norm                             # [14336]
    up      = W_up   @ h_norm                             # [14336]

    # SiLU gate
    sig     = torch.sigmoid(gate)
    gate_a  = gate * sig                                  # silu(gate) [14336]

    # SwiGLU
    activated = gate_a * up                               # [14336]

    # down projection
    mlp_out = W_down @ activated                          # [4096]

    cache = dict(rms=rms, h_norm=h_norm, gate=gate, up=up,
                 sig=sig, gate_a=gate_a, activated=activated)
    return mlp_out, cache

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 解析反向 MLP（Jacobian-vector product）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mlp_backward(v: torch.Tensor, cache: dict, h_in: torch.Tensor) -> torch.Tensor:
    """
    给定 v（损失对 mlp_out 的梯度方向），用保存的中间值反向计算 grad_h_in

    对于最小化 Loss = ||h_in + mlp_out - target||^2：
      grad_mlp_out = 2*(h_in + mlp_out - target) = -2*error
      但我们传入 v = error（正方向），所以最终梯度用 -v
    """
    rms = cache['rms']; h_norm = cache['h_norm']
    gate = cache['gate']; up = cache['up']
    sig = cache['sig']; gate_a = cache['gate_a']; activated = cache['activated']

    g_mlp = v                                             # [4096]  梯度w.r.t. mlp_out

    # ── backward: down_proj  (y = W_down @ activated)
    g_activated = W_down.T @ g_mlp                        # [14336]

    # ── backward: SwiGLU  (activated = silu(gate) * up)
    #   d/d(gate): g_activated * up * silu'(gate)
    #   d/d(up):   g_activated * gate_a
    silu_prime  = sig + gate * sig * (1 - sig)            # silu'(gate) [14336]
    g_gate      = g_activated * up * silu_prime           # [14336]
    g_up        = g_activated * gate_a                    # [14336]

    # ── backward: gate_proj & up_proj  (gate = W_gate @ h_norm)
    g_h_norm    = W_gate.T @ g_gate + W_up.T @ g_up      # [4096]

    # ── backward: RMSNorm  (h_norm = h_in / rms * gamma)
    #   Jacobian-vector product:
    #   grad_h = (gamma / rms) * (g_h_norm - h_norm * dot(g_h_norm, h_norm/rms^2 * gamma) / n)
    #   simplified (drop the quadratic correction term for stability):
    n       = h_in.shape[0]
    dot_val = (g_h_norm * h_norm).sum() / (n * rms**2 + 1e-8)
    g_h_in  = (1.0 / rms) * (norm_g * g_h_norm - h_in * dot_val)   # [4096]

    return g_h_in

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 目标向量 & 迭代
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

target = lm_head.weight[rain_id].detach().float().cpu()
print(f"\ntarget (lm_head[rain]) norm = {target.norm():.4f}")

# 用上次固定点迭代的最优结果作为起点（rain #1 logit=34.25, norm=3.04）
backprop_path = "data/outputs/neuron_backup/h34_from_rain_backprop.pt"
if os.path.exists(backprop_path):
    h_in = torch.load(backprop_path, weights_only=True).float().cpu().squeeze()
    print(f"loaded h34_backprop: norm = {h_in.norm():.4f}")
else:
    h_in = target.clone()
    print("h34_backprop not found, starting from target")

LR = 0.05   # learning rate

def eval_rain(h_out: torch.Tensor):
    with torch.no_grad():
        lg   = lm_head(final_norm(h_out.to(device).to(torch.bfloat16))).float().cpu().squeeze()
        rl   = lg[rain_id].item()
        rank = int((lg >= lg[rain_id]).sum().item())
        top3 = [repr(tokenizer.decode([i])).encode('ascii','replace').decode('ascii')
                for i in torch.topk(lg, 3).indices.tolist()]
    return rl, rank, top3

print("\n[Analytic Backprop Iteration]")
print(f"  convergence: step_size/||h|| < 1e-3  OR  rain_logit stable")
print(f"{'iter':>5}  {'rel_err':>10}  {'step/h':>10}  {'rain_logit':>12}  {'rank':>6}  top3")
print("-" * 80)

best_logit = -1e9
best_h     = h_in.clone()

for it in range(2000):
    # forward
    mlp_out, cache = mlp_forward(h_in)
    h_out   = h_in + mlp_out

    # error
    error   = target - h_out
    rel_err = error.norm() / target.norm()

    # analytic backward: grad of ||h_out - target||^2 w.r.t. h_in
    g = -error - mlp_backward(error, cache, h_in)   # = J_f^T @ (h_out-target)
    delta = LR * g
    step_ratio = delta.norm() / (h_in.norm() + 1e-8)

    # eval rain
    rain_lg, rank, top3 = eval_rain(h_out)
    if rain_lg > best_logit:
        best_logit = rain_lg
        best_h = h_in.clone()

    if it % 50 == 0:
        print(f"{it:5d}  {rel_err.item():10.4f}  {step_ratio.item():10.6f}  {rain_lg:12.4f}  #{rank:<5d}  {top3}")

    # convergence: step becomes tiny
    if step_ratio < 1e-3:
        print(f"\n  Converged (step/h < 1e-3) at iter {it}")
        break

    h_in = h_in - delta

print(f"\n  Best rain logit = {best_logit:.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 保存
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
torch.save(best_h.float(), SAVE_PATH)
print(f"  Saved h34 -> {SAVE_PATH}   norm={best_h.norm():.4f}")

# ── embed 空间近邻
print("\n[h_34 cosine neighbors in embed_tokens space, Top-15]")
embed  = model.model.embed_tokens.weight.detach().float().cpu()
h_n    = F.normalize(best_h.unsqueeze(0), dim=1)
e_n    = F.normalize(embed, dim=1)

chunk  = 32768
scores = []
for i in range(0, e_n.shape[0], chunk):
    scores.extend((e_n[i:i+chunk] @ h_n.T).squeeze(1).tolist())
scores_t = torch.tensor(scores)
top_v, top_i = torch.topk(scores_t, 15)
for r, (v, tid) in enumerate(zip(top_v.tolist(), top_i.tolist()), 1):
    marker = " <- rain" if tid == rain_id else ""
    tok = repr(tokenizer.decode([tid])).encode('ascii','replace').decode('ascii')
    print(f"  #{r:2d}  {tok:<25} cos={v:.6f}{marker}")

rain_rk = int((scores_t >= scores_t[rain_id]).sum().item())
print(f"\n  rain in embed space: rank #{rain_rk}  cos={scores_t[rain_id].item():.6f}")
