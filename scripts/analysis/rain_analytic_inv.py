"""
rain 倒推 - 解析逆推法（无梯度）
目标：从 lm_head[rain] 出发，解析反推第34层隐藏状态

正向 MLP（layer 35）:
  h_in → RMSNorm(h_in)*γ → gate_proj/up_proj → silu(gate)*up → down_proj → mlp_out
  h_out = h_in + mlp_out

反向步骤（"除以系数" + "反查找激活"）:
  mlp_out → inv(down_proj) → inv(silu gate) → inv(gate_proj) → h_in_est

迭代误差修正直到 ||error|| / ||target|| < 1‰
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
norm_mlp   = layer_35.post_attention_layernorm   # MLP 前的 RMSNorm
mlp_35     = layer_35.mlp
final_norm = model.model.norm
lm_head    = model.lm_head

# 取出权重（float32, CPU）
W_down = mlp_35.down_proj.weight.detach().float().cpu()  # [4096, 14336]
W_gate = mlp_35.gate_proj.weight.detach().float().cpu()  # [14336, 4096]
W_up   = mlp_35.up_proj.weight.detach().float().cpu()    # [14336, 4096]
norm_w = norm_mlp.weight.detach().float().cpu()          # [4096] RMSNorm gamma

print(f"W_down {W_down.shape}  W_gate {W_gate.shape}  W_up {W_up.shape}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 三个解析逆算子
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def silu_inv(y: torch.Tensor, n_iter: int = 40) -> torch.Tensor:
    """SiLU 逆：给定 y = x*sigmoid(x), 求 x（Newton 迭代逐元素）
    相当于"反查找表"
    """
    x = y.clone()
    for _ in range(n_iter):
        sig = torch.sigmoid(x)
        fx  = x * sig                          # silu(x)
        dfx = sig + x * sig * (1.0 - sig)     # silu'(x)
        x   = x - (fx - y) / dfx.clamp(min=1e-8)
    return x

def linear_inv(y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """线性层伪逆：x ≈ W^T y / ||W_col||²
    forward: y = W @ x   (W: [out, in])
    inverse: x ≈ W^T @ y / col_sq    ("除以系数")
    """
    col_sq = (W ** 2).sum(dim=0).clamp(min=1e-8)   # [in_dim]
    return (W.T @ y) / col_sq

def rms_norm_inv(y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """RMSNorm 近似逆：x ≈ y / gamma * estimated_rms
    forward: y = (x / rms(x)) * gamma
    我们用 rms(y/gamma) 估计输出范数，再外推回 x 的量级
    """
    x_unnorm = y / gamma.clamp(min=1e-4)
    # 目标：让 rms(x) 与 y 的信息量匹配，直接返回 x_unnorm（迭代会修正量级）
    return x_unnorm

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 解析反向 MLP（"部署一层反向计算"）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backward_mlp(mlp_out: torch.Tensor) -> torch.Tensor:
    """给定 mlp_out，解析反推 pre-RMSNorm 状态，再逆 RMSNorm 得 h_in

    Forward chain:
      h_in → RMSNorm*γ → [gate, up] → silu(gate)*up → down_proj → mlp_out

    Backward chain（解析）:
      mlp_out
        → inv(down_proj)    [线性逆: 除以 col 范数²]
        → activated_est [14336]
        → inv(silu) on gate path
        → gate_est [14336]
        → inv(gate_proj)   [线性逆]  +  inv(up_proj)  [线性逆]
        → h_normed_est [4096]  (pre-norm 空间)
        → inv(RMSNorm)
        → h_in_est [4096]
    """
    # Step 1: 逆 down_proj（"除以系数"）
    activated_est = linear_inv(mlp_out, W_down)          # [14336]

    # Step 2: 逆 SwiGLU gate 路径（"反查找激活"）
    #   silu(gate) * up = activated
    #   近似：把 activated 同时作为 silu(gate) 和 up 两条路径的信号
    #   gate_path: gate = silu_inv(activated)
    #   up_path:   up   = activated  （up 路径线性，不过激活函数）
    gate_est = silu_inv(activated_est)                    # [14336]

    # Step 3: 逆 gate_proj 和 up_proj（各自"除以系数"，取平均）
    h_from_gate = linear_inv(gate_est, W_gate)            # [4096]
    h_from_up   = linear_inv(activated_est, W_up)         # [4096]
    h_normed    = (h_from_gate + h_from_up) / 2.0        # [4096]

    # Step 4: 逆 RMSNorm（近似）
    h_in_est    = rms_norm_inv(h_normed, norm_w)          # [4096]
    return h_in_est

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 正向 MLP（只用第35层，不含残差）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def forward_mlp(h_in: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        h_bf = h_in.to(device).to(torch.bfloat16)
        out  = mlp_35(norm_mlp(h_bf))
        return out.float().cpu()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 目标向量 & 初始估计
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

target = lm_head.weight[rain_id].detach().float().cpu()  # [4096]
print(f"\ntarget (lm_head[rain]) 范数 = {target.norm():.4f}")

# 初始估计：从 target 出发（因残差结构，h_in ≈ target 是合理起点）
# 但 target 范数很小（~0.89），实际 h_35_out 范数应该更大
# 用 backward_mlp 给出更好的初始方向
init_delta = backward_mlp(target)
init_delta_norm = init_delta.norm()
print(f"backward_mlp(target) 范数 = {init_delta_norm:.4f}")

# 如果初始 delta 太大，裁剪到合理范围
MAX_INIT_NORM = 10.0
if init_delta_norm > MAX_INIT_NORM:
    init_delta = init_delta / init_delta_norm * MAX_INIT_NORM
    print(f"  → 裁剪到范数 = {MAX_INIT_NORM}")

h_in = target.clone() + init_delta

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 迭代误差修正
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n【迭代误差修正（正向 → 误差 → 反向修正 → 循环）】")
print(f"  收敛条件: rel_error < 1‰")
print(f"{'iter':>5}  {'rel_error':>12}  {'rain_logit':>12}  {'rain_rank':>10}  top3")
print("-" * 70)

threshold = 1e-3
LR = 0.3          # 阻尼因子：控制每步修正量，防止发散

def eval_rain(h_out: torch.Tensor):
    with torch.no_grad():
        lg = lm_head(final_norm(h_out.to(device).to(torch.bfloat16))).float().cpu().squeeze(0)
        rain_lg = lg[rain_id].item()
        rank    = int((lg >= lg[rain_id]).sum().item())
        top3    = [repr(tokenizer.decode([i])) for i in torch.topk(lg, 3).indices.tolist()]
    return rain_lg, rank, top3

for it in range(200):
    # 正向：h_in → h_out
    mlp_out  = forward_mlp(h_in)
    h_out    = h_in + mlp_out

    # 误差
    error    = target - h_out
    rel_err  = error.norm() / target.norm()

    # rain 排名
    rain_lg, rank, top3 = eval_rain(h_out)

    print(f"{it:5d}  {rel_err.item():12.6f}  {rain_lg:12.4f}  #{rank:<9d}  {top3}")

    if rel_err < threshold:
        print(f"\n  ✓ 收敛！共 {it+1} 步")
        break

    # 反向修正："把误差反算回来修正第34层的值"
    delta  = backward_mlp(error)
    # 防止 delta 爆炸
    if delta.norm() > 10 * error.norm():
        delta = delta / delta.norm() * error.norm()
    h_in   = h_in + LR * delta

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 保存结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
torch.save(h_in.float(), SAVE_PATH)
print(f"\n✓ h_34 已保存 → {SAVE_PATH}")
print(f"  范数 = {h_in.norm():.4f}")

# embed_tokens 空间近邻
print("\n【h_34 在 embed_tokens 空间的余弦近邻 Top-15】")
embed  = model.model.embed_tokens.weight.detach().float().cpu()
h_n    = F.normalize(h_in.unsqueeze(0), dim=1)
e_n    = F.normalize(embed, dim=1)

chunk  = 32768
scores = []
for i in range(0, e_n.shape[0], chunk):
    scores.extend((e_n[i:i+chunk] @ h_n.T).squeeze(1).tolist())
scores_t = torch.tensor(scores)
top_v, top_i = torch.topk(scores_t, 15)
for r, (v, tid) in enumerate(zip(top_v.tolist(), top_i.tolist()), 1):
    marker = " ← rain" if tid == rain_id else ""
    print(f"  #{r:2d}  {repr(tokenizer.decode([tid])):<25} cos={v:.6f}{marker}")

rain_rk = int((scores_t >= scores_t[rain_id]).sum().item())
print(f"\n  rain 在 embed 空间排名: #{rain_rk}  cos={scores_t[rain_id].item():.6f}")
