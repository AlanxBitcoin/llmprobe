# -*- coding: utf-8 -*-
"""
查看 Llama-3-8B-Instruct 第一层隐藏层结构，
选取 MLP gate_proj 第 0 号神经元，保存其参数到文件。
"""
import json
import torch
from pathlib import Path
from src.model_loader import load_local_model
from src.config import load_config

cfg = load_config("configs/default.yaml")
bundle = load_local_model(cfg)
model = bundle.model

# ── 打印第0层结构概览 ──────────────────────────────────────────
layer0 = model.model.layers[0]
print("=== Layer 0 子模块 ===")
for name, module in layer0.named_children():
    print(f"  {name}: {module.__class__.__name__}")

mlp = layer0.mlp
print("\n=== Layer 0 MLP 参数形状 ===")
for name, param in mlp.named_parameters():
    print(f"  {name}: {param.shape}  dtype={param.dtype}")

# ── 选定神经元：MLP gate_proj 第 42 号神经元 ──────────────────
# Llama MLP: gate_proj (hidden->intermediate), up_proj, down_proj
# 每个神经元对应 gate_proj 的一行 + up_proj 的一行 + down_proj 的一列
NEURON_IDX = 42
print(f"\n=== 选定神经元: layer=0, sub=mlp, neuron_idx={NEURON_IDX} ===")

gate_row  = mlp.gate_proj.weight[NEURON_IDX].detach().float().cpu()   # shape: [hidden_size]
up_row    = mlp.up_proj.weight[NEURON_IDX].detach().float().cpu()     # shape: [hidden_size]
down_col  = mlp.down_proj.weight[:, NEURON_IDX].detach().float().cpu()# shape: [hidden_size]

print(f"  gate_proj row {NEURON_IDX}: shape={gate_row.shape}, mean={gate_row.mean():.6f}, std={gate_row.std():.6f}")
print(f"  up_proj   row {NEURON_IDX}: shape={up_row.shape},   mean={up_row.mean():.6f}, std={up_row.std():.6f}")
print(f"  down_proj col {NEURON_IDX}: shape={down_col.shape}, mean={down_col.mean():.6f}, std={down_col.std():.6f}")

# ── 保存到文件 ──────────────────────────────────────────────────
out_dir = Path("data/outputs/neuron_backup")
out_dir.mkdir(parents=True, exist_ok=True)

# 保存为 PyTorch 格式（精确）
torch.save({
    "layer": 0,
    "sub":   "mlp",
    "neuron_idx": NEURON_IDX,
    "gate_proj_row": gate_row,
    "up_proj_row":   up_row,
    "down_proj_col": down_col,
}, out_dir / "layer0_mlp_neuron42.pt")

# 同时保存元信息为 JSON（便于阅读）
meta = {
    "model": cfg["model"]["model_name_or_path"],
    "layer": 0,
    "sub": "mlp",
    "neuron_idx": NEURON_IDX,
    "gate_proj_row_shape": list(gate_row.shape),
    "gate_proj_row_mean":  float(gate_row.mean()),
    "gate_proj_row_std":   float(gate_row.std()),
    "gate_proj_row_min":   float(gate_row.min()),
    "gate_proj_row_max":   float(gate_row.max()),
    "up_proj_row_shape":   list(up_row.shape),
    "up_proj_row_mean":    float(up_row.mean()),
    "up_proj_row_std":     float(up_row.std()),
    "down_proj_col_shape": list(down_col.shape),
    "down_proj_col_mean":  float(down_col.mean()),
    "down_proj_col_std":   float(down_col.std()),
    "description": (
        "Llama-3-8B-Instruct 第0层 MLP 第42号神经元的完整参数。"
        "gate_proj_row: 控制门控激活的权重向量；"
        "up_proj_row: 上投影权重向量；"
        "down_proj_col: 下投影权重向量（该神经元对输出的贡献）。"
    )
}
(out_dir / "layer0_mlp_neuron42_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(f"\n已保存:")
print(f"  {out_dir}/layer0_mlp_neuron42.pt      (PyTorch tensor, 精确参数)")
print(f"  {out_dir}/layer0_mlp_neuron42_meta.json (元信息)")
print(f"\n选定的神经元: 第0层 MLP gate_proj/up_proj 第 {NEURON_IDX} 号")
