import torch
from transformers import AutoTokenizer
from safetensors import safe_open
from src.config import load_config
import os, glob

cfg = load_config('configs/default.yaml')
model_path = cfg['model']['model_name_or_path']

tok = AutoTokenizer.from_pretrained(model_path)
ids = tok.encode('rain', add_special_tokens=False)
print(f"token ids: {ids}, tokens: {[tok.decode([i]) for i in ids]}")

st_files = sorted(glob.glob(os.path.join(model_path, '*.safetensors')))
emb = None
for f in st_files:
    with safe_open(f, framework='pt', device='cpu') as sf:
        if 'model.embed_tokens.weight' in sf.keys():
            emb = sf.get_tensor('model.embed_tokens.weight')
            print(f"embedding 矩阵 shape: {emb.shape}，来自: {os.path.basename(f)}")
            break

token_id = ids[0]
vec = emb[token_id].float()
print(f"\n'rain' (token_id={token_id}) embedding 统计 (维度={vec.shape[0]}):")
print(f"  最大值:       {vec.max().item():.6f}")
print(f"  最小值:       {vec.min().item():.6f}")
print(f"  绝对值平均值: {vec.abs().mean().item():.6f}")
print(f"  绝对值最大值: {vec.abs().max().item():.6f}")
print(f"  标准差:       {vec.std().item():.6f}")
