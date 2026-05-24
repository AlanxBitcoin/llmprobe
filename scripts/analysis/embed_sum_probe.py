"""
把 rain 和 train 的 embedding 直接相加，作为单个 token 输入跑大模型
对比：单独 rain / 单独 train / 相加后
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "D:/杂项/models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
model.eval()

# 获取 embedding 矩阵
embed_matrix = model.model.embed_tokens.weight  # (vocab, hidden)

id_rain  = tokenizer.encode("rain",  add_special_tokens=False)[0]
id_train = tokenizer.encode("train", add_special_tokens=False)[0]
print(f"rain  token_id={id_rain}")
print(f"train token_id={id_train}")

emb_rain  = embed_matrix[id_rain].detach().clone()   # (4096,)
emb_train = embed_matrix[id_train].detach().clone()  # (4096,)
emb_bos   = embed_matrix[tokenizer.bos_token_id].detach().clone()
emb_sum   = emb_rain + emb_train                     # 直接相加

print(f"bos   token_id={tokenizer.bos_token_id}")

def top10(logits_1d, label):
    vals, ids = torch.topk(logits_1d.float(), 10)
    print(f"\n【{label}】")
    print("="*48)
    for rank, (v, tid) in enumerate(zip(vals.tolist(), ids.tolist()), 1):
        tok = tokenizer.decode([tid])
        print(f"  #{rank:2d}  {repr(tok):<22}  logit={v:.4f}")

def run(emb_token, label):
    # 输入序列: [BOS, token_emb]，取最后一个位置的 logits
    bos = emb_bos.to(model.device).to(model.dtype).reshape(1, 1, -1)
    tok = emb_token.to(model.device).to(model.dtype).reshape(1, 1, -1)
    h = torch.cat([bos, tok], dim=1)  # (1, 2, 4096)
    with torch.no_grad():
        out = model(inputs_embeds=h).logits[0, -1]
    top10(out, label)

run(emb_rain,        "BOS + rain embedding")
run(emb_train,       "BOS + train embedding")
run(emb_sum,         "BOS + (rain + train) 相加")
run(emb_sum / 2,     "BOS + (rain + train) / 2 平均")
