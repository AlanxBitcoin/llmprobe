"""
apple vs desktop：单独输入、正常组合输入、embedding 1:1叠加对比
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "D:/杂项/models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
model.eval()

embed_matrix = model.model.embed_tokens.weight
bos_id = tokenizer.bos_token_id

def get_emb(word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    print(f"  '{word}' -> tokens={tokenizer.convert_ids_to_tokens(ids)}  ids={ids}")
    return embed_matrix[ids[-1]].detach().clone(), ids[-1]

def run_emb(emb_tok, label):
    bos = embed_matrix[bos_id].to(model.device).to(model.dtype).reshape(1,1,-1)
    tok = emb_tok.to(model.device).to(model.dtype).reshape(1,1,-1)
    h = torch.cat([bos, tok], dim=1)
    with torch.no_grad():
        logits = model(inputs_embeds=h).logits[0, -1].float()
    show(logits, label)

def run_text(text, label):
    ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([ids]).to(model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1].float()
    show(logits, label)

def show(logits, label):
    top_vals, top_ids = torch.topk(logits, 10)
    print(f"\n【{label}】")
    print("=" * 50)
    for rank, (v, tid) in enumerate(zip(top_vals.tolist(), top_ids.tolist()), 1):
        tok = tokenizer.decode([tid])
        print(f"  #{rank:2d}  {repr(tok):<22}  logit={v:.4f}")

print("Token 信息：")
emb_apple,   id_apple   = get_emb("apple")
emb_desktop, id_desktop = get_emb("desktop")

# 1. apple 单独（BOS + embedding）
run_emb(emb_apple,   "BOS + apple embedding 单独")

# 2. desktop 单独（BOS + embedding）
run_emb(emb_desktop, "BOS + desktop embedding 单独")

# 3. 正常文本："apple desktop"
run_text("apple desktop", "正常文本输入 'apple desktop'")

# 4. embedding 1:1 叠加
run_emb(emb_apple + emb_desktop, "BOS + (apple + desktop) 1:1 embedding叠加")

# 5. (apple/2) + desktop
run_emb(emb_apple * 0.5 + emb_desktop, "BOS + (apple/2 + desktop)")

# 6. (apple*2/3) + desktop
run_emb(emb_apple * (2/3) + emb_desktop, "BOS + (apple*2/3 + desktop)")
