import torch
torch.set_grad_enabled(False)
from transformers import AutoTokenizer, AutoModelForCausalLM

# 替换为你的模型存放路径
model_path = r"C:\AI_Model\Llama3_8B_Instruct"

# 加载模型与分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    device_map="auto",
    #torch_dtype=torch.float16,
    local_files_only=True,
    low_cpu_mem_usage=True
)

# 简单提问测试
prompt = "简单介绍一下人工智能"
inputs = tokenizer(prompt, return_tensors="pt")

# 生成回复
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7
)

# 打印结果
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型回复：")
print(result)

