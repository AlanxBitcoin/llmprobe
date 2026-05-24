# 项目架构文档

## 概述
这是一个 LLM 隐藏状态探针 (Probe) 项目，用于分析 Llama-3-8B 模型的内部表示。项目通过多种模式（单词模式、批处理模式、序列模式）提取和分析隐藏状态。

---

## 项目结构

```
probe/
├── main.py                          # 主入口，命令行管理器
├── configs/                         # 配置文件
│   ├── default.yaml                # 默认配置
│   └── quantized-4bit.yaml          # 量化配置
├── data/                            # 数据文件
│   ├── concept_catalog.yaml         # 概念目录
│   ├── word_*.txt                   # 各类词表
│   └── word_attributes.csv          # 词属性标注
├── src/                             # 核心代码库
│   ├── __init__.py
│   ├── model_loader.py              # ⭐ 模型加载（TRANSFORMER 调用点 #1）
│   ├── config.py                    # 配置加载
│   ├── extract_hidden.py            # 隐藏状态提取（TRANSFORMER 调用点 #2-4）
│   ├── pipeline.py                  # 管道处理流程
│   ├── probe.py                     # 线性探针训练
│   ├── attribute_probe.py            # 属性探针
│   ├── concept_match.py              # 概念匹配
│   ├── symbolic_attributes.py        # 符号属性
│   ├── utils.py                     # 工具函数
│   └── video.py                     # 视频生成
├── outputs/                         # 输出结果目录
│   ├── shape_words/                 # 形状词分析
│   ├── color_words/                 # 颜色词分析
│   ├── neuron_backup/               # 神经元数据备份
│   └── ...其他分析结果
└── 研究脚本/                        # 一次性研究脚本
    ├── head_ablation_probe.py       # 注意力头消融（TRANSFORMER 调用点）
    ├── apple_desktop_probe.py       # 特定词分析（TRANSFORMER 调用点）
    ├── embed_sum_probe.py           # 嵌入求和（TRANSFORMER 调用点）
    ├── layer_similarity_probe.py    # 层相似度（TRANSFORMER 调用点）
    ├── rain_analytic_inv.py         # 雨字分析反演（TRANSFORMER 调用点）
    ├── rain_analytic_inv2.py        # 雨字分析反演2（TRANSFORMER 调用点）
    ├── rain_backproject.py          # 雨字反投影（TRANSFORMER 调用点）
    ├── rain_backprop_layer35.py     # 雨字反向传播（TRANSFORMER 调用点）
    ├── layer_skip_probe.py          # 层跳过探针（TRANSFORMER 调用点）
    ├── inject_layer8_at24.py        # 第8层注入第24层（TRANSFORMER 调用点）
    ├── logits_from_layer8.py        # 从第8层计算logits（TRANSFORMER 调用点）
    ├── query_embedding.py           # 查询嵌入（TRANSFORMER 调用点）
    ├── generate_report.py           # 生成报告
    └── ...其他分析脚本
```

---

## TRANSFORMER 调用点详细列表

### 1️⃣ **[src/model_loader.py](src/model_loader.py)** - 模型加载核心

**作用**：统一的模型和 tokenizer 加载接口

**关键函数**：
- `load_local_model(config)` - 从配置加载本地模型
  - 导入：`from transformers import AutoModelForCausalLM, AutoTokenizer`
  - 加载 tokenizer：`AutoTokenizer.from_pretrained(tokenizer_path)`
  - 加载模型：`AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)`
  - 支持：量化配置、dtype转换、device_map管理

- `_build_quantization_config()` - 构建量化配置
  - 导入：`from transformers import BitsAndBytesConfig`

**返回**：`LocalModelBundle(tokenizer, model)`

**被调用者**：
- [main.py](main.py) - 主程序
- [src/pipeline.py](src/pipeline.py) - 管道处理
- [layer_skip_probe.py](layer_skip_probe.py)
- [inject_layer8_at24.py](inject_layer8_at24.py)

---

### 2️⃣ **[src/extract_hidden.py](src/extract_hidden.py)** - 隐藏状态提取

**作用**：从模型的各层提取隐藏状态向量

**关键函数**：

#### `extract_single_word_states(bundle, words, target_layer)`
- **调用点**：`model(**inputs_gpu, output_hidden_states=True)`
- **功能**：逐词提取隐藏状态（每个词单独forward pass）
- **返回**：各词的隐藏向量列表

#### `extract_all_words_state(bundle, words, target_layer)`
- **调用点**：`model(**inputs_gpu, output_hidden_states=True)`
- **功能**：所有词拼接后一次forward pass，提取最后词位置的隐藏状态
- **返回**：最后一个词的隐藏向量

#### `extract_sequence_positional_states(bundle, words, target_layer)`
- **调用点**：`model(**full_encoded_gpu, output_hidden_states=True)`
- **功能**：所有词拼接一次forward pass，提取各词自身位置的隐藏状态
- **返回**：各词的隐藏向量列表（按原始位置）

**模型调用模式**：
```python
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # 获取第target_layer层的隐藏状态
    hidden = outputs.hidden_states[target_layer]  # shape: [batch, seq_len, hidden_dim]
```

**被调用者**：
- [src/pipeline.py](src/pipeline.py) - 三种模式的管道调用

---

### 3️⃣ **[src/pipeline.py](src/pipeline.py)** - 核心处理管道

**作用**：协调整个分析流程（调用上述三种提取函数）

**关键类**：`ProbePipeline`

**重要方法**：
- `run_single_batch()` - 单词逐个分析
- `run_multi_batch()` - 小批量分析
- `run_global_analysis()` - 全局分析

**transformer 使用**：
```python
# 调用三种隐藏状态提取函数
per_word_results = extract_single_word_states(self.bundle, words, target_layer)
all_input_result = extract_all_words_state(self.bundle, words, target_layer)
pos_results = extract_sequence_positional_states(self.bundle, words, target_layer)
```

---

### 4️⃣ **研究脚本 - 各类单一目的分析** 

这些脚本直接导入 transformer 库，进行特定的分析：

#### [head_ablation_probe.py](head_ablation_probe.py)
- **导入**：`from transformers import AutoTokenizer, AutoModelForCausalLM`
- **调用**：直接加载和推理模型
- **目的**：分析注意力头的重要性

#### [apple_desktop_probe.py](apple_desktop_probe.py)
- **调用**：`model(inputs_embeds=h)` - 使用嵌入输入
- **目的**：分析特定词的嵌入和逻辑输出

#### [embed_sum_probe.py](embed_sum_probe.py)
- **调用**：`model()` forward pass
- **目的**：测试嵌入相加的效果

#### [layer_similarity_probe.py](layer_similarity_probe.py)
- **调用**：`model()` with `output_hidden_states=True`
- **目的**：分析各层隐藏状态相似度

#### [rain_analytic_inv.py](rain_analytic_inv.py) / [rain_analytic_inv2.py](rain_analytic_inv2.py)
- **调用**：访问 `model.model.layers[35]`、`model.lm_head` 等
- **目的**：反演分析第35层对"rain"的贡献

#### [rain_backproject.py](rain_backproject.py)
- **调用**：`model.model.embed_tokens.weight`、`model.lm_head.weight`
- **目的**：反投影分析

#### [rain_backprop_layer35.py](rain_backprop_layer35.py)
- **调用**：访问模型层结构进行反向传播分析
- **目的**：反向传播分析第35层

#### [layer_skip_probe.py](layer_skip_probe.py)
- **导入**：`from src.model_loader import load_local_model`
- **调用**：`model(**input_ids)` 和 `model(inputs_embeds=...)` 替代方案
- **目的**：跳过某些层的影响分析

#### [inject_layer8_at24.py](inject_layer8_at24.py)
- **导入**：`from src.model_loader import load_local_model`
- **调用**：使用钩子注入隐藏状态，修改模型行为
- **目的**：注入分析

#### [logits_from_layer8.py](logits_from_layer8.py)
- **导入**：`AutoTokenizer.from_pretrained()`
- **调用**：直接使用 lm_head 权重计算 logits
- **目的**：从第8层隐藏状态计算输出分布

#### [query_embedding.py](query_embedding.py)
- **导入**：`from transformers import AutoTokenizer`
- **调用**：仅使用 tokenizer，从 safetensors 加载嵌入
- **目的**：查询特定词的嵌入

---

## 调用流程图

```
main.py
  └─→ load_config()
  └─→ load_local_model(cfg)  ⭐ [model_loader.py]
      └─→ AutoTokenizer.from_pretrained()
      └─→ AutoModelForCausalLM.from_pretrained()
  
  └─→ ProbePipeline(bundle)  ⭐ [pipeline.py]
      └─→ extract_single_word_states()      ⭐ [extract_hidden.py]
          └─→ model(**inputs, output_hidden_states=True)
      
      └─→ extract_all_words_state()         ⭐ [extract_hidden.py]
          └─→ model(**inputs, output_hidden_states=True)
      
      └─→ extract_sequence_positional_states()  ⭐ [extract_hidden.py]
          └─→ model(**inputs, output_hidden_states=True)
```

---

## 关键 API 使用汇总

### Tokenizer API
```python
# 加载
tokenizer = AutoTokenizer.from_pretrained(path)

# 编码
token_ids = tokenizer.encode(word, add_special_tokens=False)
encoded = tokenizer(text, return_tensors="pt")

# 解码
token_str = tokenizer.decode([token_id])
tokens = tokenizer.convert_ids_to_tokens(ids)

# 特殊 token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token_id
```

### Model API
```python
# 加载
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=...,  # 可选
)

# 模式
model.eval()

# Forward pass
outputs = model(
    input_ids=input_ids,           # 标准输入
    # 或
    inputs_embeds=embeddings,       # 嵌入输入
    output_hidden_states=True,      # 获取所有层隐藏状态
)

# 输出访问
logits = outputs.logits              # [batch, seq, vocab]
hidden = outputs.hidden_states[i]    # 第i层隐藏状态
```

### 模型结构访问
```python
# 嵌入层
embed_matrix = model.model.embed_tokens.weight  # [vocab, hidden_dim]

# 各层
layer_i = model.model.layers[i]
layer_i.self_attn         # 自注意力
layer_i.mlp               # MLP
layer_i.post_attention_layernorm
layer_i.input_layernorm

# 输出层
final_norm = model.model.norm
lm_head = model.lm_head                # [hidden_dim, vocab]
```

---

## 配置管理

### 配置文件位置
- [configs/default.yaml](configs/default.yaml) - 默认配置
- [configs/quantized-4bit.yaml](configs/quantized-4bit.yaml) - 4bit 量化

### 关键配置项
```yaml
model:
  model_name_or_path: "path/to/model"
  tokenizer_name_or_path: optional
  torch_dtype: "float16" | "bfloat16" | "float32"
  device_map: "auto" | None
  load_in_4bit: bool
  load_in_8bit: bool
  trust_remote_code: bool
```

---

## 数据流

### 单词分析流程
1. 从 CSV 或 TXT 加载词表
2. 对每个词：
   - tokenize（可能多个 sub-token）
   - forward pass 获取隐藏状态
   - 提取指定层的隐藏向量
   - 激活值分析
   - 维度排序统计
3. 聚合统计，生成报告

### 隐藏状态提取的三种模式
| 模式 | 输入 | 特点 | 场景 |
|------|------|------|------|
| **单词模式** | 逐个词 | 无上下文 | 基准分析 |
| **全输入模式** | 所有词拼接 | 仅有左上下文 | 完整上下文 |
| **位置模式** | 所有词拼接 | 按各词位置提取 | 序列模型分析 |

---

## 常见模式

### 模型初始化和推理
```python
from src.model_loader import load_local_model
from src.config import load_config

cfg = load_config("configs/default.yaml")
bundle = load_local_model(cfg)
model = bundle.model
tok = bundle.tokenizer

# 推理
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx]
```

### 层访问
```python
# 访问第 i 层
layer = model.model.layers[i]

# 访问 MLPs
mlp = layer.mlp
ffn_out = mlp(input)

# 访问注意力头
attn = layer.self_attn
```

### 钩子注入（高级）
```python
def hook_fn(module, inp, output):
    # 修改输出或捕获中间值
    return modified_output

# 注册钩子
handle = layer.register_forward_hook(hook_fn)
# 使用...
handle.remove()  # 清理
```

---

## 依赖关系

### 核心依赖
- `transformers` - 模型和 tokenizer 加载
- `torch` - 张量操作
- `peft` - LoRA 适配（如果使用）
- `bitsandbytes` - 量化（可选）
- `accelerate` - 分布式/设备管理

### 文件依赖树
```
main.py
├── config.py
├── model_loader.py
│   └── transformers (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
├── pipeline.py
│   ├── extract_hidden.py
│   │   └── model_loader.py
│   ├── probe.py
│   └── attribute_probe.py
└── ...
```

---

## 调试和开发提示

### 获取模型信息
```python
print(f"层数: {len(model.model.layers)}")
print(f"隐藏维度: {model.config.hidden_size}")
print(f"词表大小: {model.config.vocab_size}")
print(f"注意力头数: {model.config.num_attention_heads}")
```

### 设备和dtype处理
```python
# 检查可用设备
device = next(model.parameters()).device

# 强制类型转换
h = hidden_state.float()  # -> float32
h = hidden_state.to(torch.bfloat16)

# 设备移动
h = h.to(device)
```

### 梯度管理
```python
# 全局关闭梯度
torch.set_grad_enabled(False)

# 局部关闭
with torch.no_grad():
    outputs = model(...)
```

---

## 文件清单（按功能分类）

### ⚙️ 框架文件
- [main.py](main.py) - 命令行入口
- [src/config.py](src/config.py) - 配置加载
- [src/model_loader.py](src/model_loader.py) - 模型加载

### 🔍 分析核心
- [src/extract_hidden.py](src/extract_hidden.py) - 隐藏状态提取
- [src/pipeline.py](src/pipeline.py) - 管道处理
- [src/probe.py](src/probe.py) - 线性探针

### 📊 可视化和报告
- [src/visualize_single_word.py](src/visualize_single_word.py)
- [src/visualize_multi_word.py](src/visualize_multi_word.py)
- [generate_report.py](generate_report.py)

### 🧪 研究脚本
- `head_ablation_probe.py` - 注意力头消融
- `rain_*.py` - "rain"词的各类分析
- `layer_*.py` - 层级分析
- 等...

### 📁 数据文件
- [data/word_*.txt](data/) - 词表
- [data/concept_catalog.yaml](data/) - 概念
- [data/word_attributes.csv](data/) - 标注数据

---

## 更新日志

本文档记录了项目的整体架构和所有 transformer 调用点，方便快速定位和理解代码流程。

**最后更新**：2026-05-23
