# LLaMA Model Hooks 使用指南

本文档说明 `hooks.py` 中所有直接操作 LLM 的功能。

## 1. 提取隐状态 (Hidden States Extraction)

### 读取某个层的隐状态
```python
from hooks import extract_layer_hidden_states

# 提取第5层的隐状态
states = extract_layer_hidden_states(model, layer_indices=5, input_ids=input_ids)
# 返回: {5: {'input': tensor(...), 'output': tensor(...)}}
```

### 读取多个层的隐状态
```python
# 提取第0, 3, 5层的隐状态
states = extract_layer_hidden_states(model, layer_indices=[0, 3, 5], input_ids=input_ids)
# 返回: {0: {...}, 3: {...}, 5: {...}}
```

### 读取所有层的隐状态
```python
# 提取所有层的隐状态
states = extract_layer_hidden_states(model, layer_indices=None, input_ids=input_ids)
# 返回: {0: {...}, 1: {...}, ..., n: {...}}
```

## 2. 跳过层 (Skip Layers)

### 跳过某一层
```python
from hooks import skip_layers

# 跳过第3层（直接传递隐状态）
output = skip_layers(model, layer_indices=3, input_ids=input_ids)
```

### 跳过多层
```python
# 跳过第2, 4, 6层
output = skip_layers(model, layer_indices=[2, 4, 6], input_ids=input_ids)
```

## 3. 关闭注意力头 (Disable Attention Heads)

### 关闭某个头在全部层
```python
from hooks import disable_attention_heads

# 关闭所有层的第0个头
disabler = disable_attention_heads(model, head_indices=0)
# 执行前向传播后，移除钩子
disabler.remove_all_hooks()
```

### 关闭多个头在全部层
```python
# 关闭所有层的第1, 3个头
disabler = disable_attention_heads(model, head_indices=[1, 3])
```

### 关闭特定层的头
```python
# 关闭第2, 5层的第0个头
disabler = disable_attention_heads(model, head_indices=0, layer_indices=[2, 5])
```

### 关闭多个头在多个层
```python
# 关闭第2, 4层的第1, 3, 5个头
disabler = disable_attention_heads(model, head_indices=[1, 3, 5], layer_indices=[2, 4])
```

### 重新启用头
```python
# 重新启用第2层的第0个头
disabler.enable_heads(head_indices=0, layer_indices=2)
```

## 4. 提取FFN神经元隐状态 (Extract FFN Neuron Hidden States)

### 提取所有FFN神经元
```python
from hooks import extract_ffn_neuron_hidden_states

# 提取第5层的所有FFN神经元
states = extract_ffn_neuron_hidden_states(model, layer_indices=5, input_ids=input_ids)
# 返回: {5: {'all_neurons': tensor(...)}}
```

### 提取特定FFN神经元
```python
# 提取第5层的第100-200个神经元
neuron_indices = list(range(100, 200))
states = extract_ffn_neuron_hidden_states(model, layer_indices=5, 
                                         neuron_indices=neuron_indices, 
                                         input_ids=input_ids)
# 返回: {5: {'neurons': tensor(...)}}
```

### 提取全部层的特定神经元
```python
# 提取所有层的第50-100个神经元
states = extract_ffn_neuron_hidden_states(model, layer_indices=None,
                                         neuron_indices=list(range(50, 100)),
                                         input_ids=input_ids)
# 返回: {0: {...}, 1: {...}, ..., n: {...}}
```

## 5. 提取神经元参数 (Extract Neuron Parameters)

### 提取某层的所有权重
```python
from hooks import extract_neuron_parameters

# 提取第5层的所有权重
params = extract_neuron_parameters(model, layer_idx=5, param_type='weight')
# 返回: {'attn_q_proj': tensor(...), 'attn_k_proj': tensor(...), 'mlp_down_proj': tensor(...), ...}
```

### 提取某层的偏置
```python
# 提取第3层的所有偏置
params = extract_neuron_parameters(model, layer_idx=3, param_type='bias')
```

## 6. 比较参数变化 (Compare Parameter Changes)

### 对比操作前后的参数
```python
from hooks import compare_neuron_parameters

def before_operation():
    # 操作前的处理
    pass

def after_operation():
    # 操作后的处理
    pass

# 比较第5层参数的变化
result = compare_neuron_parameters(model, layer_idx=5,
                                  before_fn=before_operation,
                                  after_fn=after_operation)

# 获取变化
weight_change = result['after']['mlp_down_proj'] - result['before']['mlp_down_proj']
```

## 7. 完整示例

```python
import torch
from hooks import (
    extract_layer_hidden_states,
    extract_ffn_neuron_hidden_states,
    disable_attention_heads,
    extract_neuron_parameters,
    skip_layers
)

# 初始化
model = load_model()  # 加载模型
input_ids = tokenizer("Hello world")['input_ids']

# 示例1: 提取隐状态并分析
print("=" * 50)
print("提取隐状态")
print("=" * 50)
hidden_states = extract_layer_hidden_states(model, layer_indices=[0, 5, 10], input_ids=input_ids)
for layer_idx, states in hidden_states.items():
    print(f"Layer {layer_idx}: {states['output'].shape}")

# 示例2: 关闭特定的注意力头
print("\n" + "=" * 50)
print("关闭注意力头")
print("=" * 50)
disabler = disable_attention_heads(model, head_indices=[0, 1], layer_indices=[5, 10])
# 执行推理...
disabler.remove_all_hooks()

# 示例3: 提取FFN神经元
print("\n" + "=" * 50)
print("提取FFN神经元")
print("=" * 50)
ffn_states = extract_ffn_neuron_hidden_states(
    model, 
    layer_indices=[5],
    neuron_indices=list(range(100, 150)),
    input_ids=input_ids
)
for layer_idx, states in ffn_states.items():
    print(f"Layer {layer_idx}: {states['neurons'].shape}")

# 示例4: 提取并比较参数
print("\n" + "=" * 50)
print("提取参数")
print("=" * 50)
params_before = extract_neuron_parameters(model, layer_idx=5, param_type='weight')
print(f"权重形状: {params_before['mlp_down_proj'].shape}")

# 示例5: 跳过某些层
print("\n" + "=" * 50)
print("跳过层")
print("=" * 50)
output = skip_layers(model, layer_indices=[3, 4], input_ids=input_ids)
```

## 注意事项

1. **内存管理**: 大模型提取隐状态时可能占用大量内存，建议在 `torch.no_grad()` 下执行
2. **钩子清理**: 使用完 `AttentionHeadDisabler` 和其他钩子管理器后，记得调用 `remove_all_hooks()`
3. **GPU/CPU**: 确保输入张量和模型在同一设备上
4. **Batch大小**: 对于大批量数据，建议分批处理
5. **参数类型**: 支持 'weight' 和 'bias'，但不是所有层都有偏置

## API 速查表

| 函数/类 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `extract_layer_hidden_states` | 提取层隐状态 | 层索引、input_ids | Dict[层, 隐状态] |
| `skip_layers` | 跳过层 | 层索引、input_ids | 输出张量 |
| `disable_attention_heads` | 关闭注意力头 | 头索引、层索引 | AttentionHeadDisabler |
| `extract_ffn_neuron_hidden_states` | 提取FFN隐状态 | 层索引、神经元索引 | Dict[层, FFN隐状态] |
| `extract_neuron_parameters` | 提取参数 | 层索引、参数类型 | Dict[参数名, 张量] |
| `compare_neuron_parameters` | 比较参数变化 | 层索引、操作函数 | Dict[before/after, 参数] |
