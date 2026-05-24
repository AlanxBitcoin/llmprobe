# 配置参数指南

## 概述

所有硬编码的参数都已迁移到 `configs/default.yaml` 配置文件中，便于快速调整分析参数，无需修改代码。

---

## 配置文件位置

- **主配置**: `configs/default.yaml`
- **量化配置**: `configs/quantized-4bit.yaml`

---

## 配置参数详细说明

### 模型加载参数 (`model`)

```yaml
model:
  model_name_or_path: "C:/AI_Model/Llama3_8B_Instruct"  # 模型路径
  tokenizer_name_or_path: null                          # tokenizer 路径（null 则用模型路径）
  device_map: "auto"                                    # 设备分配策略
  torch_dtype: "bfloat16"                               # 精度：float16, bfloat16, float32
  load_in_4bit: false                                   # 4bit 量化
  load_in_8bit: false                                   # 8bit 量化
  trust_remote_code: true                               # 是否信任远程代码
```

### 分析参数 (`analysis`)

```yaml
analysis:
  target_layer: 8                      # 目标分析层（0-35）
  top_k_dims: 24                       # 维度报告中选择的维度数
  top_k_concepts: 8                    # 概念匹配中的 top-k
  max_multi_word_group: 3              # 多词组合的最大词数
  aggregation: "mean_abs"              # 聚合方式：mean_abs, max_abs 等
  top_k_words_per_dim: 8               # 维度报告中每个维度的词数
  top_k_predictions: 5                 # 属性预测的候选数
  top_k_concepts_per_axis: 5           # 维度解释中的概念数
```

#### 参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `target_layer` | 8 | 0-35 | 需要分析的模型层编号，8-12 层通常效果好 |
| `top_k_dims` | 24 | 1-4096 | 报告中突出的维度数 |
| `top_k_concepts` | 8 | 1-50 | 概念匹配时返回的匹配数 |
| `top_k_words_per_dim` | 8 | 1-100 | 维度报告中每个维度显示的正向/负向词数 |
| `top_k_predictions` | 5 | 1-20 | 属性预测时返回的候选类别数 |
| `top_k_concepts_per_axis` | 5 | 1-20 | 维度解释时的概念数 |

### 输入输出参数 (`input`, `output`)

```yaml
input:
  word_file: "data/word_list_100.txt"  # 默认词表文件
  attribute_file: "data/word_attributes.csv"  # 属性标注文件

output:
  base_dir: "data/outputs"             # 输出根目录（v1.8: 从项目根 outputs/ 迁移到 data/outputs/）
  save_json: true                      # 保存 JSON 中间结果
  save_plots: true                     # 保存可视化图表
```

### 视频参数 (`video`)

```yaml
video:
  enabled: true                        # 是否生成视频
  fps: 2                               # 视频帧率（每秒帧数）
  codec: "libx264"                     # 视频编码器
```

### 可视化参数 (`visualization`)

```yaml
visualization:
  figure_dpi: 160                      # 图表分辨率（DPI）
  cmap: "magma"                        # matplotlib 颜色映射
```

---

## 机器学习训练参数 (`probe`)

### 线性探针参数 (`probe.linear_probe`)

```yaml
probe:
  linear_probe:
    random_state: 42                   # 随机种子
    test_size: 0.2                     # 测试集比例（20%）
    solver: "lbfgs"                    # sklearn LogisticRegression 求解器
    max_iter: 3000                     # 最大迭代次数
```

#### 影响分析

| 参数 | 说明 | 影响 |
|------|------|------|
| `random_state` | 随机种子 | 确保可复现性 |
| `test_size` | 测试集比例 | 较小→更多训练数据，可能过拟合 |
| `solver` | 优化算法 | lbfgs（推荐）, liblinear, sag 等 |
| `max_iter` | 迭代上限 | 过低会导致不收敛 |

### 属性探针参数 (`probe.attribute_probe`)

```yaml
probe:
  attribute_probe:
    random_state: 42
    solver: "lbfgs"
    max_iter: 3000
    test_size_ratio: 0.25              # 测试集比例（25%）
    min_class_count: 2                 # 每个类最少样本数
    min_total_samples: 8               # 属性最少总样本数
    min_unique_classes: 2              # 属性最少类别数
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `test_size_ratio` | 0.25 | 测试集比例 |
| `min_class_count` | 2 | 排除样本太少的类别 |
| `min_total_samples` | 8 | 排除总样本太少的属性 |
| `min_unique_classes` | 2 | 至少需要 2 个类别才能训练 |

---

## 预处理和报告参数

### 预处理参数 (`preprocess`)

```yaml
preprocess:
  min_label_count_attr_probe: 2        # 标签出现的最小次数
```

### 报告参数 (`report`)

```yaml
report:
  top_positive_words: 8                # 维度报告中的正向词数
  top_negative_words: 8                # 维度报告中的负向词数
  largest_lists_limit: 12              # 全局分析中的最大列表数
  review_candidates_limit: 12          # 审查候选的限制
```

---

## 常见配置调整

### 快速分析（少量词表）

```yaml
analysis:
  target_layer: 8
  top_k_dims: 12              # 减少维度数
  top_k_concepts: 5           # 减少概念数

video:
  fps: 1                      # 减慢视频帧率以加快处理
```

### 深度分析（完整研究）

```yaml
analysis:
  target_layer: 12            # 更深层
  top_k_dims: 50              # 增加维度数
  top_k_concepts: 15          # 更多概念

probe:
  linear_probe:
    max_iter: 5000            # 更多迭代
```

### 量化模型配置

参考 `configs/quantized-4bit.yaml`，配合：

```yaml
model:
  load_in_4bit: true
  torch_dtype: "float16"
  device_map: "auto"
```

---

## 参数修改流程

### 1. 修改配置文件

编辑 `configs/default.yaml`，调整需要的参数：

```yaml
analysis:
  target_layer: 12     # 从 8 改为 12
  top_k_dims: 40       # 从 24 改为 40
```

### 2. 运行分析

代码会自动读取配置：

```bash
python main.py --config configs/default.yaml run-single-batch
```

### 3. 查看结果

结果会保存到 `outputs/` 目录，参数值会记录在 JSON 文件中。

---

## 代码调用示例

### Python 中使用配置

```python
from src.config import load_config
from src.model_loader import load_local_model
from src.probe import train_linear_probe, build_probe_dataset

# 加载配置
config = load_config("configs/default.yaml")

# 加载模型
bundle = load_local_model(config)

# 训练探针（自动使用配置参数）
dataset = build_probe_dataset(bundle, label_rows,
                               int(config["analysis"]["target_layer"]))
results = train_linear_probe(dataset, config=config)
```

### 覆盖配置参数

```python
from copy import deepcopy

config = load_config("configs/default.yaml")
# 临时修改
config["analysis"]["target_layer"] = 15
config["probe"]["linear_probe"]["max_iter"] = 5000

results = train_linear_probe(dataset, config=config)
```

---

## 配置参数来源

| 功能 | 配置来源 | 代码位置 |
|------|---------|--------|
| 模型加载 | `model.*` | `src/model_loader.py` |
| 隐藏状态提取 | `analysis.target_layer` | `src/extract_hidden.py` |
| 维度报告 | `analysis.top_k_*` | `src/pipeline.py::run_dimension_report` |
| 线性探针训练 | `probe.linear_probe.*` | `src/probe.py::train_linear_probe` |
| 属性探针训练 | `probe.attribute_probe.*` | `src/attribute_probe.py::train_attribute_probes` |
| 可视化 | `visualization.*`, `video.*` | `src/visualize_*.py` |

---

## 常见问题

### Q: 如何只改变某个分析的参数？

**A**: 在命令行调用前，读取配置、修改参数再创建管道：

```python
config = load_config("configs/default.yaml")
config["analysis"]["target_layer"] = 12
pipeline = ProbePipeline(bundle, config)
```

### Q: 模型层数和 target_layer 有什么关系？

**A**: Llama-3-8B 共有 32 层（0-31，有些版本 36 层）。建议：
- `target_layer: 8-12` - 早期层，捕捉低级特征
- `target_layer: 16-24` - 中间层，混合特征
- `target_layer: 28-31` - 晚期层，高级语义

### Q: top_k 参数的作用？

**A**:
- `top_k_dims` - 报告中突出显示的维度数
- `top_k_words_per_dim` - 每个维度显示多少个支持词
- `top_k_concepts` - 概念匹配时的候选数
- `top_k_predictions` - 属性预测时显示的候选类别数

增加 top_k 会增加计算量但提供更多信息。

---

## 最后更新

**时间**: 2026-05-23
**覆盖范围**: 所有主要 probe 分析模块的参数化
