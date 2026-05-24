# Probe 参数配置 - 快速参考

## 快速开始

### 修改参数的三种方式

#### 方式 1️⃣: 直接编辑配置文件（推荐）

**文件**: `configs/default.yaml`

```yaml
# 修改目标分析层
analysis:
  target_layer: 12          # 从 8 改为 12

# 修改训练参数
probe:
  linear_probe:
    max_iter: 5000          # 更多迭代
    test_size: 0.3          # 30% 测试集
```

然后直接运行：
```bash
python main.py run-probe --label-file data/labels.csv
```

#### 方式 2️⃣: 创建新配置文件

```bash
# 复制默认配置
cp configs/default.yaml configs/custom-analysis.yaml

# 编辑自定义配置
# ... 修改参数 ...

# 使用自定义配置
python main.py --config configs/custom-analysis.yaml run-probe --label-file data/labels.csv
```

#### 方式 3️⃣: 在 Python 代码中覆盖

```python
from src.config import load_config
from src.probe import train_linear_probe

config = load_config("configs/default.yaml")

# 临时修改配置
config["probe"]["linear_probe"]["max_iter"] = 5000
config["analysis"]["target_layer"] = 12

# 使用修改后的配置
results = train_linear_probe(dataset, config=config)
```

---

## 常用参数速查表

### 模型选择

```yaml
model:
  model_name_or_path: "C:/AI_Model/Llama3_8B_Instruct"
  torch_dtype: "bfloat16"           # 或 float16, float32
  device_map: "auto"                # 自动分配设备
  # 如果内存不足，启用量化：
  load_in_4bit: true
  load_in_8bit: false
```

### 分析层选择

```yaml
analysis:
  target_layer: 8                   # 改为你要分析的层
  # 推荐值：
  # 8-10   - 早期层，捕捉低级特征
  # 16-18  - 中间层，混合特征
  # 24-28  - 晚期层，高级语义
  # 30-31  - 最后层，输出特征
```

### 报告详度控制

```yaml
analysis:
  top_k_dims: 24                    # 突出的维度数（小→快，大→详细）
  top_k_words_per_dim: 8            # 每个维度的示例词数
  top_k_concepts: 8                 # 概念匹配数
  top_k_predictions: 5              # 预测候选数
```

### 训练参数调整

```yaml
probe:
  linear_probe:
    test_size: 0.2                  # 测试集比例
                                    # 0.1  - 多训练少测试
                                    # 0.2  - 推荐（默认）
                                    # 0.3  - 多测试少训练
    max_iter: 3000                  # 迭代上限
                                    # 3000 - 推荐（默认）
                                    # 5000 - 样本多时使用
                                    # 1000 - 快速测试
```

### 性能优化

```yaml
# 快速模式（适合快速测试）
analysis:
  target_layer: 8
  top_k_dims: 12
  
video:
  enabled: false                    # 禁用视频加快处理

visualization:
  figure_dpi: 96                    # 降低分辨率

# 高质量模式（适合最终报告）
analysis:
  top_k_dims: 50
  
visualization:
  figure_dpi: 300                   # 高分辨率
```

---

## 常见使用场景

### 场景 1: 快速原型验证

```yaml
analysis:
  target_layer: 8
  top_k_dims: 12

probe:
  linear_probe:
    max_iter: 1000          # 快速收敛

video:
  enabled: false            # 跳过视频
```

### 场景 2: 完整深度分析

```yaml
analysis:
  target_layer: 16
  top_k_dims: 50

probe:
  linear_probe:
    max_iter: 5000
    test_size: 0.25

video:
  fps: 2
  enabled: true
```

### 场景 3: 属性探针训练

```yaml
probe:
  attribute_probe:
    max_iter: 3000
    test_size_ratio: 0.25
    min_total_samples: 8          # 至少需要 8 个样本
    min_class_count: 2            # 每个类至少 2 个
```

---

## 配置参数说明

### model 段 - 模型加载

| 参数 | 选项 | 说明 |
|------|------|------|
| `model_name_or_path` | 路径字符串 | 模型文件夹位置 |
| `torch_dtype` | bfloat16, float16, float32 | 运算精度 |
| `device_map` | auto, cuda, cpu | 设备分配 |
| `load_in_4bit` | true/false | 4 位量化 |
| `trust_remote_code` | true/false | 是否信任远程代码 |

### analysis 段 - 分析参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| `target_layer` | 0-31 | 8 | 分析的模型层编号 |
| `top_k_dims` | 1-4096 | 24 | 报告中的维度数 |
| `top_k_words_per_dim` | 1-100 | 8 | 每个维度的词数 |
| `top_k_concepts` | 1-50 | 8 | 概念匹配数 |
| `top_k_predictions` | 1-20 | 5 | 预测候选数 |

### probe段 - 训练参数

| 参数 | 类型 | 默认值 | 范围 |
|------|------|--------|------|
| `random_state` | int | 42 | 任意 |
| `max_iter` | int | 3000 | 100-10000 |
| `test_size` | float | 0.2 | 0.1-0.5 |
| `solver` | string | lbfgs | lbfgs, liblinear |
| `min_total_samples` | int | 8 | 2-100 |
| `min_class_count` | int | 2 | 1-10 |

---

## 参数调整示例

### 例 1: 提高模型准确率（但变慢）

```yaml
probe:
  linear_probe:
    max_iter: 5000          # 从 3000 增加到 5000
    test_size: 0.2          # 保持测试集比例

analysis:
  top_k_dims: 50            # 从 24 增加到 50（更详细）
```

### 例 2: 加快处理速度（牺牲精度）

```yaml
analysis:
  target_layer: 8           # 不改深层
  top_k_dims: 12            # 从 24 降到 12

probe:
  linear_probe:
    max_iter: 1000          # 从 3000 降到 1000
    test_size: 0.1          # 从 0.2 降到 0.1（少测试）

video:
  enabled: false            # 跳过视频生成
  fps: 1                    # 如果要视频，降帧率
```

### 例 3: 分析不同层数

```yaml
# 分析第 12 层
analysis:
  target_layer: 12

# 分析第 20 层
analysis:
  target_layer: 20

# 分析第 30 层（输出前）
analysis:
  target_layer: 30
```

---

## 配置验证

### ✅ 有效配置

```yaml
analysis:
  target_layer: 12
  top_k_dims: 32
  
probe:
  linear_probe:
    max_iter: 3000
    test_size: 0.2
```

### ❌ 无效配置（会出错）

```yaml
# 层编号超出范围
analysis:
  target_layer: 99          # ❌ Llama-3-8B 只有 32 层

# 测试集比例不合理
probe:
  linear_probe:
    test_size: 0.99         # ❌ 太大，训练数据不足

# 迭代次数为 0
probe:
  linear_probe:
    max_iter: 0             # ❌ 无法训练
```

---

## 配置文件位置

```
configs/
├── default.yaml             # 主配置文件（推荐修改）
└── quantized-4bit.yaml      # 量化配置示例
```

---

## 快速命令

### 使用特定配置运行

```bash
# 使用默认配置
python main.py run-single-batch

# 使用自定义配置
python main.py --config configs/custom-analysis.yaml run-single-batch

# 运行属性探针
python main.py run-attribute-probe --attribute-file data/word_attributes.csv

# 生成维度报告
python main.py run-dim-report --word-file data/color_words.txt
```

---

## 更多信息

- 详细配置指南: [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)
- 迁移日志: [PARAMETER_MIGRATION_LOG.md](PARAMETER_MIGRATION_LOG.md)
- 项目架构: [ARCHITECTURE.md](ARCHITECTURE.md)

---

**最后更新**: 2026-05-23
