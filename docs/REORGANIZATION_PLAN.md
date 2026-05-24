# 项目代码整理方案

## 整理概述
本方案将散落在根目录和 `tools/` 目录的脚本文件进行分类整理，建立清晰的项目结构。

## 新目录结构
```
probe/
├── src/                          # 核心库模块
│   ├── __init__.py
│   ├── config.py
│   ├── model_loader.py
│   ├── extract_hidden.py
│   ├── pipeline.py
│   ├── probe.py
│   ├── attribute_probe.py
│   ├── concept_match.py
│   ├── symbolic_attributes.py
│   ├── utils.py
│   ├── video.py
│   ├── visualize_single_word.py
│   ├── visualize_multi_word.py
│   └── visualize_color_experiment.py
│
├── scripts/                      # 脚本和工具
│   ├── __init__.py
│   ├── analysis/                 # 分析脚本（layer, embedding, etc.）
│   │   ├── __init__.py
│   │   ├── analyze_dims.py
│   │   ├── analyze_skip_results.py
│   │   ├── compare_embedding_layer8.py
│   │   ├── rain_analytic_inv.py
│   │   ├── rain_analytic_inv2.py
│   │   ├── rain_backproject.py
│   │   ├── rain_backprop_layer35.py
│   │   ├── layer_similarity_probe.py
│   │   ├── layer_skip_probe.py
│   │   ├── head_ablation_probe.py
│   │   ├── show_head_top10.py
│   │   ├── query_embedding.py
│   │   ├── embed_sum_probe.py
│   │   ├── apple_desktop_probe.py
│   │   ├── embed_to_logits.py
│   │   ├── logits_from_layer8.py
│   │   └── inject_layer8_at24.py
│   │
│   ├── experiments/              # 实验脚本
│   │   ├── __init__.py
│   │   ├── run_experiments.py
│   │   ├── run_shuffled_experiments.py
│   │   └── save_neuron.py
│   │
│   └── utils/                    # 工具和报告生成脚本
│       ├── __init__.py
│       ├── generate_report.py
│       ├── generate_report_cn.py
│       ├── generate_multi_report.py
│       ├── generate_shuffle_report.py
│       ├── render_command_diff_charts.py
│       ├── render_full_4096_charts.py
│       ├── generate_comprehensive_perception_report.py
│       ├── format_deep_interpretation_docx.py
│       ├── analyze_command_word_layers.py
│       └── analyze_sequence_word_layers.py
│
├── tests/                        # 测试文件
│   ├── __init__.py
│   ├── test.py                   # 模型加载测试
│   └── test1.py                  # 环境检查测试
│
├── configs/                      # 配置文件（保持不变）
│   ├── default.yaml
│   └── quantized-4bit.yaml
│
├── data/                         # 数据目录（保持不变）
│
├── outputs/                      # 输出目录（保持不变）
│
├── main.py                       # 主入口脚本（保持在根目录）
├── requirements.txt              # 依赖声明（保持不变）
└── README.md                     # 项目文档（保持不变）
```

## 文件分类规则

### 分析脚本 (scripts/analysis/)
- `analyze_*.py` - 分析脚本
- `compare_*.py` - 对比脚本
- `rain_*.py` - 反投影和反向传播脚本
- `layer_*.py` - 层级分析脚本
- `head_*.py` - 头部分析脚本
- `*_probe.py` - 探测脚本（除了主要的）
- 其他如 `query_embedding.py`, `show_head_top10.py` 等

### 实验脚本 (scripts/experiments/)
- `run_*.py` - 实验运行脚本
- `save_neuron.py` - 神经元保存脚本

### 工具脚本 (scripts/utils/)
- `generate_*.py` - 报告生成脚本
- `render_*.py` - 渲染脚本
- `analyze_command_*.py` - 命令分析脚本
- `analyze_sequence_*.py` - 序列分析脚本
- `format_*.py` - 格式化脚本

### 测试文件 (tests/)
- `test.py` - 模型加载和推理测试
- `test1.py` - 环境和依赖检查

## 更新导入路径

### 脚本中的导入需要调整
当脚本从根目录移动到 `scripts/` 子目录时，相对导入需要调整：

**原始** (在根目录):
```python
from src.config import load_config
```

**调整后** (在 scripts/analysis/ 等):
```python
from src.config import load_config  # 由于使用了绝对导入，无需改变
```

或者使用相对导入：
```python
from ...src.config import load_config
```

**推荐方法：** 保持 `sys.path` 调整，在脚本开头添加：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

## 迁移清单

- [x] 创建新的目录结构
- [ ] 移动分析脚本到 scripts/analysis/
- [ ] 移动实验脚本到 scripts/experiments/
- [ ] 移动报告和工具脚本到 scripts/utils/
- [ ] 移动测试文件到 tests/
- [ ] 更新脚本中的导入路径
- [ ] 验证所有脚本仍可运行
- [ ] 更新 README.md 中的使用说明
- [ ] 删除空的 tools/ 目录

## 益处
1. ✅ 清晰的项目结构，易于维护
2. ✅ 脚本分类合理，便于查找
3. ✅ 测试文件单独管理
4. ✅ 核心库与使用脚本分离
5. ✅ 便于未来的单元测试扩展
