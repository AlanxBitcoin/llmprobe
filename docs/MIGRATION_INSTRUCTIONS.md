# 项目整理迁移说明

## 概述

本项目需要进行代码整理，将散落的脚本文件按功能分类整理到新的目录结构中。

## 快速执行

在项目根目录运行以下命令：

```bash
python run_migration.py
```

这个脚本会自动完成所有的文件迁移和整理工作。

## 手动迁移步骤

如果自动脚本失败，可按以下步骤手动执行：

### 1. 创建目录结构

```bash
mkdir scripts\analysis
mkdir scripts\experiments
mkdir scripts\utils
mkdir tests
```

或者用Python：
```python
from pathlib import Path
for d in ["scripts", "scripts/analysis", "scripts/experiments", "scripts/utils", "tests"]:
    Path(d).mkdir(parents=True, exist_ok=True)
```

### 2. 创建 __init__.py 文件

在以下目录创建空的 `__init__.py` 文件：
- scripts/__init__.py
- scripts/analysis/__init__.py
- scripts/experiments/__init__.py
- scripts/utils/__init__.py
- tests/__init__.py

### 3. 移动分析脚本到 scripts/analysis/

以下文件需要移动到 `scripts/analysis/`：

```
analyze_dims.py
analyze_skip_results.py
apple_desktop_probe.py
compare_embedding_layer8.py
embed_sum_probe.py
embed_to_logits.py
head_ablation_probe.py
inject_layer8_at24.py
layer_similarity_probe.py
layer_skip_probe.py
logits_from_layer8.py
query_embedding.py
rain_analytic_inv.py
rain_analytic_inv2.py
rain_backproject.py
rain_backprop_layer35.py
show_head_top10.py
```

批量移动命令：
```bash
# Windows
move analyze_dims.py scripts\analysis\
move analyze_skip_results.py scripts\analysis\
# ... 继续移动其他文件
```

或用PowerShell：
```powershell
$files = "analyze_dims.py", "analyze_skip_results.py", ...
$files | ForEach-Object { Move-Item $_ "scripts\analysis\" }
```

### 4. 移动实验脚本到 scripts/experiments/

以下文件需要移动到 `scripts/experiments/`：

```
run_experiments.py
run_shuffled_experiments.py
save_neuron.py
```

### 5. 移动工具脚本到 scripts/utils/

以下文件需要移动到 `scripts/utils/`：

**来自根目录：**
```
generate_report.py
generate_report_cn.py
generate_multi_report.py
generate_shuffle_report.py
```

**来自 tools/ 目录：**
```
tools/analyze_command_word_layers.py
tools/analyze_sequence_word_layers.py
tools/format_deep_interpretation_docx.py
tools/generate_comprehensive_perception_report.py
tools/render_command_diff_charts.py
tools/render_full_4096_charts.py
```

### 6. 移动测试文件到 tests/

```
test.py
test1.py
```

### 7. 清理

删除空的 `tools/` 目录（如果其中所有文件都已移动）

## 导入路径调整

大多数脚本使用绝对导入（如 `from src.config import ...`），这些不需要改变。

如果有脚本使用相对导入，需要调整如下：

**原始**（在根目录）：
```python
from src.config import load_config
import sys
sys.path.insert(0, '.')
```

**调整后**（在 scripts/analysis/ 等）：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config
```

## 验证

完成迁移后，验证以下内容：

1. ✅ 所有文件已移动到正确的位置
2. ✅ main.py 仍在根目录
3. ✅ src/ 目录保持不变  
4. ✅ configs/, data/, outputs/ 保持不变
5. ✅ 所有 __init__.py 文件已创建
6. ✅ 脚本仍能正常运行

测试命令：
```bash
python main.py --help
python scripts/analysis/analyze_dims.py
```

## 预期结果

完成迁移后，项目结构应如下所示：

```
probe/
├── src/                          # 核心库 ✅ 保持不变
│   ├── __init__.py
│   ├── config.py
│   ├── model_loader.py
│   └── ... (其他模块)
│
├── scripts/                      # 📁 新增
│   ├── __init__.py
│   ├── analysis/                 # 分析脚本
│   │   ├── __init__.py
│   │   ├── analyze_dims.py
│   │   ├── analyze_skip_results.py
│   │   └── ... (其他分析脚本)
│   │
│   ├── experiments/              # 实验脚本
│   │   ├── __init__.py
│   │   ├── run_experiments.py
│   │   └── ...
│   │
│   └── utils/                    # 工具脚本
│       ├── __init__.py
│       ├── generate_report.py
│       └── ...
│
├── tests/                        # 📁 新增
│   ├── __init__.py
│   ├── test.py
│   └── test1.py
│
├── configs/                      # ✅ 保持不变
├── data/                         # ✅ 保持不变
├── outputs/                      # ✅ 保持不变
├── main.py                       # ✅ 保持在根目录
├── requirements.txt              # ✅ 保持不变
└── README.md                     # ✅ 保持不变
```

## 常见问题

### 问题：脚本在移动后找不到导入的模块

**解决方案**：确保 `sys.path` 包含项目根目录。在脚本开头添加：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### 问题：相对路径（如 'data/file.txt'）失效

**解决方案**：使用绝对路径或相对于项目根目录的路径：
```python
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent.parent / "data"
```

### 问题：import src 找不到

**解决方案**：确保 `src/` 是一个包，包含 `__init__.py`（已确认存在）。

## 需要帮助？

如果自动迁移脚本失败，请：
1. 检查错误消息
2. 手动按照上述步骤进行迁移
3. 使用 `run_migration.py` 脚本作为参考
