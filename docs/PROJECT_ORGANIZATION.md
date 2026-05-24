# LLM Probe 项目整理 - 完整指南

## 📋 整理概述

您的 LLM Probe 项目包含25+个独立脚本散落在根目录和 tools/ 文件夹中。本整理方案将这些脚本按功能分类整理到结构化的目录中。

## 🎯 整理目标

将混乱的文件结构：
```
probe/
├── analyze_dims.py
├── rain_analytic_inv.py
├── run_experiments.py
├── generate_report.py
├── test.py
├── tools/
│   ├── analyze_command_word_layers.py
│   └── ...
└── ... (20+ 其他脚本)
```

整理为清晰的结构：
```
probe/
├── src/              # 核心库
├── scripts/
│   ├── analysis/     # 分析脚本
│   ├── experiments/  # 实验脚本
│   └── utils/        # 工具脚本
├── tests/            # 测试文件
└── main.py           # 入口
```

## ⚡ 快速执行（3种方法）

### 方法A：运行迁移脚本（推荐）

```bash
python run_migration.py
```

这是最完整的脚本，包含详细的日志和错误处理。

### 方法B：运行快速脚本

```bash
python quick_organize.py
```

这是精简版本，功能相同但输出较少。

### 方法C：执行内联脚本

```bash
python -c "exec(open('_organize_exec.py').read())"
```

或在 Python REPL 中：
```python
exec(open('_organize_exec.py').read())
```

## 📊 文件分类说明

### scripts/analysis/ - 分析脚本（17个）
这类脚本用于分析和调查LLM的隐藏层状态：

- `analyze_dims.py` - 维度分析
- `analyze_skip_results.py` - 跳跃结果分析
- `compare_embedding_layer8.py` - 嵌入对比
- `rain_analytic_inv.py` - RAIN反投影分析
- `rain_analytic_inv2.py` - RAIN反投影分析v2
- `rain_backproject.py` - 反向投影
- `rain_backprop_layer35.py` - 第35层反向传播
- `layer_similarity_probe.py` - 层相似性探测
- `layer_skip_probe.py` - 层跳跃探测
- `head_ablation_probe.py` - 头部切除探测
- `show_head_top10.py` - 头部Top10显示
- `query_embedding.py` - 查询嵌入
- `embed_sum_probe.py` - 嵌入和探测
- `apple_desktop_probe.py` - 桌面探测
- `embed_to_logits.py` - 嵌入到logits
- `logits_from_layer8.py` - 第8层的logits
- `inject_layer8_at24.py` - 在第24层注入第8层

### scripts/experiments/ - 实验脚本（3个）
运行系统实验的脚本：

- `run_experiments.py` - 运行主要实验
- `run_shuffled_experiments.py` - 运行打乱的实验
- `save_neuron.py` - 保存神经元数据

### scripts/utils/ - 工具和报告脚本（10个）
生成报告和数据处理的工具：

**来自根目录：**
- `generate_report.py` - 生成报告
- `generate_report_cn.py` - 生成中文报告
- `generate_multi_report.py` - 生成多报告
- `generate_shuffle_report.py` - 生成打乱报告

**来自 tools/ 目录：**
- `analyze_command_word_layers.py` - 命令词层分析
- `analyze_sequence_word_layers.py` - 序列词层分析
- `format_deep_interpretation_docx.py` - DOCX格式化
- `generate_comprehensive_perception_report.py` - 综合感知报告
- `render_command_diff_charts.py` - 命令差异图表渲染
- `render_full_4096_charts.py` - 全4096图表渲染

### tests/ - 测试文件（2个）
环境和功能测试：

- `test.py` - 模型加载和推理测试
- `test1.py` - 环境信息检查（Python、PyTorch、CUDA）

## ✅ 整理前检查清单

在运行迁移脚本前，请确认：

- [ ] 项目根目录是 `c:\AI_Model\probe`
- [ ] Python 3.7+ 已安装
- [ ] 没有其他进程占用根目录中的文件
- [ ] 有足够的磁盘空间
- [ ] 您有完整的备份（可选但推荐）

## ✅ 整理后验证清单

完成整理后，验证以下内容：

- [ ] `scripts/` 目录已创建，包含3个子目录
- [ ] 所有17个分析脚本在 `scripts/analysis/` 中
- [ ] 所有3个实验脚本在 `scripts/experiments/` 中
- [ ] 所有10个工具脚本在 `scripts/utils/` 中
- [ ] 所有2个测试脚本在 `tests/` 中
- [ ] 每个目录都有 `__init__.py` 文件
- [ ] `main.py` 仍在根目录
- [ ] `src/` 目录保持不变

### 验证命令

```bash
# 检查目录结构
ls -R scripts/
ls tests/

# 测试主程序
python main.py --help

# 测试脚本可访问性
python scripts/analysis/analyze_dims.py
python scripts/experiments/run_experiments.py
```

## 🔄 如果迁移失败

如果自动脚本失败，尝试以下步骤：

1. **检查错误信息** - 查看脚本输出的具体错误
2. **手动创建目录** - 使用文件管理器或命令行
3. **分批移动文件** - 手动拖拽或使用 `move` 命令
4. **检查文件权限** - 确保文件不被占用
5. **重启系统** - 如果有文件锁定问题

## 📝 导入路径调整

**好消息：** 大多数脚本使用绝对导入（如 `from src.config import ...`），这些**无需修改**。

**如果有脚本使用相对导入，** 需要在脚本开头添加：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

## 📚 文档清单

本整理过程中创建的文档：

- **REORGANIZATION_PLAN.md** - 详细的整理方案
- **MIGRATION_INSTRUCTIONS.md** - 迁移说明
- **run_migration.py** - 完整迁移脚本
- **quick_organize.py** - 快速迁移脚本
- **_organize_exec.py** - 可导入执行的脚本
- **整理步骤.txt** - 中文步骤说明
- **PROJECT_ORGANIZATION.md** - 本文档

## 🚀 后续步骤

整理完成后，建议：

1. ✅ **更新 README.md** - 说明新的目录结构和如何运行脚本
2. ✅ **创建 .gitignore** - 添加 `__pycache__`, `.pyc` 等
3. ✅ **运行测试** - `python tests/test.py` 确认环境正常
4. ✅ **文档化主流脚本** - 为常用脚本添加使用说明
5. ✅ **考虑创建 setup.py** - 如果要发布为包

## ❓ 常见问题

### Q: 迁移后脚本找不到模块？
A: 确保 `src/` 目录在 Python 路径中。脚本中已有 `sys.path` 调整。

### Q: 可以撤销迁移吗？
A: 如果担心，请先备份项目。迁移过程不删除文件，只是移动。

### Q: 需要更新 main.py 吗？
A: 不需要。main.py 的导入是绝对导入，移动其他脚本不影响它。

### Q: 可以选择性迁移吗？
A: 可以。手动编辑迁移脚本，注释掉不需要迁移的部分。

## 📞 技术支持

如果遇到问题：

1. 检查错误信息 - 通常会指出具体原因
2. 查看日志 - 迁移脚本会打印每个操作的结果
3. 逐个文件手动移动 - 如果自动脚本某个部分失败
4. 检查文件权限 - 确保有读写权限

## 🎉 总结

整理后，您将获得：

✅ **更清晰的结构** - 脚本分类合理，便于维护
✅ **更易于扩展** - 未来添加新脚本时有清晰的位置
✅ **更好的可读性** - 项目一目了然
✅ **便于测试** - 测试文件单独管理
✅ **专业化** - 符合 Python 项目最佳实践

---

**版本:** 1.0  
**日期:** 2024-05-23  
**作者:** LLM Probe 自动整理系统

祝您的项目整理顺利！ 🎊
