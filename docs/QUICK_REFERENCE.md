# 项目整理 - 快速参考卡片

## 🚀 一分钟快速开始

### 执行整理
```bash
python run_migration.py
```

完成！项目已整理。

---

## 📊 整理范围一览

| 类别 | 数量 | 目标位置 |
|------|------|---------|
| 分析脚本 | 17 | scripts/analysis/ |
| 实验脚本 | 3 | scripts/experiments/ |
| 工具脚本 | 10 | scripts/utils/ |
| 测试文件 | 2 | tests/ |
| **总计** | **32** | - |

---

## 📁 新目录结构

```
probe/
├── src/              # 核心库（不变）
├── scripts/          # 📁 新增
│   ├── analysis/     (17个分析脚本)
│   ├── experiments/  (3个实验脚本)
│   └── utils/        (10个工具脚本)
├── tests/            # 📁 新增（2个测试）
├── configs/          # 配置（不变）
├── data/             # 数据（不变）
├── outputs/          # 输出（不变）
└── main.py           # 入口（不变）
```

---

## 📚 文档快速链接

| 文档 | 适用场景 |
|------|---------|
| **PROJECT_ORGANIZATION.md** | 完整指南（推荐首读） |
| **CLEANUP_SUMMARY.md** | 整理总结和检查清单 |
| **MIGRATION_INSTRUCTIONS.md** | 手动执行步骤 |
| **REORGANIZATION_PLAN.md** | 整理策略和方案 |
| **整理步骤.txt** | 中文步骤说明 |

---

## ⚡ 三种执行方法

### 方法1：完整脚本（推荐）
```bash
python run_migration.py
```
特点：详细日志、错误处理完善

### 方法2：快速脚本
```bash
python quick_organize.py
```
特点：代码精简、执行迅速

### 方法3：内联执行
```bash
python -c "exec(open('_organize_exec.py').read())"
```
特点：最小化、无额外文件

---

## ✅ 验证整理成功

### 快速验证命令
```bash
# 检查目录结构
dir scripts\
dir tests\

# 测试主程序
python main.py --help

# 随机测试一个脚本
python scripts\analysis\analyze_dims.py
```

### 完整验证清单
- [ ] scripts/analysis/ 包含 17 个 .py 文件
- [ ] scripts/experiments/ 包含 3 个 .py 文件
- [ ] scripts/utils/ 包含 10 个 .py 文件
- [ ] tests/ 包含 2 个 .py 文件
- [ ] 每个目录都有 __init__.py
- [ ] main.py 仍在根目录
- [ ] main.py 可正常运行

---

## ❓ 常见问题快速解答

| 问题 | 答案 |
|------|------|
| 会删除文件吗？ | 不会，只是移动 |
| 需要改代码吗？ | 几乎不需要，导入已处理 |
| 能撤销吗？ | 可以，文件完整保留 |
| 需要多长时间？ | < 10 秒 |
| 有备份吗？ | 整理脚本不删除任何内容 |

---

## 🔍 如果整理失败

1. **检查错误信息** - 脚本会清楚地说明哪里出错
2. **手动执行** - 参考 MIGRATION_INSTRUCTIONS.md
3. **分批移动** - 一次只移动一个脚本
4. **检查权限** - 确保文件未被占用

---

## 📝 脚本分类说明

### 分析脚本 (scripts/analysis/)
用于分析LLM隐藏层的脚本
```
analyze_dims.py
rain_analytic_inv.py
layer_similarity_probe.py
...等
```

### 实验脚本 (scripts/experiments/)
运行系统级实验的脚本
```
run_experiments.py
run_shuffled_experiments.py
save_neuron.py
```

### 工具脚本 (scripts/utils/)
生成报告和处理数据的脚本
```
generate_report.py
render_full_4096_charts.py
format_deep_interpretation_docx.py
...等
```

### 测试文件 (tests/)
环境和功能测试
```
test.py          # 模型加载测试
test1.py         # 环境检查
```

---

## 🎯 下一步

整理完成后的推荐步骤：

1. ✅ **运行迁移脚本** - 完成主要整理
2. ✅ **验证结构** - 确认所有文件位置正确
3. ✅ **测试脚本** - 运行几个脚本确认可用
4. ✅ **更新文档** - 更新 README 和使用说明
5. ✅ **提交代码** - 如果使用Git，提交整理结果

---

## 💾 包含的工具文件

- **run_migration.py** (10KB) - 完整迁移脚本
- **quick_organize.py** (1.5KB) - 快速脚本  
- **_organize_exec.py** (3KB) - 可导入脚本
- **PROJECT_ORGANIZATION.md** - 完整指南
- **CLEANUP_SUMMARY.md** - 总结报告
- **MIGRATION_INSTRUCTIONS.md** - 手动步骤
- **REORGANIZATION_PLAN.md** - 整理方案
- **整理步骤.txt** - 中文说明

---

## 📞 需要帮助？

1. 查看 PROJECT_ORGANIZATION.md（最详细）
2. 查看 MIGRATION_INSTRUCTIONS.md（手动步骤）
3. 查看脚本输出的错误信息（通常很清楚）

---

**准备好了吗？运行 `python run_migration.py` 开始整理！** 🚀
