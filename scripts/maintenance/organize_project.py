#!/usr/bin/env python3
"""
项目代码整理脚本
将散落的脚本分类到合适的目录
"""

import os
import shutil
from pathlib import Path

# 定义根目录
PROBE_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROBE_ROOT / "scripts"
SCRIPTS_ANALYSIS = SCRIPTS_DIR / "analysis"
SCRIPTS_EXPERIMENTS = SCRIPTS_DIR / "experiments"
SCRIPTS_UTILS = SCRIPTS_DIR / "utils"
TESTS_DIR = PROBE_ROOT / "tests"

# 创建目录
for d in [SCRIPTS_ANALYSIS, SCRIPTS_EXPERIMENTS, SCRIPTS_UTILS, TESTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {d}")

# 分类规则
ANALYSIS_SCRIPTS = {
    "analyze_dims.py",
    "analyze_skip_results.py",
    "compare_embedding_layer8.py",
    "rain_analytic_inv.py",
    "rain_analytic_inv2.py",
    "rain_backproject.py",
    "rain_backprop_layer35.py",
    "layer_similarity_probe.py",
    "layer_skip_probe.py",
    "head_ablation_probe.py",
    "show_head_top10.py",
    "query_embedding.py",
    "embed_sum_probe.py",
    "apple_desktop_probe.py",
    "embed_to_logits.py",
    "logits_from_layer8.py",
    "inject_layer8_at24.py",
}

EXPERIMENT_SCRIPTS = {
    "run_experiments.py",
    "run_shuffled_experiments.py",
    "save_neuron.py",
}

REPORT_SCRIPTS = {
    "generate_report.py",
    "generate_report_cn.py",
    "generate_multi_report.py",
    "generate_shuffle_report.py",
}

TEST_FILES = {
    "test.py",
    "test1.py",
}

# 移动分析脚本
print("\n📊 Moving analysis scripts...")
for script in ANALYSIS_SCRIPTS:
    src = PROBE_ROOT / script
    if src.exists():
        dst = SCRIPTS_ANALYSIS / script
        shutil.move(str(src), str(dst))
        print(f"  → {script}")

# 移动实验脚本
print("\n🧪 Moving experiment scripts...")
for script in EXPERIMENT_SCRIPTS:
    src = PROBE_ROOT / script
    if src.exists():
        dst = SCRIPTS_EXPERIMENTS / script
        shutil.move(str(src), str(dst))
        print(f"  → {script}")

# 移动报告脚本到utils
print("\n📝 Moving report scripts...")
for script in REPORT_SCRIPTS:
    src = PROBE_ROOT / script
    if src.exists():
        dst = SCRIPTS_UTILS / script
        shutil.move(str(src), str(dst))
        print(f"  → {script}")

# 从tools移动脚本
print("\n🔧 Moving tools scripts...")
TOOLS_DIR = PROBE_ROOT / "tools"
if TOOLS_DIR.exists():
    for script in TOOLS_DIR.glob("*.py"):
        if script.name != "__init__.py":
            dst = SCRIPTS_UTILS / script.name
            shutil.move(str(script), str(dst))
            print(f"  → {script.name}")

# 移动测试文件
print("\n✅ Moving test files...")
for test in TEST_FILES:
    src = PROBE_ROOT / test
    if src.exists():
        dst = TESTS_DIR / test
        shutil.move(str(src), str(dst))
        print(f"  → {test}")

# 创建__init__.py文件
print("\n🎯 Creating __init__.py files...")
for d in [SCRIPTS_DIR, SCRIPTS_ANALYSIS, SCRIPTS_EXPERIMENTS, SCRIPTS_UTILS, TESTS_DIR]:
    init_file = d / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"  → {d.name}/__init__.py")

print("\n✓ 项目整理完成！")
print("\n新目录结构:")
print("""
probe/
├── src/                    # 核心库
├── scripts/
│   ├── analysis/          # 分析脚本
│   ├── experiments/       # 实验脚本
│   └── utils/             # 工具和报告脚本
├── tests/                 # 测试文件
├── configs/               # 配置文件
├── data/                  # 数据目录
├── outputs/               # 输出目录
├── main.py                # 主入口
└── requirements.txt
""")
