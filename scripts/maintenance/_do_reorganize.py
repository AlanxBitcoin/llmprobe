"""
自动整理项目结构的脚本
将散落的Python脚本分类到合适的目录
"""

import os
import shutil
import sys
from pathlib import Path

def create_directories():
    """创建所需的目录结构"""
    base = Path(__file__).parent
    dirs = [
        base / "scripts",
        base / "scripts" / "analysis",
        base / "scripts" / "experiments", 
        base / "scripts" / "utils",
        base / "tests"
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory created: {d.name}")
    return base

def move_files(base):
    """按分类移动文件"""
    
    # 分析脚本
    analysis_files = [
        "analyze_dims.py", "analyze_skip_results.py", "compare_embedding_layer8.py",
        "rain_analytic_inv.py", "rain_analytic_inv2.py", "rain_backproject.py",
        "rain_backprop_layer35.py", "layer_similarity_probe.py", "layer_skip_probe.py",
        "head_ablation_probe.py", "show_head_top10.py", "query_embedding.py",
        "embed_sum_probe.py", "apple_desktop_probe.py", "embed_to_logits.py",
        "logits_from_layer8.py", "inject_layer8_at24.py"
    ]
    
    # 实验脚本
    experiment_files = ["run_experiments.py", "run_shuffled_experiments.py", "save_neuron.py"]
    
    # 工具和报告脚本
    utils_files = [
        "generate_report.py", "generate_report_cn.py", "generate_multi_report.py",
        "generate_shuffle_report.py"
    ]
    
    # 测试文件
    test_files = ["test.py", "test1.py"]
    
    # 执行移动
    moved_count = 0
    
    print("\n📊 Moving analysis scripts...")
    for f in analysis_files:
        src = base / f
        if src.exists():
            dst = base / "scripts" / "analysis" / f
            try:
                shutil.move(str(src), str(dst))
                print(f"  ✓ {f}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ {f}: {e}")
    
    print("\n🧪 Moving experiment scripts...")
    for f in experiment_files:
        src = base / f
        if src.exists():
            dst = base / "scripts" / "experiments" / f
            try:
                shutil.move(str(src), str(dst))
                print(f"  ✓ {f}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ {f}: {e}")
    
    print("\n📝 Moving utility scripts...")
    for f in utils_files:
        src = base / f
        if src.exists():
            dst = base / "scripts" / "utils" / f
            try:
                shutil.move(str(src), str(dst))
                print(f"  ✓ {f}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ {f}: {e}")
    
    # 从tools目录移动脚本
    tools_dir = base / "tools"
    if tools_dir.exists():
        for py_file in tools_dir.glob("*.py"):
            if py_file.name != "__init__.py":
                dst = base / "scripts" / "utils" / py_file.name
                try:
                    shutil.move(str(py_file), str(dst))
                    print(f"  ✓ {py_file.name}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ {py_file.name}: {e}")
    
    print("\n✅ Moving test files...")
    for f in test_files:
        src = base / f
        if src.exists():
            dst = base / "tests" / f
            try:
                shutil.move(str(src), str(dst))
                print(f"  ✓ {f}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ {f}: {e}")
    
    return moved_count

def create_init_files(base):
    """创建__init__.py文件"""
    dirs = [
        base / "scripts",
        base / "scripts" / "analysis",
        base / "scripts" / "experiments",
        base / "scripts" / "utils",
        base / "tests"
    ]
    
    print("\n🎯 Creating __init__.py files...")
    for d in dirs:
        init_file = d / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"  ✓ {d.name}/__init__.py")

def main():
    """主程序"""
    print("=" * 60)
    print("  LLM Probe 项目整理脚本")
    print("=" * 60)
    
    base = create_directories()
    moved = move_files(base)
    create_init_files(base)
    
    print("\n" + "=" * 60)
    print(f"✓ 整理完成！共移动 {moved} 个文件")
    print("=" * 60)
    
    print("\n新的项目结构:")
    print("""
probe/
├── src/                    # 核心库代码
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

if __name__ == "__main__":
    main()
