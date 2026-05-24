"""
快速整理脚本 - 可通过 import 执行

使用方法：
  python -c "exec(open('_organize_exec.py').read())"

或在Python REPL中：
  exec(open('_organize_exec.py').read())
"""

import os
import shutil
from pathlib import Path

def main():
    os.chdir(Path(__file__).parent)
    
    print("🚀 启动项目整理...\n")
    
    # 创建目录
    print("📁 创建目录结构...")
    dirs = ["scripts", "scripts\\analysis", "scripts\\experiments", "scripts\\utils", "tests"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")
    
    # 创建__init__.py
    print("\n🎯 创建 __init__.py...")
    for d in dirs:
        init = Path(d) / "__init__.py"
        init.touch(exist_ok=True)
        print(f"  ✓ {d}/__init__.py")
    
    # 移动分析脚本
    print("\n📊 移动分析脚本...")
    analysis = [
        "analyze_dims.py", "analyze_skip_results.py", "apple_desktop_probe.py",
        "compare_embedding_layer8.py", "embed_sum_probe.py", "embed_to_logits.py",
        "head_ablation_probe.py", "inject_layer8_at24.py", "layer_similarity_probe.py",
        "layer_skip_probe.py", "logits_from_layer8.py", "query_embedding.py",
        "rain_analytic_inv.py", "rain_analytic_inv2.py", "rain_backproject.py",
        "rain_backprop_layer35.py", "show_head_top10.py"
    ]
    count = 0
    for f in analysis:
        if Path(f).exists():
            try:
                shutil.move(f, f"scripts\\analysis\\{f}")
                print(f"  ✓ {f}")
                count += 1
            except:
                pass
    
    # 移动实验脚本
    print("\n🧪 移动实验脚本...")
    exp = ["run_experiments.py", "run_shuffled_experiments.py", "save_neuron.py"]
    for f in exp:
        if Path(f).exists():
            try:
                shutil.move(f, f"scripts\\experiments\\{f}")
                print(f"  ✓ {f}")
                count += 1
            except:
                pass
    
    # 移动工具脚本
    print("\n📝 移动工具脚本...")
    utils = ["generate_report.py", "generate_report_cn.py", "generate_multi_report.py", "generate_shuffle_report.py"]
    for f in utils:
        if Path(f).exists():
            try:
                shutil.move(f, f"scripts\\utils\\{f}")
                print(f"  ✓ {f}")
                count += 1
            except:
                pass
    
    # 从tools移动
    print("\n🔧 从tools/移动脚本...")
    tools = Path("tools")
    if tools.exists():
        for f in tools.glob("*.py"):
            if f.name != "__init__.py":
                try:
                    shutil.move(str(f), f"scripts\\utils\\{f.name}")
                    print(f"  ✓ {f.name}")
                    count += 1
                except:
                    pass
    
    # 移动测试
    print("\n✅ 移动测试文件...")
    tests = ["test.py", "test1.py"]
    for f in tests:
        if Path(f).exists():
            try:
                shutil.move(f, f"tests\\{f}")
                print(f"  ✓ {f}")
                count += 1
            except:
                pass
    
    print(f"\n{'='*50}")
    print(f"✅ 整理完成！共移动 {count} 个文件")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
