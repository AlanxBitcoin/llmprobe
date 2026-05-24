import os, shutil
from pathlib import Path

base = Path(".")
for d in ["scripts", "scripts/analysis", "scripts/experiments", "scripts/utils", "tests"]:
    Path(d).mkdir(parents=True, exist_ok=True)

for d in ["scripts", "scripts/analysis", "scripts/experiments", "scripts/utils", "tests"]:
    Path(d, "__init__.py").touch()

files_analysis = ["analyze_dims.py", "analyze_skip_results.py", "apple_desktop_probe.py", "compare_embedding_layer8.py", "embed_sum_probe.py", "embed_to_logits.py", "head_ablation_probe.py", "inject_layer8_at24.py", "layer_similarity_probe.py", "layer_skip_probe.py", "logits_from_layer8.py", "query_embedding.py", "rain_analytic_inv.py", "rain_analytic_inv2.py", "rain_backproject.py", "rain_backprop_layer35.py", "show_head_top10.py"]
files_exp = ["run_experiments.py", "run_shuffled_experiments.py", "save_neuron.py"]
files_utils = ["generate_report.py", "generate_report_cn.py", "generate_multi_report.py", "generate_shuffle_report.py"]
files_test = ["test.py", "test1.py"]

for f in files_analysis:
    if Path(f).exists(): shutil.move(f, f"scripts/analysis/{f}")
for f in files_exp:
    if Path(f).exists(): shutil.move(f, f"scripts/experiments/{f}")
for f in files_utils:
    if Path(f).exists(): shutil.move(f, f"scripts/utils/{f}")
for f in files_test:
    if Path(f).exists(): shutil.move(f, f"tests/{f}")

tools = Path("tools")
if tools.exists():
    for f in tools.glob("*.py"):
        if f.name != "__init__.py": shutil.move(str(f), f"scripts/utils/{f.name}")

print("✓ 项目整理完成")
