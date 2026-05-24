#!/usr/bin/env python3
"""
LLM Probe 项目自动整理脚本

本脚本将散落在根目录和 tools/ 的脚本按功能分类
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class ProjectOrganizer:
    """项目组织器"""
    
    def __init__(self, base_path: str = "."):
        self.base = Path(base_path).resolve()
        self.moved_files: List[str] = []
        self.failed_files: List[Tuple[str, str]] = []
        
    def create_directories(self) -> bool:
        """创建所需的目录结构"""
        try:
            dirs = [
                self.base / "scripts",
                self.base / "scripts" / "analysis",
                self.base / "scripts" / "experiments",
                self.base / "scripts" / "utils",
                self.base / "tests"
            ]
            
            for d in dirs:
                d.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ {d.relative_to(self.base)}")
            
            return True
        except Exception as e:
            print(f"  ✗ Error creating directories: {e}")
            return False
    
    def create_init_files(self) -> bool:
        """创建 __init__.py 文件"""
        try:
            dirs = [
                self.base / "scripts",
                self.base / "scripts" / "analysis",
                self.base / "scripts" / "experiments",
                self.base / "scripts" / "utils",
                self.base / "tests"
            ]
            
            for d in dirs:
                init_file = d / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    print(f"  ✓ {init_file.relative_to(self.base)}")
            
            return True
        except Exception as e:
            print(f"  ✗ Error creating __init__.py files: {e}")
            return False
    
    def move_files_to_analysis(self) -> int:
        """移动分析脚本到 scripts/analysis/"""
        files = [
            "analyze_dims.py",
            "analyze_skip_results.py",
            "apple_desktop_probe.py",
            "compare_embedding_layer8.py",
            "embed_sum_probe.py",
            "embed_to_logits.py",
            "head_ablation_probe.py",
            "inject_layer8_at24.py",
            "layer_similarity_probe.py",
            "layer_skip_probe.py",
            "logits_from_layer8.py",
            "query_embedding.py",
            "rain_analytic_inv.py",
            "rain_analytic_inv2.py",
            "rain_backproject.py",
            "rain_backprop_layer35.py",
            "show_head_top10.py"
        ]
        
        return self._move_files(files, "scripts/analysis")
    
    def move_files_to_experiments(self) -> int:
        """移动实验脚本到 scripts/experiments/"""
        files = [
            "run_experiments.py",
            "run_shuffled_experiments.py",
            "save_neuron.py"
        ]
        
        return self._move_files(files, "scripts/experiments")
    
    def move_files_to_utils(self) -> int:
        """移动工具脚本到 scripts/utils/"""
        files = [
            "generate_report.py",
            "generate_report_cn.py",
            "generate_multi_report.py",
            "generate_shuffle_report.py"
        ]
        
        moved = self._move_files(files, "scripts/utils")
        
        # 从 tools/ 目录移动
        tools_dir = self.base / "tools"
        if tools_dir.exists():
            for py_file in tools_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    src = py_file
                    dst = self.base / "scripts" / "utils" / py_file.name
                    
                    try:
                        shutil.move(str(src), str(dst))
                        print(f"  ✓ {py_file.name}")
                        self.moved_files.append(str(src.relative_to(self.base)))
                        moved += 1
                    except Exception as e:
                        print(f"  ✗ {py_file.name}: {e}")
                        self.failed_files.append((py_file.name, str(e)))
        
        return moved
    
    def move_files_to_tests(self) -> int:
        """移动测试文件到 tests/"""
        files = ["test.py", "test1.py"]
        
        return self._move_files(files, "tests")
    
    def _move_files(self, files: List[str], dest_dir: str) -> int:
        """通用文件移动方法"""
        moved = 0
        dest_path = self.base / dest_dir
        
        for filename in files:
            src = self.base / filename
            if src.exists():
                dst = dest_path / filename
                
                try:
                    shutil.move(str(src), str(dst))
                    print(f"  ✓ {filename}")
                    self.moved_files.append(str(src.relative_to(self.base)))
                    moved += 1
                except Exception as e:
                    print(f"  ✗ {filename}: {e}")
                    self.failed_files.append((filename, str(e)))
        
        return moved
    
    def cleanup_empty_dirs(self) -> bool:
        """清理空目录"""
        try:
            tools_dir = self.base / "tools"
            if tools_dir.exists():
                # 检查是否为空
                if not any(tools_dir.glob("*.py")):
                    # 移除此目录
                    shutil.rmtree(str(tools_dir))
                    print(f"  ✓ Removed empty tools/ directory")
            
            return True
        except Exception as e:
            print(f"  ! Warning: Could not cleanup: {e}")
            return True  # 不中断流程
    
    def run(self) -> bool:
        """执行完整的整理流程"""
        print("=" * 70)
        print("  LLM Probe 项目整理脚本")
        print("=" * 70)
        print()
        
        print("📁 Creating directory structure...")
        if not self.create_directories():
            return False
        print()
        
        print("🎯 Creating __init__.py files...")
        if not self.create_init_files():
            return False
        print()
        
        total_moved = 0
        
        print("📊 Moving analysis scripts...")
        total_moved += self.move_files_to_analysis()
        print()
        
        print("🧪 Moving experiment scripts...")
        total_moved += self.move_files_to_experiments()
        print()
        
        print("📝 Moving utility scripts...")
        total_moved += self.move_files_to_utils()
        print()
        
        print("✅ Moving test files...")
        total_moved += self.move_files_to_tests()
        print()
        
        print("🧹 Cleaning up...")
        self.cleanup_empty_dirs()
        print()
        
        # 报告结果
        print("=" * 70)
        print(f"✓ 整理完成！共移动 {total_moved} 个文件")
        if self.failed_files:
            print(f"⚠ {len(self.failed_files)} 个文件移动失败:")
            for filename, error in self.failed_files:
                print(f"  - {filename}: {error}")
        print("=" * 70)
        print()
        
        self.print_new_structure()
        
        return True
    
    def print_new_structure(self):
        """打印新的项目结构"""
        structure = """
新的项目结构:

probe/
├── src/                          # 核心库代码
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
├── scripts/                      # 脚本和工具集合
│   ├── __init__.py
│   │
│   ├── analysis/                 # 分析脚本
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
│   └── utils/                    # 工具和报告脚本
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
│   ├── test.py                   # 模型加载和推理测试
│   └── test1.py                  # 环境和依赖检查
│
├── configs/                      # 配置文件 ✓ 保持不变
│   ├── default.yaml
│   └── quantized-4bit.yaml
│
├── data/                         # 数据目录 ✓ 保持不变
│
├── outputs/                      # 输出目录 ✓ 保持不变
│
├── main.py                       # 主入口脚本 ✓ 保持不变
├── requirements.txt              # 依赖声明 ✓ 保持不变
├── README.md                     # 项目文档 ✓ 保持不变
├── REORGANIZATION_PLAN.md        # 整理方案文档
├── MIGRATION_INSTRUCTIONS.md     # 迁移说明文档
└── run_migration.py              # 本脚本
        """
        print(structure)


def main():
    """主函数"""
    try:
        organizer = ProjectOrganizer()
        success = organizer.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
