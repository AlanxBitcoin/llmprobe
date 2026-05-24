# -*- coding: utf-8 -*-
"""
一次加载模型，顺序运行味道/声音/形状三个三模式第八层隐藏层分析实验
"""
import sys
from pathlib import Path

# 确保工作目录在路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.model_loader import load_local_model
from src.pipeline import ProbePipeline

def main():
    config = load_config("configs/default.yaml")
    print("Loading model...")
    bundle = load_local_model(config)
    print("Model loaded. Starting experiments...\n")

    pipeline = ProbePipeline(config, bundle)

    experiments = [
        ("data/taste_words.txt",  "taste_words",  "味道词"),
        ("data/sound_words.txt",  "sound_words",  "声音词"),
        ("data/shape_words.txt",  "shape_words",  "形状词"),
    ]

    for word_file, run_name, label in experiments:
        print(f"=== 正在运行实验：{label} ({word_file}) ===")
        pipeline.run_color_words_experiment(word_file=word_file, run_name=run_name)
        print(f"=== 完成：{label} ===\n")

    print("所有实验完成！")

if __name__ == "__main__":
    main()
