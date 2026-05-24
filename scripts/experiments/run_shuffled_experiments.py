# -*- coding: utf-8 -*-
"""
随机打乱四个词库顺序，生成新词库，然后一次加载模型运行三模式第八层分析
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.model_loader import load_local_model
from src.pipeline import ProbePipeline

SEED = 42

WORD_FILES = {
    "color_words":  "data/color_words.txt",
    "taste_words":  "data/taste_words.txt",
    "sound_words":  "data/sound_words.txt",
    "shape_words":  "data/shape_words.txt",
}

def shuffle_and_save(src_path: str, dst_path: str, seed: int) -> list[str]:
    words = [line.strip() for line in open(src_path, encoding="utf-8") if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(words)
    Path(dst_path).write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"  Shuffled {len(words)} words -> {dst_path}")
    return words

def main():
    print("=== Step 1: Shuffling word lists ===")
    for key, src in WORD_FILES.items():
        dst = src.replace("data/", "data/shuffled_")
        shuffle_and_save(src, dst, seed=SEED)

    print("\n=== Step 2: Loading model ===")
    config = load_config("configs/default.yaml")
    bundle = load_local_model(config)
    print("Model loaded.\n")

    pipeline = ProbePipeline(config, bundle)

    experiments = [
        ("data/shuffled_color_words.txt", "color_words_shuffled", "颜色词（打乱顺序）"),
        ("data/shuffled_taste_words.txt", "taste_words_shuffled", "味道词（打乱顺序）"),
        ("data/shuffled_sound_words.txt", "sound_words_shuffled", "声音词（打乱顺序）"),
        ("data/shuffled_shape_words.txt", "shape_words_shuffled", "形状词（打乱顺序）"),
    ]

    for word_file, run_name, label in experiments:
        print(f"=== 正在运行：{label} ===")
        pipeline.run_color_words_experiment(word_file=word_file, run_name=run_name)
        print(f"=== 完成：{label} ===\n")

    print("所有打乱顺序实验完成！")

if __name__ == "__main__":
    main()
