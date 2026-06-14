from __future__ import annotations

"""线性探针实验的 study 层编排入口。

功能:
- 加载配置与模型资源。
- 读取带标签词表并构建探针训练数据集。
- 训练线性探针并导出评估与结果文件。
- 保持 study 层聚焦流程编排，算法细节下沉到 probe 模块。
"""

from pathlib import Path
from typing import Any

from ..config import load_config
from ..model_loader import get_model_bundle
from ..probes.probe_linear import (
    build_probe_dataset,
    export_probe_results,
    load_labeled_words,
    train_linear_probe,
)


def run_study(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path = "configs/custom.yaml",
    label_file: str | Path = "data/word_labels.csv",
    output_dir: str | Path = "data/outputs/probe",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    bundle = get_model_bundle(cfg)
    rows = load_labeled_words(label_file)
    target_layer = int(cfg["analysis"]["target_layer"])
    dataset = build_probe_dataset(bundle, rows, target_layer, config=cfg)
    results = train_linear_probe(dataset, config=cfg)
    export_probe_results(output_dir, results)
    return {"output_dir": str(output_dir), "metrics": results}
