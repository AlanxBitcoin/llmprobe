from __future__ import annotations

"""属性探针实验的 study 层编排入口。

功能:
- 加载配置与模型资源。
- 读取带属性标签的数据集并构建特征。
- 训练属性探针并导出实验结果文件。
- 保持本层只做流程编排，不承载底层算法细节。
"""

from pathlib import Path
from typing import Any

from ..config import load_config
from ..model_loader import get_model_bundle
from ..probes.probe_attribute import (
    build_feature_bank,
    export_attribute_probe_results,
    load_attribute_rows,
    train_attribute_probes,
)


def run_study(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path = "configs/custom.yaml",
    attribute_file: str | Path = "data/word_attributes.csv",
    output_dir: str | Path = "data/outputs/attribute_probe",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    bundle = get_model_bundle(cfg)
    rows = load_attribute_rows(attribute_file)
    target_layer = int(cfg["analysis"]["target_layer"])
    feature_bank = build_feature_bank(bundle, rows, target_layer, config=cfg)
    results = train_attribute_probes(feature_bank, rows, config=cfg)
    export_attribute_probe_results(output_dir, results)
    return {"output_dir": str(output_dir), "metrics": results}
