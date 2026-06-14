"""本地 CUDA 与 PyTorch 运行环境快速自检脚本。

功能:
- 输出 Python、PyTorch 与 CUDA 可用性信息。
- 在 CUDA 可用时显示显卡数量、名称与显存。
- 仅用于本地诊断，不属于正式 study/probe 流水线接口。
"""

import sys
import torch

print("Python版本：", sys.version)
print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())
if torch.cuda.is_available():
    print("可用显卡数量：", torch.cuda.device_count())
    print("当前显卡名称：", torch.cuda.get_device_name(0))
    print("显卡显存总量(GB)：", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
