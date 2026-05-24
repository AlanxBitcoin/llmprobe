import sys
import torch

print("Python版本：", sys.version)
print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())
if torch.cuda.is_available():
    print("可用显卡数量：", torch.cuda.device_count())
    print("当前显卡名称：", torch.cuda.get_device_name(0))
    print("显卡显存总量(GB)：", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))