import torch

print("PyTorch 版本:", torch.__version__)
print("编译时 CUDA 版本:", torch.version.cuda)   # PyTorch 是用哪个 CUDA 编译的
print("运行时是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())
    print("当前设备:", torch.cuda.current_device())
    print("GPU 名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("运行时 CUDA 版本:", torch.version.cuda)  # 实际运行时用的 CUDA 驱动
