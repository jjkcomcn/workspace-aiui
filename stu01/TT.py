import torch
print(torch.cuda.is_available())  # 应该输出 True
print(torch.cuda.current_device())  # 输出当前设备的索引
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 输出当前设备的名称
