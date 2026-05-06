import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
import onnxruntime as ort
print(ort.get_available_providers())