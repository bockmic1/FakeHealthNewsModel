import torch
print("CUDA verfügbar:", torch.cuda.is_available())
if torch.cuda.is_available():
    tensor = torch.rand(3, 3).cuda()
    print(tensor)
else:
    print("Keine CUDA GPUs verfügbar.")
