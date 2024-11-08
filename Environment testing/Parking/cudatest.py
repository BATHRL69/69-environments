import torch

print("PyTorch version: ", torch.__version__)
print("CUDA version: ", torch.version.cuda)
print("CUDA available? ", torch.cuda.is_available())