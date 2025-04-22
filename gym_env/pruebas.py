import torch

# Verificar si CUDA está disponible
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# Verificar la versión de PyTorch y si está instalada con soporte CUDA
torch_version = torch.__version__
print(f"PyTorch Version: {torch_version}")

# Verificar qué versión de CUDA está soportada por PyTorch
cuda_version = torch.version.cuda
print(f"CUDA Version: {cuda_version}")
