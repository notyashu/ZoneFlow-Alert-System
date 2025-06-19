import torch

print("PyTorch version:", torch.__version__)
print("CUDA toolkit version PyTorch was built with:", torch.version.cuda)
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\n--- Device {i} ---")
        print("Device Name:", torch.cuda.get_device_name(i))
        print("Memory Allocated:", round(torch.cuda.memory_allocated(i)/1024**2, 2), "MB")
        print("Memory Reserved:", round(torch.cuda.memory_reserved(i)/1024**2, 2), "MB")
else:
    print("CUDA is not available.")
