import torch

print(torch.__version__)
print(torch.version.cuda)  # Check CUDA version PyTorch was built with
print(torch.backends.cudnn.enabled)  # Check if cuDNN is enabled

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA-compatible GPU found.")