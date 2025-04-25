import torch
# import kikuchipy as kp

print("Setting device")

if torch.cuda.is_available():
    print("Running on CUDA")
else:
    print("Running on CPU")

print("CUDA Available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading networks")

ones = torch.tensor([1], device = device)
print(ones)
print("Loaded everything")

