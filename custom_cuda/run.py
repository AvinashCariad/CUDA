import torch
from torch.utils.cpp_extension import load

# Compilation of the code depends upon the particular hardware architecture
# Run this code using:  TORCH_CUDA_ARCH_LIST="8.9" python3 run.py

print("Compiling custom CUDA extension dynamically...")
# Compiles vector_add.cu into a usable Python module named 'cuda_extension'
cuda_extension = load(
    name="cuda_extension",
    sources=["vector_add.cu"],
    verbose=True
)
print("Compilation successful!\n")

# Create two standard PyTorch tensors directly on the GPU
tensor_a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
tensor_b = torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda")

# Call your custom raw CUDA kernel directly
tensor_c = cuda_extension.vector_add(tensor_a, tensor_b)

# Print results
print("Tensor A:", tensor_a.cpu())
print("Tensor B:", tensor_b.cpu())
print("Result C (from CUDA):", tensor_c.cpu())

