#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 1. The Raw CUDA Kernel running on individual GPU threads
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

// 2. The C++ Wrapper function that bridges PyTorch and your CUDA Kernel
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    // Create an empty output tensor with the same shape and type on the GPU
    auto c = torch::empty_like(a);

    int size = a.numel(); // Total number of elements
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    // Launch the CUDA kernel using data pointers from PyTorch tensors
    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );

    return c;
}

// 3. Bind the C++ function to Python so it can be imported as a module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Custom Vector Addition (CUDA)");
}

