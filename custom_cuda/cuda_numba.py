import numpy as np
from numba import cuda

# This Python function compiles directly into a native CUDA kernel
@cuda.jit
def process_gpu_data(io_array):
    # Map thread positions exactly like raw C++ CUDA (threadIdx.x, blockIdx.x)
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2.0  # Element-wise multiplication on the GPU

# Create data on the Host (CPU)
data = np.array([1, 2, 3, 4], dtype=np.float32)

# Copy data to the Device (GPU), run the kernel, and copy it back
device_data = cuda.to_device(data)
process_gpu_data[1, 4](device_data) # [Blocks, Threads per block]
result = device_data.copy_to_host()

print("Result from compiled GPU code:", result)

