import torch
import torch.nn as nn
import time
from torch.utils.checkpoint import checkpoint

# Ensure we are running on a CUDA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ------------------------------------------------------------------
# 1. Define a heavy model layer that consumes significant VRAM
# ------------------------------------------------------------------
class HeavyBlock(nn.Module):
    def __init__(self, dim=4096):
        super().__init__()
        # Large linear layers that create massive activation matrices
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            # Drop activations inside this function and recalculate on backprop
            # use_reentrant=False is recommended for modern PyTorch workflows
            return checkpoint(self._inner_forward, x, use_reentrant=False)
        else:
            return self._inner_forward(x)

    def _inner_forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

# ------------------------------------------------------------------
# 2. Benchmark Function to measure VRAM and Compute Time
# ------------------------------------------------------------------
def run_benchmark(use_checkpoint):
    # Reset CUDA memory stats for clean measurements
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Initialize model, optimizer, and large input batch
    model = HeavyBlock().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(512, 4096, device=device) # Massive batch size

    # Measure execution time
    torch.cuda.synchronize()
    start_time = time.time()

    # Forward Pass
    outputs = model(inputs, use_checkpoint=use_checkpoint)
    loss = outputs.sum()

    # Backward Pass & Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    end_time = time.time()

    # Extract peak VRAM utilization
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # Convert to MB
    execution_time = end_time - start_time

    return peak_vram, execution_time

# ------------------------------------------------------------------
# 3. Execution & Comparison
# ------------------------------------------------------------------
print("Running STANDARD Backpropagation...")
std_vram, std_time = run_benchmark(use_checkpoint=False)
print(f"-> Peak VRAM: {std_vram:.2f} MB")
print(f"-> Time:      {std_time:.4f} seconds\n")

print("Running ACTIVATION CHECKPOINTING Backpropagation...")
cp_vram, cp_time = run_benchmark(use_checkpoint=True)
print(f"-> Peak VRAM: {cp_vram:.2f} MB")
print(f"-> Time:      {cp_time:.4f} seconds\n")

# Calculate metrics
vram_saved = ((std_vram - cp_vram) / std_vram) * 100
time_overhead = ((cp_time - std_time) / std_time) * 100

print("-" * 50)
print(f"Memory Saved:      {vram_saved:.1f}% Reduction")
print(f"Compute Overhead:  {time_overhead:.1f}% Increase")
print("-" * 50)

