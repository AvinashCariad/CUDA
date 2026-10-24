import numpy as np
import time

# Set up large synthetic data dimensions
BATCH_SIZE = 512
DIM = 4096

# Initialize inputs and random weights for a 2-layer network
X = np.random.randn(BATCH_SIZE, DIM).astype(np.float32)
W1 = np.random.randn(DIM, DIM).astype(np.float32) * 0.01
W2 = np.random.randn(DIM, DIM).astype(np.float32) * 0.01

def relu(x):
    return np.maximum(0, x)

def relu_backward(grad_output, x_forward):
    # Gradient of ReLU is 1 if x > 0 else 0
    return grad_output * (x_forward > 0)

# ------------------------------------------------------------------
# APPROACH 1: Standard Backpropagation (Caches Everything)
# ------------------------------------------------------------------
def run_standard():
    start = time.time()

    # FORWARD PASS: Cache every single intermediate state in RAM
    A1 = X @ W1          # Cached
    H1 = relu(A1)        # Cached
    A2 = H1 @ W2         # Cached
    H2 = relu(A2)        # Cached
    loss = np.sum(H2)

    # Track fake memory footprint based on elements stored
    total_elements_cached = A1.size + H1.size + A2.size + H2.size

    # BACKWARD PASS: Read directly from the caches
    dH2 = np.ones_like(H2)
    dA2 = relu_backward(dH2, A2)
    dW2 = H1.T @ dA2
    dH1 = dA2 @ W2.T
    dA1 = relu_backward(dH1, A1)
    dW1 = X.T @ dA1

    end = time.time()
    return end - start, total_elements_cached

# ------------------------------------------------------------------
# APPROACH 2: Activation Checkpointing (Drops & Recalculates)
# ------------------------------------------------------------------
def run_checkpointed():
    start = time.time()

    # FORWARD PASS: Save only the "Checkpoint" (X), immediately drop hidden states
    # (We simulate dropping by not keeping references to A1, H1, A2 after H2 is made)
    A1_temp = X @ W1
    H1_checkpoint = relu(A1_temp) # This is our designated save-state boundary

    A2_temp = H1_checkpoint @ W2
    H2 = relu(A2_temp)
    loss = np.sum(H2)

    # Free memory simulation: Delete intermediate structures we don't want to store
    del A1_temp, A2_temp

    # At peak forward pass, we are only holding H1_checkpoint and H2
    peak_elements_cached = H1_checkpoint.size + H2.size

    # BACKWARD PASS: Recalculate omitted layers on the fly
    dH2 = np.ones_like(H2)

    # Recomputation Phase: Recreate A2_temp right when we need it
    A2_recomputed = H1_checkpoint @ W2
    dA2 = relu_backward(dH2, A2_recomputed)
    dW2 = H1_checkpoint.T @ dA2
    dH1 = dA2 @ W2.T

    # Recomputation Phase: Recreate A1_temp right when we need it
    A1_recomputed = X @ W1
    dA1 = relu_backward(dH1, A1_recomputed)
    dW1 = X.T @ dA1

    end = time.time()
    return end - start, peak_elements_cached

# ------------------------------------------------------------------
# Run and Compare
# ------------------------------------------------------------------
print("Simulating workflows in NumPy...\n")

std_time, std_mem = run_standard()
print(f"STANDARD BACKPROPAGATION:")
print(f"-> Active Matrix Elements Stored: {std_mem:,}")
print(f"-> Execution Time:                {std_time:.4f} seconds\n")

cp_time, cp_mem = run_checkpointed()
print(f"ACTIVATION CHECKPOINTING:")
print(f"-> Active Matrix Elements Stored: {cp_mem:,}")
print(f"-> Execution Time:                {cp_time:.4f} seconds\n")

mem_saved = ((std_mem - cp_mem) / std_mem) * 100
time_overhead = ((cp_time - std_time) / std_time) * 100

print("-" * 50)
print(f"Memory Matrix Footprint Saved:  {mem_saved:.1f}% Reduction")
print(f"Compute Recomputation Overhead: {time_overhead:.1f}% Increase")
print("-" * 50)

