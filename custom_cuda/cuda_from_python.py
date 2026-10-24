import torch
import torchvision.models as models


#torch.backends.cuda.matmul.fp32_precision = 'tf32' # Options: 'ieee' or 'tf32'
#torch.backends.cudnn.fp32_precision = 'tf32'


# Use only the standardized modern configuration method
torch.set_float32_matmul_precision('high')




# 1. Load a standard model
model = models.resnet18().cuda()

# 2. Enable debugging to print the generated CUDA/Triton source code
import os
os.environ["TORCH_LOGS"] = "output_code"

# 3. Compile the model
optimized_model = torch.compile(model)

# 4. Run a dummy forward pass (this triggers the CUDA code generation)
dummy_input = torch.randn(1, 3, 224, 224).cuda()
out = optimized_model(dummy_input)

