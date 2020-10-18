import random
import os
import torch
import numpy as np

print('[SEED] Setting seeds for reproducibility...')

# Random
random.seed(42)

# Python
os.environ['PYTHONHASHSEED'] = str(42)

# Numpy
np.random.seed(42)

# PyTorch
torch.manual_seed(42)

# CUDA
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42) # if you are using multi-GPU

# CuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
