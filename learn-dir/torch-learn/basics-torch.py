import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

'''
1. Tensors
2. Autograd
3. Numpy->Torch
4. Input pipeline
5. Pretrained Model
6. Custom Model
7. Save Model
'''

#TENSORS:
x = torch.tensor(1, requires_grad = true)
w = torch.tensor(2, requires_grad = true)

