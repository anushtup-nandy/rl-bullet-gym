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

# TENSORS:
x = torch.tensor(1.0, requires_grad =  True)  #remmeber to use floating point numbers --> helpful
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(3.0, requires_grad = True)

y = w*x + b
y.backward() #compute the gradient
print(x.grad) #print gradients
print(w.grad)
print(b.grad)

# AUTOGRAD-tensor2:
x = torch.randn(10, 3)
y = torch.randn(10, 2)
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

