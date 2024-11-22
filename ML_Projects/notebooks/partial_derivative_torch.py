import torch

# Define variables as tensors with gradients enabled
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

# Define any function
f = x**2 + y**2  # Replace with any function

# Compute gradients
f.backward()  # Automatically computes the gradients

# Access gradients
print("df/dx:", x.grad.item())
print("df/dy:", y.grad.item())
