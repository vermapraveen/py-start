# import sympy as sp
import autograd.numpy as np
from autograd import grad

#################### autograd #####################
import autograd.numpy as np  # Use autograd's numpy
from autograd import grad


# Define the function to minimize
def f(xy):
    x, y = xy  # Unpack the tuple
    return x**2 + y**2  # Use numpy's operators (autograd.numpy)


# Compute partial derivatives
grad_f = grad(f)  # Gradient function returns both df/dx and df/dy

# Test the gradient function
xy_val = [3.0, 4.0]
gradient = grad_f(xy_val)
print("Gradient (df/dx, df/dy):", gradient)
