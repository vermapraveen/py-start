import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import torch


# Define the function to be minimized (a simple quadratic function)
def f(x, y):
    return x**2 + y**2


# Define the gradient descent algorithm (∇f)
def gradient_descent_with_torch(start_x, start_y, learning_rate, num_iteration):
    # Init params
    x = torch.tensor(start_x, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(start_y, dtype=torch.float32, requires_grad=True)
    history = []

    # Preform the gradient descent iterations
    for i in range(num_iteration):
        # Compute the function value
        z = f(x, y)

        # Backpropagation to calculate gradients
        z.backward()

        history.append((x.detach().item(), y.detach().item(), z.detach().item()))
        # Update parameters manually using gradients
        with torch.no_grad():
            x -= learning_rate * x.grad
            y -= learning_rate * y.grad

            # Clear gradients for the next iteration
            x.grad.zero_()
            y.grad.zero_()

    return x.item(), y.item(), z.item(), history


# start plotting the function
# Define mesg grid
x_range = np.arange(-10, 10, 0.1)
y_range = np.arange(-10, 10, 0.1)

X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# Perform gradient descent and plot the result
start_x, start_y = 8, 8
learning_rate = 0.1
num_iterration = 20
x_opt, y_opt, f_opt, history = gradient_descent_with_torch(
    start_x, start_y, learning_rate, num_iterration
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="coolwarm")
ax.scatter(*zip(*history), c="r", marker="o")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.show()
