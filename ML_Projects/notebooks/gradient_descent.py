import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D


# Define the function to be minimized (a simple quadratic function)
def f(x, y):
    return x**2 + y**2


# Define the partial derivatives of the function with respect to x and y
# ∂f/∂x=2x
def df_dx(x, y):
    return 2 * x


def df_dy(x, y):
    return 2 * y


# Define the gradient descent algorithm (∇f)
def gradient_descent(start_x, start_y, learning_rate, num_iteration):
    # Init params
    x = start_x
    y = start_y
    history = []

    # Preform the gradient descent iterations
    for i in range(num_iteration):
        # calculate gradient
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)

        # Update parameters
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

        history.append((x, y, f(x, y)))

    return x, y, f(x, y), history


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
x_opt, y_opt, f_opt, history = gradient_descent(
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
