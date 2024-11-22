import sympy as sp

def f(x, y):
    return x**2 + y**2  # Replace this with any function

x_val, y_val = 3.0, 4.0
x_1, y_1 = sp.symbols("x y")

# define a function
f_1 = x_1**2 + y_1**2

#################### sympy #####################
# compute 'symbolic' gradient
df_dx_sympy = sp.diff(f_1, x_1)
df_dy_sympy = sp.diff(f_1, y_1)

# Convert symbolic gradient to numeric functions
df_dx_func_sympy = sp.lambdify((x_1, y_1), df_dx_sympy)
df_dy_func_sympy = sp.lambdify((x_1, y_1), df_dy_sympy)

print("sympy df/dx:", df_dx_func_sympy(x_val, y_val))
print("sympy df/dy:", df_dy_func_sympy(x_val, y_val))
