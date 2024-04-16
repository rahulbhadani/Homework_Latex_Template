import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective(x):
    x, y = x
    return np.exp(x*y)

# Define the constraint function
def constraint(x):
    x, y = x
    return ((x**3) + (y**3) - 16)

# Initial guess for x and y
x0 = np.array([0.0, 0.001])

# Set up the optimization problem
constraints = {'type': 'eq', 'fun': constraint}
result = minimize(objective, x0, constraints=constraints)

# Print the results
print("Optimal values are achieved at:")
print(f"x = {result.x[0]:.4f}")
print(f"y = {result.x[1]:.4f}")
print(f"Minimum value of f(x, y) = {result.fun:.4f}")


# %%
import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective(x):
    x, y = x
    return np.exp(x*y)

# Define the constraint function
def constraint(x):
    x, y = x
    return ((x**3) + (y**3) - 16)

# Initial guess for x and y
x0 = np.array([0.01, 0.01])

# Set up the optimization problem
constraints = {'type': 'eq', 'fun': constraint}
result = minimize(objective, x0, constraints=constraints)

# Print the results
print("Optimal values are achieved at:")
print(f"x = {result.x[0]:.4f}")
print(f"y = {result.x[1]:.4f}")
print(f"Optimal value of f(x, y) = {result.fun:.4f}")


# %%
import numpy as np
import plotly.graph_objects as go

# Define the objective function f(x, y)
def f(x, y):
    return np.exp(x*y)

# Define the constraint function g(x, y)
def g(x, y):
    return ((x**3) + (y**3) - 16)

# Create a meshgrid for x and y values
x_vals = np.linspace(-1.5, 1.5, 100)
y_vals = np.linspace(-3.5, 3.5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z_f = f(X, Y)
Z_g = g(X, Y)

# Create the 3D surface plot
fig = go.Figure()

# Add the surface for f(x, y)
fig.add_trace(go.Surface(z=Z_f, x=X, y=Y, colorscale='Viridis', name='f(x, y)'))

# Add the constraint surface for g(x, y)
fig.add_trace(go.Surface(z=Z_g, x=X, y=Y, 
                         colorscale='Reds',
                         opacity=0.5, showscale=False,
                         name='4x^2 + y^2 = 9'))

# Set layout
fig.update_layout(
    title='3D Surface Plot',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='Value',
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

# Show the interactive plot
fig.show()