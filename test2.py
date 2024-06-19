import jax
import jax.numpy as jnp
from jax import grad
from scipy.optimize import minimize

# Define the objective function
def objective(x):
    return x[0]**2 + x[1]**2

# JAX gradient of the objective function
objective_grad = jax.grad(objective)

# Equality constraint
def eq_constraint(x):
    return x[0] + x[1] - 1

# Inequality constraint
def ineq_constraint(x):
    return x[0] - 0.5

# Initial guess
x0 = jnp.array([0.0, 0.0])

# Constraints dictionary
constraints = (
    {'type': 'eq', 'fun': eq_constraint},
    {'type': 'ineq', 'fun': ineq_constraint}
)

# Bounds
bounds = [(0, None), (0, None)]

# Custom gradient function for Scipy's minimize
def objective_with_grad(x):
    return float(objective(x)), jax.numpy.asarray(objective_grad(x))

# Minimization using SLSQP
result = minimize(objective_with_grad, x0, method='SLSQP', bounds=bounds, constraints=constraints, jac=True)

print(result)
