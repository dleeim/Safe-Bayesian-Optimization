import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)


# Objective function
def objective(x):
    print(type(x[0]**2 + x[1]**2))
    return x[0]**2 + x[1]**2

# Equality constraint: x0 + x1 = 1
def eq_constraint(x):
    return x[0] + x[1] - 1

# Inequality constraint: x0 >= 0, x1 >= 0
def ineq_constraint(x):
    return x  # This will apply x0 >= 0 and x1 >= 0

# Initial guess
x0 = jnp.array([0.1,0.1])

# Define the constraints
constraints = (
    {'type': 'eq', 'fun': eq_constraint},    # Equality constraint
    {'type': 'ineq', 'fun': lambda x: x[0]}, # x0 >= 0
    {'type': 'ineq', 'fun': lambda x: x[1]}  # x1 >= 0
)

# Solve the optimization problem
result = minimize(objective, x0, method='SLSQP', constraints=constraints,jac='3-point')

# Display the results
print("Optimal solution:", result.x)
print("Objective value at the optimum:", result.fun)
