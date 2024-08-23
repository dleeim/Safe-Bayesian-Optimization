import numpy as np
import jax.numpy as jnp
from scipy.optimize import differential_evolution, NonlinearConstraint

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2 + 3

# Define the constraints


# Define the bounds
bounds = [(-5, 5), (-5, 5)]
con1 = lambda x: x[0]
nlc1 = NonlinearConstraint(con1,-np.inf,-2)
con2 = lambda x: x[1]
nlc2 = NonlinearConstraint(con2, -np.inf,-3)
# Run the differential evolution algorithm with constraints
result = differential_evolution(objective_function, bounds,constraints=[nlc1,nlc2])

# Print the results
print('Optimal parameters:', result.x)
print('Objective function value:', result.fun)

