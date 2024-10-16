import jax.numpy as jnp
from scipy.optimize import differential_evolution, NonlinearConstraint

b = 2
for i in range(2):
    while b > 1:
        print(i)