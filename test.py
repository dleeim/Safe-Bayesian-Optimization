import time
import jax.numpy as jnp
from jax import jit

# Define the function to be jitted
def squared_seuclidean_jax_optimized(X, Y, V):
    V_sqrt_inv = V**-0.5
    X_adjusted = X * V_sqrt_inv
    Y_adjusted = Y * V_sqrt_inv
    dist_mat = -2 * jnp.dot(X_adjusted, Y_adjusted.T) + jnp.sum(X_adjusted**2, axis=1)[:, None] + jnp.sum(Y_adjusted**2, axis=1)
    return dist_mat

# Example data
X_norm = jnp.array([[1.0, 2.0], [3.0, 4.0]])
Y_norm = jnp.array([[5.0, 6.0], [7.0, 8.0]])
V = jnp.array([1.0, 1.0])

# Measure execution time for the first call with re-jitting
start_time = time.time()
squared_seuclidean_jax_jit = jit(squared_seuclidean_jax_optimized)
dist_mat_jit = squared_seuclidean_jax_jit(X_norm, Y_norm, V)
first_rejit_call_time = time.time() - start_time
print("First re-jit call execution time:", first_rejit_call_time)

# Measure execution time for the second call with re-jitting
start_time = time.time()
squared_seuclidean_jax_jit = jit(squared_seuclidean_jax_optimized)
dist_mat_jit = squared_seuclidean_jax_jit(X_norm, Y_norm, V)
second_rejit_call_time = time.time() - start_time
print("Second re-jit call execution time:", second_rejit_call_time)

# Measure execution time for the third call with re-jitting
start_time = time.time()
squared_seuclidean_jax_jit = jit(squared_seuclidean_jax_optimized)
dist_mat_jit = squared_seuclidean_jax_jit(X_norm, Y_norm, V)
third_rejit_call_time = time.time() - start_time
print("Third re-jit call execution time:", third_rejit_call_time)
