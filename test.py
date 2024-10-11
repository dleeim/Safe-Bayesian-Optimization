import jax.numpy as jnp

# Sample array with some infinite values
value = jnp.array([10, -5, jnp.inf, 3, -jnp.inf, 8, -1, 4])

# Remove the infinite values
value = value[jnp.isfinite(value)]

# Output the filtered array
print(max(value,key=abs))



