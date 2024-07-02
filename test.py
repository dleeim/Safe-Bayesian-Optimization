import jax.numpy as jnp


a = jnp.array([[1,1],
               [2,2]])
print(jnp.linalg.norm(a,axis=1))
