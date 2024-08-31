import jax.numpy as jnp
import time


import jax.numpy as jnp

my_list = jnp.array([[1],
                     [2],
                     [3],
                     [4]])
a = jnp.array([4])

print(jnp.any(my_list[:]==a))