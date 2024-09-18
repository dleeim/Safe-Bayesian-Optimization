import jax.numpy as jnp
import sobol_seq
import matplotlib.pyplot as plt

eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
print(eps)
