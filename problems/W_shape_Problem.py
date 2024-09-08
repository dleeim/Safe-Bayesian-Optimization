import jax.numpy as jnp

def W_shape(x,d):
    val = jnp.sin(x*d) + jnp.sqrt(d)*x**2 - 0.5*x
    return val

