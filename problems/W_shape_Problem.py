import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def W_shape(x,d):
    value = jnp.sin(x*d) + jnp.sqrt(d)*x**2 - 0.5*x
    return value

def W_shape_constraint(x,d):
    value = jnp.sin(x*d) + jnp.sqrt(d)*x**2 - 0.5*x -2.
    return value