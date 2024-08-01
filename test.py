import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import copy

a = jnp.array([[1,1],
               [2,2],
               [3,3]])
b = jnp.array([5,5])
c = jnp.array([2,2])
print(a*b-c)