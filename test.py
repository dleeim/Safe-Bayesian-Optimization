import jax.numpy as jnp
import sobol_seq
import matplotlib.pyplot as plt

a = jnp.array([1,2,3,4,5])
plt.figure()
plt.plot(a,'k-')
plt.plot(jnp.array([1.]*len(a)),'r--')
plt.show()