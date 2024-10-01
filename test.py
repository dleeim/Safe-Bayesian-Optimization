import jax.numpy as jnp
from scipy.stats import qmc
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint

a = jnp.array([[1.,2.,3.,4.]])
b = jnp.array([[1.,2.]])

max_len = max(a.shape[1],b.shape[1])
b_nan = jnp.array([(max_len-b.shape[1])*[jnp.nan]])
print(b_nan)
b = jnp.hstack((b,b_nan))
c = jnp.append(a,b,axis=0)
print(c)
print(jnp.nanmean(c,axis=0))




