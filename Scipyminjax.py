import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

def objective(x):
    x = x.squeeze()
    jax.debug.print('x: {}', x)
    x = jnp.exp(x)
    a = jnp.array([[x,0.],[0.,x]])
    b = jnp.diag(a)
    c = jnp.log(b)
    jax.debug.print('a: {}', a)
    jax.debug.print('b: {}', b)
    jax.debug.print('c: {}', c)
    return jnp.sum(c)

def objective_with_grad(x):
    return (objective(x),grad(objective, argnums=0)(x))

a = jnp.array([1.])
bound = [(-10.0, 10.0)]
res = minimize(objective_with_grad,a,method='SLSQP',bounds=bound,jac=True,tol=1e-12)
print(res.x,res.fun)
print(objective(jnp.array([0.5])))
