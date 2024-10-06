import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import optimize
jax.config.update("jax_enable_x64", True)


def bioreactor_ss_equations(input):
    
    mu_max = 0.5
    Ks = 0.2
    Yxs = 0.5
    si = 20

    x,s,D = input

    mu = ((mu_max*s)/(Ks + s)) 
    
    df = [
        mu - D,
        D*si - D*s - ((mu * x)/(Yxs))    
    ]

    return df

def bioreactor_obj(x_sp,s_0,D_0):
    x0 = jnp.array([s_0,D_0])
    bioreactor_ss_equations_const_x= lambda x: bioreactor_ss_equations(x.append(x_sp))
    s,D = optimize.fsolve(bioreactor_ss_equations_const_x,x0)
