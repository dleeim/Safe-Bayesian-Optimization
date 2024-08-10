import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd

from models import SafeRTO
from problems import Benoit_Problem
from problems import Rosenbrock_Problem

# Preparation: 
jax.config.update("jax_enable_x64", True)
plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system_tight]
GP_m = SafeRTO.SafeOpt(plant_system)
n_sample = 4
x_i = jnp.array([1.1,-0.8])
r = 0.5
b = 3
X,Y = GP_m.Data_sampling(n_sample,x_i,r)
GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=10, var_out=True)

def test_Safeset_sampling():
    x_dim = 2
    n_sample = 2000
    bound = jnp.array([[-0.6,-1], # row 1 = lower bounds
                       [1.5,1]],dtype=jnp.float64) # row 2 = upper bounds

    # Safe set
    sample = GP_m.sobol_seq_sampling(x_dim,n_sample,bound)
    # print(sample)
    sample_safe = GP_m.Safe_filter(sample,b)

    # Minimizer set
    multi_start = 5
    xopt, minucb = GP_m.minimize_ucb(sample_safe,b,multi_start)
    print(f"xopt, minucb: {xopt, minucb}")
    sample_minimizer = GP_m.minimizer_filter(sample_safe,b,minucb)
    # print(f"sample minimizer: {sample_minimizer}")

    # Expander set
    eps = 0.1
    sample_expander = GP_m.expander_filter(sample_safe,b,eps)
    # print(sample_expander)

    set_minimizer, set_expander = GP_m.Set_sampling(x_dim,b,multi_start,bound,eps)
    print(f"set_minimizer: {set_minimizer}")
    print(f"set_expander: {set_expander}")
    plt.figure()
    plt.plot(sample[:,0],sample[:,1],'ro',label='unsafe')
    plt.plot(sample_safe[:,0],sample_safe[:,1],'bo',label='safe')
    plt.plot(sample_minimizer[:,0],sample_minimizer[:,1],'go',label='minimizer')
    plt.plot(sample_expander[:,0],sample_expander[:,1],'yo',label='expander')
    plt.legend()
    plt.show()

    # xopt, funopt = GP_m.minimize_minimizer(set_minimizer, b, minucb)
    # print(f"opt for minimizer: {xopt, funopt}")
    # xopt,funopt = GP_m.minimize_expander(set_expander,b,eps)
    # print(f"opt for expander: {xopt, funopt}")



if __name__ == "__main__":
    test_Safeset_sampling()
