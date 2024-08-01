import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd

from models  import SafeRTO
from problems import Benoit_Problem
from problems import Rosenbrock_Problem

# Preparation: 
plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system_tight]

GP_m = SafeRTO.SafeOpt(plant_system)

def test_Safe_set_sampling():
    x_dim = 2
    n_sample = 2000
    bound = jnp.array([[-0.6,-1], # row 1 = lower bounds
                       [1.5,1]]) # row 2 = upper bounds
    
    sample = GP_m.Safe_set_sampling(x_dim,n_sample,bound)
    # print(sample)
    plt.figure()
    plt.plot(sample[:,0],sample[:,1],'ro')
    plt.xlim(-0.6, 1.5)
    plt.ylim(-1,1)
    plt.show()
    
    

