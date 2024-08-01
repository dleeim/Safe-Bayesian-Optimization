import jax 
import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize
import sobol_seq
from models.GP_Classic import GP

class SafeOpt(GP):
    def __init__(self,plant_system):
        GP.__init__(self,plant_system)

    def Safe_set_sampling(self,x_dim,n_sample,bound):
        '''
        Description:
        Arguments:
        Results
        '''
        fraction = sobol_seq.i4_sobol_generate(x_dim,n_sample)
        lb = bound[0]
        ub = bound[1]
        sample = lb + (ub-lb) * fraction

        return sample