import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import sobol_seq
from models.GP_SASBO import GP

'''
Pseudocode:
1. Initialization: n_iteration, bound, plant_system(obj_fun, cosntraint1,...), Lipschitz constant L, seed set S0
2. Initialize GP
3. for n_iteration:
    4. define safe set
    5. 
'''

class SafeOpt(GP):
    def __init__(self,plant_system):
        GP.__init__(self,plant_system)
        self.GP_inference_np_jit = jit(self.GP_inference_np)

    def lcb(self,x,b,index):
        pass

    def DE_minimize(self,n_iteration:int,bound,b:float):
        # Initialization
        cons = []
        for i in range(self.n_fun):
            if i == 0:
                obj_fun = self.lcb
            else:
                cons.append([self.lcb])

            


        for i in range(n_iteration):
            

