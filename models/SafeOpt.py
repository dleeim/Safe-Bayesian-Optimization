import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import cdist
import sobol_seq
from models.GP_SafeOpt import GP

'''
Pseudocode:
1. Initialization: n_iteration, bound, plant_system(obj_fun, cosntraint1,...), Lipschitz constant L, seed set S0
2. Initialize GP
3. for n_iteration:
    4. define safe set
    5. 
'''

class SafeOpt(GP):
    def __init__(self,plant_system,bound,b):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.b = b
        self.GP_inference_jit = jit(self.GP_inference)
        self.safe_set_cons = []
        for i in range(1, self.n_fun):
            con = NonlinearConstraint(lambda x: self.lcb(x,i),0,jnp.inf)
            self.safe_set_cons.append(con)
        

    def calculate_plant_outputs(self,x):
        plant_output            = []
        for plant in self.plant_system:
            plant_output.append(plant(x)) 

        return jnp.array(plant_output)
    
    def ucb(self,x,i):
        GP_inference = self.GP_inference_jit(x)
        mean, var = GP_inference[0][i], GP_inference[1][i]
        value = mean + self.b*jnp.sqrt(var)
        return value
    
    def lcb(self,x,i):
        GP_inference = self.GP_inference_jit(x)
        mean, var = GP_inference[0][i], GP_inference[1][i]
        value = mean - self.b*jnp.sqrt(var)
        return value
    
    def minimize_objective_ucb(self):
        obj_fun = lambda x: self.ucb(x,0)
        result = differential_evolution(obj_fun,self.bound,constraints=self.safe_set_cons)

        return result.x, result.fun

    
    def Minimizer(self):
        '''
        Constriants: 
        1. safe set for every constriants index from i to n
        2. need to find min upper confidence bound and lcb < min ucb
        3. 
        '''
        # Setting Optimization Problem
        obj_fun = lambda x: self.GP_inference_jit(x)[1][0] # objective function standard deviation
        cons = self.safe_set_cons
        min_objective_ucb = self.minimize_objective_ucb()
        con = NonlinearConstraint(lambda x: min_objective_ucb - self.lcb(x,0),0,jnp.inf)
        cons.append(con)

        # Differential Evolution
        result = differential_evolution(obj_fun,self.bound,constraints=cons)

        return result.x, result.fun


        

        

            

