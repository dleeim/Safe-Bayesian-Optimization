import random
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import cdist
import sobol_seq
from models.GP_SafeOpt import GP

class SafeOpt(GP):
    def __init__(self,plant_system,bound,b):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.b = b
        self.GP_inference_jit = jit(self.GP_inference)
        self.GP_inference_arb_jit = jit(self.GP_inference_arb)
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
    
    def lcb_arb(self,x,i):
        GP_inference = self.GP_inference_arb_jit(x)
        mean, var = GP_inference[0][i], GP_inference[1][i]
        value = mean - self.b*jnp.sqrt(var)
        return value
    
    def minimize_obj_ucb(self):
        obj_fun = lambda x: self.ucb(x,0)
        result = differential_evolution(obj_fun,self.bound,constraints=self.safe_set_cons)

        return result.x, result.fun
    
    def Minimizer(self):
        # Setting Optimization Problem
        obj_fun = lambda x: -self.GP_inference_jit(x)[1][0] # objective function is -variance as differential equation finds min (convert to max)
        cons = self.safe_set_cons
        x_min, min_obj_ucb = self.minimize_obj_ucb()
        con = NonlinearConstraint(lambda x: min_obj_ucb - self.lcb(x,0),0,jnp.inf)
        cons.append(con)

        # Differential Evolution
        result = differential_evolution(obj_fun,self.bound,constraints=cons)

        return result.x, jnp.array(-result.fun)

    def create_point_arb(self,x):
        point_arb            = []
        for i in range(self.n_fun):
            point_arb.append(self.ucb(x,i)) 

        return jnp.array(point_arb)
    
    def Expander_constraint(self,x):
        point_arb = self.create_point_arb(x)
        self.create_GP_arb(x,point_arb)
        indicator = 0.
        # Create random list of numbers 
        numbers = list(range(1, self.n_fun))
        random.shuffle(numbers)

        for i in numbers:
            # Create unsafe constraint i lcb
            con = NonlinearConstraint(lambda x: self.lcb(x,i),-jnp.inf,0)
            obj_fun = lambda x: -self.lcb_arb(x,i) # Find max of constraint lcb of aribitrary GP in unsafe zone of main GP
            
            result = differential_evolution(obj_fun,self.bound,constraints=con)
            print(result.x,-result.fun)
            if -result.fun > 0.:
                indicator = 1.
                break

        return indicator
    
    def Expander(self):
        obj_fun = lambda x: -self.GP_inference_jit(x)[1][0] # objective function is -variance as differential equation finds min (convert to max)
        cons = self.safe_set_cons
        cons.append(NonlinearConstraint(lambda x: self.Expander_constraint(x),0.,jnp.inf))

        result = differential_evolution(obj_fun,self.bound,constraints=cons)

        return result.x, jnp.sqrt(-result.fun)

        

        

            

