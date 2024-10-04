import copy
import jax 
import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from models.GP_Safe import GP
jax.config.update("jax_enable_x64", True)


class BO(GP):
    def __init__(self,plant_system,bound,b,TR_parameters):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.b = b
        self.TR_parameters = TR_parameters
        self.GP_inference_jit = jit(self.GP_inference)
        
        self.safe_set_cons = []
        for i in range(1,self.n_fun):
            safe_con = NonlinearConstraint(lambda x: self.lcb(x,i),0.,jnp.inf)
            self.safe_set_cons.append(safe_con)

    def calculate_plant_outputs(self,x):
        plant_output            = []
        for plant in self.plant_system:
            plant_output.append(plant(x)) 

        return jnp.array(plant_output)
    
    def mean(self,x,i):
        GP_inference = self.GP_inference_jit(x,self.inference_datasets)
        value         = GP_inference[0][i]
        return value
    
    def ucb(self,x,i):
        GP_inference = self.GP_inference_jit(x,self.inference_datasets)
        mean, var = GP_inference[0][i], GP_inference[1][i]
        value = mean + self.b*jnp.sqrt(var)
        return value
    
    def lcb(self,x,i):
        GP_inference = self.GP_inference_jit(x,self.inference_datasets)
        mean, var = GP_inference[0][i], GP_inference[1][i]
        value = mean - self.b*jnp.sqrt(var)

        return value

    def minimize_obj_lcb(self,r,x_0):
        satisfied = False

        obj_fun = lambda x: self.lcb(x,0)
        safe_set_cons = copy.deepcopy(self.safe_set_cons)
        safe_set_cons.append(NonlinearConstraint(lambda x: jnp.linalg.norm(x-x_0),0,r))
        
        while not satisfied:
            result = differential_evolution(obj_fun,self.bound,constraints=safe_set_cons)
            for i in range(1,self.n_fun):
                lcb_value = self.lcb(result.x,i)
                if lcb_value < -0.001:
                    break
            
            if jnp.linalg.norm(result.x-x_0) > r:
                continue

            satisfied = True

        return result.x, result.fun

    def TR_constraint(self,x,x_0,r):
        return r - jnp.linalg.norm(x-x_0+1e-8)
    
    def update_TR(self,x_initial,x_new,radius,plant_oldoutput,plant_newoutput):
        # TR Parameters:
        r = radius
        r_max = self.TR_parameters['radius_max']
        r_red = self.TR_parameters['radius_red']
        r_inc = self.TR_parameters['radius_inc']
        rho_lb = self.TR_parameters['rho_lb']
        rho_ub = self.TR_parameters['rho_ub']

        # objective functions
        plant_obj_old = plant_oldoutput[0]
        plant_obj_new = plant_newoutput[0]
        GP_obj_old = self.GP_inference_jit(x_initial,self.inference_datasets)[0][0]
        GP_obj_new = self.GP_inference_jit(x_new,self.inference_datasets)[0][0]

        # Check plant constraints to update trust region
        for i in range(1,self.n_fun):
            if plant_newoutput[i] < -0.001:
                return x_initial, r*r_red
            else:
                pass

        rho = (plant_obj_new-plant_obj_old)/(GP_obj_new-GP_obj_old)

        if plant_obj_old < plant_obj_new:
            return x_initial, r*r_red
        else:
            pass

        if rho < rho_lb:
            return x_initial, r*r_red

        elif rho >= rho_lb and rho < rho_ub: 
            return x_new, r

        else: # rho >= rho_ub
            return x_new, min(r*r_inc,r_max)
        
