import random
import time
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import cdist
import sobol_seq
from models.GP_Safe import GP
jax.config.update("jax_enable_x64", True)

class BO(GP):
    def __init__(self,plant_system,bound,b):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.b = b
        self.GP_inference_jit = jit(self.GP_inference)  

    def calculate_plant_outputs(self,x,noise=0):
        plant_output            = []
        for plant in self.plant_system:
            plant_output.append(plant(x,noise)) 

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
    
    def minimize_obj_ucb(self,safe_set_cons):
        obj_fun = lambda x: self.ucb(x,0)
        result = differential_evolution(obj_fun,self.bound,constraints=safe_set_cons,polish=False)

        return result.x, result.fun
    
    def Minimizer(self):
        # Setting Optimization Problem
        obj_fun = lambda x: -self.GP_inference_jit(x,self.inference_datasets)[1][0] # objective function is -variance as differential equation finds min (convert to max)
        
        Minimizer_cons = []
        for i in range(1, self.n_fun):
            Minimizer_cons.append(NonlinearConstraint(lambda x, i=i: self.lcb(x,i),0.,jnp.inf)) 
             
        x_min, min_obj_ucb = self.minimize_obj_ucb(safe_set_cons=Minimizer_cons)
        Minimizer_cons.append(NonlinearConstraint(lambda x: min_obj_ucb - self.lcb(x,0),0,jnp.inf))

        # Differential Evolution
        result = differential_evolution(obj_fun,self.bound,constraints=Minimizer_cons,polish=False)

        return result.x, jnp.sqrt(-result.fun)
    
    def infnorm_mean_grad(self,x,i):
        mean_grad_jit = grad(self.mean,argnums=0)
        grad_mean = mean_grad_jit(x,i)
        return jnp.max(jnp.abs(grad_mean))
    
    def maxmimize_infnorm_mean_grad(self,i):
        lcb_maxnorm_grad_jit = jit(self.infnorm_mean_grad)
        infnorm_mean_constraints = []
        
        for i in range(1,self.n_fun):
            obj_fun = lambda x: -lcb_maxnorm_grad_jit(x,i)
            result = differential_evolution(obj_fun,self.bound,polish=False)
            infnorm_mean_constraints.append(-result.fun)
        max_infnorm_mean_constraints = max(infnorm_mean_constraints)
        
        return max_infnorm_mean_constraints
    
    def lcb_constraint_min(self,x):
        lcb_values = []
        for i in range(1,self.n_fun):
            lcb_values.append(self.lcb(x,i))
        return max(lcb_values)

    def Lipschitz_continuity_constraint(self,x,i,maximum_infnorm_mean_constraints):
        ucb_value = self.ucb(x[:self.nx_dim],i)    
        value = ucb_value - maximum_infnorm_mean_constraints*jnp.linalg.norm(x[:self.nx_dim]-x[self.nx_dim:])  
        return value

    def Expander(self):
        eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
        obj_fun = lambda x: -self.GP_inference_jit(x[:self.nx_dim],self.inference_datasets)[1][0] # objective function is -1*variance as differential equation finds min (convert to max)
        bound = jnp.vstack((self.bound,self.bound)) 
        
        safe_unsafe_cons = []
        for i in range(1, self.n_fun):
            safe_unsafe_cons.append(NonlinearConstraint(lambda x, i=i: self.lcb(x[:self.nx_dim],i)*self.Y_mean[0]/self.Y_mean[i],0.,jnp.inf)) # x[:self.nx_dim] represents input variable in safe region to be optimized for expander.
        safe_unsafe_cons.append(NonlinearConstraint(lambda x: self.lcb_constraint_min(x[self.nx_dim:]),-jnp.inf,0.)) # x[self.nx_dim:] represents represents variable in unsafe region to be used for Lipschitz Continuity
        
        infnorm_mean_grad = jit(self.infnorm_mean_grad)

        # Find expander for each constraint
        expanders = []
        std_expanders = []

        for index in range(1,self.n_fun):
            Lipschitz_continuity_constraint_jit = jit(self.Lipschitz_continuity_constraint)
            Expander_cons = copy.deepcopy(safe_unsafe_cons)
            maximum_infnorm_mean_constraints = self.maxmimize_infnorm_mean_grad(index)
            Expander_cons.append(NonlinearConstraint(lambda x, index=index: self.lcb(x[:self.nx_dim],index),0,eps))
            Expander_cons.append(NonlinearConstraint(lambda x, index=index: Lipschitz_continuity_constraint_jit(x,index,maximum_infnorm_mean_constraints),0.,jnp.inf))    
            result = differential_evolution(obj_fun,bound,constraints=Expander_cons,polish=False)
            
            # Collect optimal point and standard deviation
            expanders.append(result.x[:self.nx_dim])
            std_expanders.append(jnp.sqrt(-result.fun))

        # Find most uncertain expander
        max_std = max(std_expanders)
        max_index = std_expanders.index(max_std)
        argmax_x = expanders[max_index]

        lcb_constraint = []
        for index in range(1,self.n_fun):
            lcb_constraint.append(self.lcb(argmax_x,index))

        return argmax_x, max_std
