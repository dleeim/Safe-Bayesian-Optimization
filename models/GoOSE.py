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
        
        self.safe_set_cons = []
        for i in range(1, self.n_fun):
            safe_con = NonlinearConstraint(lambda x, i=i: self.lcb(x,i),0.,jnp.inf)
            self.safe_set_cons.append(safe_con)

    def lcb_constraint_min(self,x):
        lcb_values = []
        for i in range(1,self.n_fun):
            lcb_values.append(self.lcb(x,i))
        return max(lcb_values)

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
        if jnp.isnan(mean) or jnp.isnan(var):
            print(f"NaN detected in LCB at x={x}")
        elif jnp.isinf(mean) or jnp.isinf(var):
            print(f"inf detected in LCB at x={x}")
        return value
    
    def infnorm_mean_grad(self,x,i):
        mean_grad_jit = jit(grad(self.mean,argnums=0))
        grad_mean = mean_grad_jit(x,i)
        return jnp.max(jnp.abs(grad_mean))
    
    def maxmimize_infnorm_mean_grad(self,i):
        lcb_maxnorm_grad_jit = jit(self.infnorm_mean_grad)
        obj_fun = lambda x, i=i: -lcb_maxnorm_grad_jit(x,i)
        result = differential_evolution(obj_fun,self.bound,polish=False)
        return -result.fun
    
    def minimize_obj_lcb(self):
        obj_fun = lambda x: self.lcb(x,0)
        result = differential_evolution(obj_fun,self.bound,constraints=self.safe_set_cons,polish=False)

        return result.x,result.fun

    def Lipschitz_continuity_constraint(self,x,i,max_infnorm_mean_constraints):
        ucb_value = self.ucb(x[self.nx_dim:],i)
        value = ucb_value - max_infnorm_mean_constraints*jnp.linalg.norm(x[:self.nx_dim]-x[self.nx_dim:])  
        return value

    def Target(self):
        eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
        obj_fun = lambda x: self.lcb(x[self.nx_dim:],0)
        bound = jnp.vstack((self.bound,self.bound))

        safe_unsafe_cons = []
        for i in range(1,self.n_fun):
            safe_unsafe_cons.append(NonlinearConstraint(lambda x, i=i: self.lcb(x[:self.nx_dim],i),0.,jnp.inf))
        safe_unsafe_cons.append(NonlinearConstraint(lambda x: self.lcb_constraint_min(x[self.nx_dim:]),-jnp.inf,0.))

        # Find target for each constraint
        target = []
        lcb_target = []

        for index in range(1,self.n_fun):
            Lipschitz_continuity_constraint_jit = jit(self.Lipschitz_continuity_constraint)
            target_cons = copy.deepcopy(safe_unsafe_cons)
            max_infnorm_mean_constraints = self.maxmimize_infnorm_mean_grad(index)
            target_cons.append(NonlinearConstraint(lambda x, index=index: self.lcb(x[:self.nx_dim],index),0,eps))
            target_cons.append(NonlinearConstraint(lambda x, index=index: Lipschitz_continuity_constraint_jit(x,index,max_infnorm_mean_constraints),0.,jnp.inf))
            result = differential_evolution(obj_fun,bound,constraints=target_cons,polish=False,popsize=30)
            print(result.x,result.fun)
            target.append(result.x[self.nx_dim:])
            lcb_target.append(result.fun)
        
        # Find most uncertain expander
        min_lcb_target = min(lcb_target)
        min_target_index = lcb_target.index(min_lcb_target)
        argmin_x = target[min_target_index]

        return argmin_x,min_lcb_target

    def explore_safeset(self,target):
        obj_fun = lambda x: cdist(x.reshape(1,-1),target.reshape(1,-1))[0][0]
        result = differential_evolution(obj_fun,self.bound,constraints=self.safe_set_cons,polish=False)
        return result.x
        
