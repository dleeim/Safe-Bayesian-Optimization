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

class BO(GP):
    def __init__(self,plant_system,bound,b):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.b = b
        self.GP_inference_jit = jit(self.GP_inference)
        self.mean_grad_jit = jit(grad(self.mean,argnums=0))
        
        self.safe_set_cons = []
        self.unsafe_set_cons = []
        for i in range(1, self.n_fun):
            safe_con = NonlinearConstraint(lambda x: self.lcb(x,i),0.,jnp.inf)
            self.safe_set_cons.append(safe_con)
            unsafe_con = NonlinearConstraint(lambda x: self.lcb(x,i),-jnp.inf,0.)
            self.unsafe_set_cons.append(unsafe_con)


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
    
    def safe_sobol_seq_sampling(self,x_dim,n_sample,bound,skip=0):
        # Create sobol_seq sample for unsafe region
        fraction = sobol_seq.i4_sobol_generate(x_dim,n_sample,skip)
        lb = bound[:,0]
        ub = bound[:,1]
        sample = lb + (ub-lb) * fraction

        # Filter to create sample in unsafe region
        lcb_vmap = vmap(self.lcb,in_axes=(0,None))
        for i in range(1,self.n_fun):
            mask_unsafe = lcb_vmap(jnp.array(sample),i) >= 0.
            sample = sample[mask_unsafe]

        return jnp.array(sample)
    
    def maxnorm_mean_grad(self,x,i):
        grad_mean = self.mean_grad_jit(x,i)
        return jnp.max(jnp.abs(grad_mean))
    
    def maxmimize_maxnorm_mean_grad(self):
        lcb_maxnorm_grad_jit                = jit(self.maxnorm_mean_grad)
        maximum_maxnorm_mean_constraints    = []
        for i in range(1,self.n_fun):
            obj_fun = lambda x: -lcb_maxnorm_grad_jit(x,i)
            result = differential_evolution(obj_fun,self.bound,tol=0.1)
            maximum_maxnorm_mean_constraints.append(-result.fun)

        return maximum_maxnorm_mean_constraints
    
    def optimistic_safeset_condition(self,x,i,sobol_point,maximum_maxnorm_mean_constraint):
        V = jnp.array([1.]*self.nx_dim)
        distance = jnp.sqrt(self.squared_seuclidean_jax(x.reshape(1,-1),sobol_point.reshape(1,-1),V))
        
        return self.ucb(sobol_point,i) - maximum_maxnorm_mean_constraint*distance
    
    def optimistic_safeset_constraint(self,x,safe_sobol_sample,maximum_maxnorm_mean_constraints):
        # Initialization
        optimistic_safeset_condition_vmap = jit(vmap(self.optimistic_safeset_condition,in_axes=(None,None,0,None,)))
        indicator                       = 0.
        n_constraints                   = list(range(1, self.n_fun))
        random.shuffle(n_constraints)

        for i in n_constraints:
            maximum_maxnorm_mean_constraint = maximum_maxnorm_mean_constraints[i-1]
            condition = optimistic_safeset_condition_vmap(x,i,safe_sobol_sample,maximum_maxnorm_mean_constraint)
            satisfied = jnp.any(condition >= 0.).astype(int)

            if satisfied:
                indicator = 10.
                return indicator
            else:
                pass

        return indicator
    
    def minimize_obj_lcb(self):
        obj_fun = lambda x: self.ucb(x,0)
        result = differential_evolution(obj_fun,self.bound,constraints=self.safe_set_cons)

        return result.x,result.fun

    def Target(self,safe_sobol_sample,maximum_maxnorm_mean_constraints):
        obj_fun = lambda x: self.lcb(x,0)
        cons = copy.deepcopy(self.unsafe_set_cons)
        cons.append(NonlinearConstraint(lambda x:self.optimistic_safeset_constraint(x,safe_sobol_sample,maximum_maxnorm_mean_constraints),10.,jnp.inf))
        result = differential_evolution(obj_fun,self.bound,constraints=cons,tol=0.1)

        return result.x, result.fun

    def explore_safeset(self,target):
        obj_fun = lambda x: cdist(x.reshape(1,-1),target.reshape(1,-1))[0][0]
        cons = copy.deepcopy(self.safe_set_cons)
        result = differential_evolution(obj_fun,self.bound,constraints=cons,tol=0.1)
        # ALERT!!!!!!!!!: Need to add constraint that g(x,z) > 0 when finding closest distance..
        return result.x
        
