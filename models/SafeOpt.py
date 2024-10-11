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
        self.mean_grad_jit = jit(grad(self.mean,argnums=0))
        self.safe_set_cons = []
        for i in range(1, self.n_fun):
            con = NonlinearConstraint(lambda x: self.lcb(x,i),0.,jnp.inf)
            self.safe_set_cons.append(con)        

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
    
    def minimize_obj_ucb(self):
        obj_fun = lambda x: self.ucb(x,0)
        result = differential_evolution(obj_fun,self.bound,constraints=self.safe_set_cons)

        return result.x, result.fun
    
    def Minimizer(self):
        # Setting Optimization Problem
        obj_fun = lambda x: -self.GP_inference_jit(x,self.inference_datasets)[1][0] # objective function is -variance as differential equation finds min (convert to max)
        cons = copy.deepcopy(self.safe_set_cons)
        x_min, min_obj_ucb = self.minimize_obj_ucb()
        con = NonlinearConstraint(lambda x: min_obj_ucb - self.lcb(x,0),0,jnp.inf)
        cons.append(con)

        # Differential Evolution
        result = differential_evolution(obj_fun,self.bound,constraints=cons)

        return result.x, jnp.array(-result.fun)
    
    def maxnorm_mean_grad(self,x,i):
        grad_mean = self.mean_grad_jit(x,i)
        return jnp.max(jnp.abs(grad_mean))
    
    def unsafe_sobol_seq_sampling(self,x_dim,n_sample,bound,skip=0):
        # Create sobol_seq sample for unsafe region
        fraction = sobol_seq.i4_sobol_generate(x_dim,n_sample,skip)
        lb = bound[:,0]
        ub = bound[:,1]
        sample = lb + (ub-lb) * fraction

        # Filter to create sample in unsafe region
        lcb_vmap = vmap(self.lcb,in_axes=(0,None))
        for i in range(1,self.n_fun):
            mask_unsafe = lcb_vmap(jnp.array(sample),i) < 0.
            sample = sample[mask_unsafe]

        return jnp.array(sample)
    
    def maxmimize_maxnorm_mean_grad(self):
        lcb_maxnorm_grad_jit                = jit(self.maxnorm_mean_grad)
        maximum_maxnorm_mean_constraints    = []
        for i in range(1,self.n_fun):
            obj_fun = lambda x: -lcb_maxnorm_grad_jit(x,i)
            result = differential_evolution(obj_fun,self.bound,tol=0.1)
            maximum_maxnorm_mean_constraints.append(-result.fun)

        return maximum_maxnorm_mean_constraints

    def Expander_constraint(self,x,unsafe_sobol_sample,maximum_maxnorm_mean_constraints,eps):
        min_val = jnp.inf
        lcb_non_expander = []
        for i in range(1,self.n_fun):
            lcb_value = self.lcb(x,i)

            if lcb_value <= eps and lcb_value >= 0.:
                for j in range(len(unsafe_sobol_sample)):
                    sobol_point = unsafe_sobol_sample[j]
                    value = self.ucb(x,i) - maximum_maxnorm_mean_constraints*cdist(x.reshape(1, -1),sobol_point.reshape(1, -1))

                    if value >= 0.:
                        if j > int(0.05*len(unsafe_sobol_sample)):
                            unsafe_sobol_sample = jnp.vstack((sobol_point,unsafe_sobol_sample[:j],unsafe_sobol_sample[j+1:]))
                        return lcb_value
                    else:
                        if value < min_val:
                            min_val = value.item()
            else:
                lcb_non_expander.append(-1*lcb_value)
            
        if len(lcb_non_expander)==0:
            return min_val
        else:
            lcb_value = max(lcb_non_expander,key=abs)
            if abs(lcb_value)>abs(min_val) or min_val==jnp.inf:
                return lcb_value
            else:
                return min_val

    # def Expander_constraint(self,x,unsafe_sobol_sample,maximum_maxnorm_mean_constraints,eps):
    #     max_lcb = -jnp.inf
    #     min_lcb = jnp.inf
    #     max_val = -jnp.inf

    #     for i in range(1,self.n_fun):
    #         lcb_value = self.lcb(x,i)

    #         if lcb_value > eps and lcb_value > max_lcb:
    #             max_lcb = lcb_value
    #         elif lcb_value < 0 and lcb_value < min_lcb:
    #             min_lcb = lcb_value

    #         else:
    #             for j in range(len(unsafe_sobol_sample)):
    #                 sobol_point = unsafe_sobol_sample[j]
    #                 value = self.ucb(x,i) - maximum_maxnorm_mean_constraints*cdist(x.reshape(1, -1),sobol_point.reshape(1, -1))

    #                 if value >= 0.:
    #                     if j > int(0.05*len(unsafe_sobol_sample)):
    #                         unsafe_sobol_sample = jnp.vstack((sobol_point,unsafe_sobol_sample[:j],unsafe_sobol_sample[j+1:]))
    #                     return lcb_value
    #                 else:
    #                     if value > max_val:
    #                         max_val = value.item()
        
    #     values = jnp.array([max_lcb,min_lcb,max_val])
    #     values = values[jnp.isfinite(values)]
    #     print(max(values,key=abs))
    #     return max(values,key=abs)
    
    def Expander(self,unsafe_sobol_sample,maximum_maxnorm_mean_constraints):
        eps = jnp.sqrt(jnp.finfo(jnp.float64).eps)
        obj_fun = lambda x: -self.GP_inference_jit(x,self.inference_datasets)[1][0] # objective function is -variance as differential equation finds min (convert to max)
        cons = []
        cons.append(NonlinearConstraint(lambda x: self.Expander_constraint(x,unsafe_sobol_sample,maximum_maxnorm_mean_constraints,eps),0.,eps))
        satisfied = False
        while not satisfied:
            result = differential_evolution(obj_fun,self.bound,constraints=cons)
            for i in range(1, self.n_fun):
                lcb_value = self.lcb(result.x,i)
                
                if lcb_value < 0.:
                    break
            
            satisfied = True
        return result.x, jnp.sqrt(-result.fun)
    
    # Overall Algorithm
    def Safeminimize(self,n_sample,x_initial,radius,n_iter):

        # Initialization
        X,Y = self.Data_sampling(n_sample,x_initial,radius)
        self.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)

        # SafeOpt
        for i in range(n_iter):
            # Create sobol_seq sample for Expander
            n_sample = 1000
            unsafe_sobol_sample = self.unsafe_sobol_seq_sampling(self.nx_dim,n_sample,self.bound)
            minimizer,std_minimizer = self.Minimizer()
            expander,std_expander = self.Expander(unsafe_sobol_sample)

            if std_minimizer > std_expander:
                x_new = minimizer
            else:
                x_new = expander
        
            plant_output = self.calculate_plant_outputs(x_new)
            self.add_sample(x_new,plant_output)
        
        return x_new, plant_output



