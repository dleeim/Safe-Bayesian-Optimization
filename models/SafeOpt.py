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

    def Expander_constraint(self,x,unsafe_sobol_sample):
        # Initialization
        lcb_maxnorm_grad_jit    = jit(self.maxnorm_mean_grad)
        eps                     = jnp.sqrt(jnp.finfo(jnp.float32).eps)
        boundary                = False
        indicator               = 0.
        n_constraints           = list(range(1, self.n_fun))
        random.shuffle(n_constraints)
        print(x)
        for i in n_constraints:
            if self.lcb(x,i) <= eps:
                boundary        = True
                index           = i
                break
        
        if boundary == False:
            print(indicator)
            return indicator
        
        for sobol_point in unsafe_sobol_sample:

            if self.ucb(x,index) - lcb_maxnorm_grad_jit(x,index)*cdist(x.reshape(1, -1),sobol_point.reshape(1, -1)) >= 0.:
                indicator = 10.
                print(indicator)

                if jnp.any(unsafe_sobol_sample[:50] == sobol_point):
                    pass
                else:
                    index = jnp.where(sobol_point == unsafe_sobol_sample)[0][0]
                    unsafe_sobol_sample = jnp.vstack((sobol_point,unsafe_sobol_sample[:index],unsafe_sobol_sample[index+1:]))

                return indicator
        print(indicator)
        return indicator
    
    def Expander(self,unsafe_sobol_sample):
        eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
        obj_fun = lambda x: -self.GP_inference_jit(x,self.inference_datasets)[1][0] # objective function is -variance as differential equation finds min (convert to max)
        cons = copy.deepcopy(self.safe_set_cons)
        cons.append(NonlinearConstraint(lambda x: self.Expander_constraint(x,unsafe_sobol_sample),10.,jnp.inf))
        result = differential_evolution(obj_fun,self.bound,constraints=cons,tol=eps)
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



