import time
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, random
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import cdist
import sobol_seq
from models.GP_Classic import GP

class BO(GP):
    def __init__(self,plant_system,bound,bound_d,b):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.bound_d = bound_d
        self.nxc_dim = bound.shape[0] # x controlled dimension number
        self.nd_dim = bound_d.shape[0] # d out of control dimension number
        self.b = b
        self.GP_inference_jit = jit(self.GP_inference) 

    def calculate_plant_outputs(self,x,d):
        plant_output            = []
        for plant in self.plant_system:
            plant_output.append(plant(x,d)) 

        return jnp.array(plant_output)
    
    def Data_sampling_with_perbutation(self,n_sample,x=None,r=None):
        if x==None and r==None:
            samples = random.uniform(self.key, shape=(n_sample,1))
            xc_samples = samples*(self.bound[:,1]-self.bound[:,0])+self.bound[:,0]
            d_samples = samples*(self.bound_d[:,1]-self.bound_d[:,0])+self.bound_d[:,0] 
            plant_output = self.calculate_plant_outputs(xc_samples,d_samples)[0]
            x_samples = jnp.hstack((xc_samples,d_samples))
        
        return x_samples,plant_output

    def mean(self,xc,d,i):
        if xc.ndim != 1 or d.ndim != 1:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        
        x = jnp.concatenate((xc,d))
        GP_inference = self.GP_inference_jit(x,self.inference_datasets)
        mean, var = GP_inference[0][i], GP_inference[1][i] 
        return mean

    def ucb(self,xc,d,i):
        if xc.ndim != 1 or d.ndim != 1:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        
        x = jnp.concatenate((xc,d))
        GP_inference = self.GP_inference_jit(x,self.inference_datasets)
        mean, var = GP_inference[0][i], GP_inference[1][i]
        value = mean + self.b*jnp.sqrt(var)
        return value
    
    def lcb(self,xc,d,i):
        if xc.ndim != 1 or d.ndim != 1:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        
        x = jnp.concatenate((xc,d))
        GP_inference = self.GP_inference_jit(x,self.inference_datasets)
        mean, var = GP_inference[0][i], GP_inference[1][i]
        value = mean - self.b*jnp.sqrt(var)
        return value
    
    def robust_filter(self,fun,x,i):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
    
        obj_fun = lambda d: -fun(x,d,i) # negative sign for maximization
        res = differential_evolution(obj_fun,bounds=self.bound_d)
        return -res.fun

    def Minimize_Robust(self):
        obj_fun = lambda x: self.robust_filter(self.lcb,x,0)
        cons = []
        for i in range(1,self.n_fun):
            cons.append(NonlinearConstraint(lambda x: self.robust_filter(self.lcb,x,i),-jnp.inf,0))
        
        res = differential_evolution(obj_fun,bounds=self.bound,constraints=cons)

        return res.x, res.fun
    
    def Maximise_d(self,fun,x):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        
        obj_fun = lambda d: -fun(x,d,0)
        cons = []
        for i in range(1,self.n_fun):
            cons.append(NonlinearConstraint(lambda x: self.robust_filter(self.lcb,x,i),-jnp.inf,0))

        res = differential_evolution(obj_fun,bounds=self.bound_d,constraints=cons)

        return res.x
