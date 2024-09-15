import time
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, random
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import cdist
import sobol_seq
from models.GP_Robust import GP

class BO(GP):
    def __init__(self,plant_system,bound,bound_d,b):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.bound_d = bound_d
        self.nxc_dim = bound.shape[0] # x controlled dimension number
        self.nd_dim = bound_d.shape[0] # d out of control dimension number
        self.b = b
        self.GP_inference_jit = jit(self.GP_inference) 
        self.max_safe_cons = []
        for i in range(1, self.n_fun):
            con = NonlinearConstraint(lambda xc: self.Maximise_d(self.lcb,xc,i),0.,jnp.inf)
            self.max_safe_cons.append(con) 

    def calculate_plant_outputs(self,x,d):
        plant_output            = []
        for plant in self.plant_system:
            plant_output.append(plant(x,d)) 

        return jnp.array(plant_output)
    
    def Data_sampling_with_perbutation(self,n_sample,x=None,r=None):
        if x==None and r==None:
            fraction_x = sobol_seq.i4_sobol_generate(self.nxc_dim,n_sample)
            fraction_d = sobol_seq.i4_sobol_generate(self.nd_dim,n_sample)

            xc_samples = fraction_x*(self.bound[:,1]-self.bound[:,0])+self.bound[:,0]
            d_samples = fraction_d*(self.bound_d[:,1]-self.bound_d[:,0])+self.bound_d[:,0] 

            plant_output = self.calculate_plant_outputs(xc_samples,d_samples)[0]
            x_samples = jnp.hstack((xc_samples,d_samples))
        
        return x_samples,plant_output
    
    def xc0_sampling(self,n_sample):
        # Need to sample initial xc points that satisfy constraints. 
        xc0_samples = []
        i = -1
        while len(xc0_samples) < n_sample:
            i = i+1
            fraction_x = sobol_seq.i4_sobol_generate(self.nxc_dim,1,i)
            xc = fraction_x*(self.bound[:,1]-self.bound[:,0])+self.bound[:,0]
            xc = xc.flatten()
            if self.max_safe_cons == []:
                xc0_samples.append(xc)
                continue

            for con in self.max_safe_cons:
                if con(xc) >= 0.:
                    break 
                
                if con == self.max_safe_cons[-1]:
                    xc0_samples.append(xc)

        return xc0_samples
    
    def d0_sampling(self,n_sample):
        # Need to sample initial d points that satisfy constraints. 
        d0_samples = []
        i = -1
        while len(d0_samples) < n_sample:
            i = i+1
            fraction_d = sobol_seq.i4_sobol_generate(self.nd_dim,1,i)
            d = fraction_d*(self.bound_d[:,1]-self.bound_d[:,0])+self.bound_d[:,0]
            d = d.flatten()
            if self.max_safe_cons == []:
                d0_samples.append(d)
                continue

            for con in self.max_safe_cons:
                if con(d) >= 0.:
                    break 
                
                if con == self.max_safe_cons[-1]:
                    d0_samples.append(d)
        return d0_samples

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
        # xc = xc+jnp.finfo(jnp.float64).eps
        # d = d+jnp.finfo(jnp.float64).eps
        if xc.ndim != 1 or d.ndim != 1:
            raise ValueError("xc or d needs to be in 1d")
        
        x = jnp.concatenate((xc,d))
        GP_inference = self.GP_inference_jit(x,self.inference_datasets)
        mean, var = GP_inference[0][i], GP_inference[1][i]
        value = mean - self.b*jnp.sqrt(var)
        return value
    
    def lcb_grad(self,xc,d,i):
        lcb_grad = grad(self.lcb,argnums=1)
        value = lcb_grad(xc,d,i)
        return value
    
    def Maximise_d(self,fun,xc,i):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")

        obj_fun = lambda d: -fun(xc,d,i) # negative sign for maximization
        n_start = 5
        max_val = -jnp.inf
        fun_grad_jit = jit(grad(fun,argnums=1))
        obj_grad = lambda d: fun_grad_jit(xc,d,i)
        for x0 in jnp.linspace(self.bound_d[:,0],self.bound_d[:,1],n_start):
            res = minimize(obj_fun,x0,bounds=self.bound_d,jac=obj_grad,method='SLSQP')

            if -res.fun > max_val:
                res_best = res
                max_val = -res.fun
    
        return max_val

    def Minimize_Maximise(self,fun,xc0_sample): # NEED TO FILTER OUT START VALUE TO BE IN SAFE SET
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        

        obj_fun = lambda xc: self.Maximise_d(fun,xc,0)
        n_start = 1
        min_val = jnp.inf
        for xc0 in xc0_sample:
            res = minimize(obj_fun,xc0,method='COBYQA',bounds=self.bound,constraints=self.max_safe_cons)
    
            if res.fun == jnp.nan:
                res = differential_evolution(xc0,self.bound,constraints=self.max_safe_cons)
            
            if res.fun < min_val:
                res_best = res
                min_val = res.fun
        
        return res_best.x, res_best.fun
    
    def Maximise_d_with_constraints(self,fun,xc):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        obj_fun = lambda d: -1*fun(xc,d,0)
        res_best = differential_evolution(obj_fun,self.bound_d,constraints=self.max_safe_cons)
        return res_best.x, -res_best.fun
        
