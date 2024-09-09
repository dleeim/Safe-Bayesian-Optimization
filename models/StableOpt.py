import time
import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, random
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import cdist
import sobol_seq
from models.GP_Safe import GP

class BO(GP):
    def __init__(self,plant_system,bound,bound_d,b):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.bound_d = bound_d
        self.nxc_dim = bound.shape[0] # x controlled dimension number
        self.nd_dim = bound_d.shape[0] # d out of control dimension number
        self.b = b
        self.xc_min = []
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
        lcb_grad = grad(self.lcb)
        value = lcb_grad(xc,d,i)
        return value
    
    def Maximise_d(self,fun,xc,i):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")

        obj_fun = lambda d: -fun(xc,d,i) # negative sign for maximization
        n_start = 5
        max_val = -jnp.inf
        fun_grad_jit = jit(grad(fun))
        obj_grad = lambda d: fun_grad_jit(xc,d,i)
        for x0 in jnp.linspace(self.bound_d[:,0],self.bound_d[:,1],n_start):
            res = minimize(obj_fun,x0,bounds=self.bound_d,jac=obj_grad,method='SLSQP')
            if -res.fun > max_val:
                res_best = res
                max_val = -res.fun

        return -res_best.fun
        
        # res = differential_evolution(obj_fun,bounds=self.bound_d,tol=1e-3)
        # print(-res.fun)
        # return -res.fun

    def Minimize_Maximise(self,fun):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        
        n_start = 1
        min_val = jnp.inf
        obj_fun = lambda xc: self.Maximise_d(fun,xc,0)
        cons = []
        for i in range(1,self.n_fun):
            cons.append(NonlinearConstraint(lambda xc: self.Maximise_d(self.lcb,xc,i),-jnp.inf,0))
        
        for x0 in jnp.linspace(self.bound_d[:,0],self.bound_d[:,1],n_start):
            res = minimize(obj_fun,x0,method='COBYQA',bounds=self.bound,constraints=cons)
            if res.fun < min_val:
                res_best = res
                min_val = res.fun
        # res_best = differential_evolution(obj_fun,bounds=self.bound,constraints=cons)
        self.xc_min.append(res_best.x)
        return res_best.x, res_best.fun
    
    def Maximise_d_with_constraints(self,fun,x):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        
        obj_fun = lambda d: -fun(x,d,0)
        cons = []
        for i in range(1,self.n_fun):
            cons.append(NonlinearConstraint(lambda x: self.robust_filter(self.lcb,x,i),-jnp.inf,0))

        res = differential_evolution(obj_fun,bounds=self.bound_d,constraints=cons)

        return res.x, -res.fun
    
    def Best_xc_Guess(self,fun):
        output_min = jnp.inf

        for i in self.xc_min:
            dmax, output = self.Maximise_d_with_constraints(fun,i)
            print(f"x, dmax, ucb output {i,dmax,output}")
            if output < output_min:
                output_min = output
                xc_min_of_min = i
                d = dmax
        
        return xc_min_of_min,d
        
