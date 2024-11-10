import time
import random
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import cdist
import sobol_seq
from models.GP_Robust import GP

jax.config.update("jax_enable_x64", True)

class BO(GP):
    def __init__(self,plant_system,bound,bound_d,b):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.bound_d = bound_d
        self.nxc_dim = bound.shape[0] # x controlled dimension number
        self.nd_dim = bound_d.shape[0] # d out of control dimension number
        self.b = b
        self.GP_inference_jit = jit(self.GP_inference) 

    def calculate_plant_outputs(self,x,noise=0):
        plant_output            = []
        disturbance             = 0.
        for plant in self.plant_system:
            output,disturbance = plant(x,noise)
            plant_output.append(output) 

        return jnp.array(plant_output), disturbance
    
    def Data_sampling_with_perbutation(self,n_sample,x=None,r=None):
        if x==None and r==None:
            fraction_x = sobol_seq.i4_sobol_generate(self.nxc_dim,n_sample)
            fraction_d = sobol_seq.i4_sobol_generate(self.nd_dim,n_sample)

            xc_samples = fraction_x*(self.bound[:,1]-self.bound[:,0])+self.bound[:,0]
            d_samples = fraction_d*(self.bound_d[:,1]-self.bound_d[:,0])+self.bound_d[:,0] 

            plant_output = self.calculate_plant_outputs(xc_samples,d_samples)[0]
            x_samples = jnp.hstack((xc_samples,d_samples))
        
        return x_samples,plant_output

    def Data_sampling_output_and_perturbation(self,n_sample,x_0,r,noise=0.):
        # === Collect Training Dataset (Input) === #
        self.key, subkey            = jax.random.split(self.key) 
        x_dim                       = jnp.shape(x_0)[0]                           
        X                           = self.Ball_sampling(x_dim,n_sample,r,subkey)
        X                           += x_0

        # === Collect Training Dataset === #
        n_fun                       = len(self.plant_system)
        Y                           = jnp.zeros((n_sample,n_fun))
        D                           = jnp.zeros((n_sample,len(self.bound_d)))

        for i in range(len(X)):
            for j in range(n_fun):
                y, d                = self.plant_system[j](X[i],noise)
                Y                   = Y.at[i,j].set(y)
            D                       = D.at[i].set(d)

        return X,Y,D

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
    
    def Maximise_d(self,fun,xc,i):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")

        obj_fun = lambda d, i=i: -fun(xc,d,i) # negative sign for maximization
        n_start = 5
        max_val = -jnp.inf

        for x0 in jnp.linspace(self.bound_d[:,0],self.bound_d[:,1],n_start):
            res = minimize(obj_fun,x0,bounds=self.bound_d,jac='3-point',method='SLSQP')

            if -res.fun > max_val:
                max_val = -res.fun
        
        if max_val == jnp.inf or max_val == np.inf:
            res = differential_evolution(obj_fun,bounds=self.bound_d)
            max_val = -res.fun

        # print(xc,i,max_val)
        return max_val
    
    def Minimise_d(self,fun,xc,i):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")

        obj_fun = lambda d, i=i: fun(xc,d,i)
        n_start = 5
        min_val = jnp.inf

        for x0 in jnp.linspace(self.bound_d[:,0],self.bound_d[:,1],n_start):
            res = minimize(obj_fun,x0,bounds=self.bound_d,jac='3-point',method='SLSQP')

            if res.fun < min_val:
                min_val = res.fun
        
        if min_val == jnp.inf or min_val == np.inf:
            res = differential_evolution(obj_fun,bounds=self.bound_d)
            min_val = res.fun

        # print(xc,i,min_val)
        return min_val

    def Minimize_Maximise(self,fun): # NEED TO FILTER OUT START VALUE TO BE IN SAFE SET
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        max_safe_cons = []

        for index in range(1, self.n_fun):
            con = NonlinearConstraint(lambda xc, index=index: self.Minimise_d(self.lcb,xc,index),0.,jnp.inf)
            max_safe_cons.append(con) 

        obj_fun = lambda xc: self.Maximise_d(fun,xc,0)

        res = differential_evolution(obj_fun,self.bound,constraints=max_safe_cons,polish=False)
        
        return res.x, res.fun
    
    def Maximise_d_with_constraints(self,fun,xc):
        if fun not in [self.ucb, self.lcb, self.mean]:
            raise ValueError("fun needs to be either self.ucb, lcb or mean")
        max_safe_cons = []
        for index in range(1, self.n_fun):
            con = NonlinearConstraint(lambda d, index=index: self.lcb(xc,d,index),0.,jnp.inf)
            max_safe_cons.append(con) 
        obj_fun = lambda d: -fun(xc,d,0)
        # res = differential_evolution(obj_fun,self.bound_d,constraints=max_safe_cons,polish=False)
        res = differential_evolution(obj_fun,self.bound_d,polish=False,popsize=100)
        
        return res.x, -res.fun
