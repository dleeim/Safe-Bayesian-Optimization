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
        result = differential_evolution(obj_fun,self.bound,constraints=safe_set_cons)

        return result.x, result.fun
    
    def Minimizer(self):
        # Setting Optimization Problem
        obj_fun = lambda x: -self.GP_inference_jit(x,self.inference_datasets)[1][0] # objective function is -variance as differential equation finds min (convert to max)
        
        Minimizer_cons = []
        for i in range(1, self.n_fun):
            Minimizer_cons.append(NonlinearConstraint(lambda x: self.lcb(x,i),0.,jnp.inf)) 
             
        x_min, min_obj_ucb = self.minimize_obj_ucb(safe_set_cons=Minimizer_cons)
        Minimizer_cons.append(NonlinearConstraint(lambda x: min_obj_ucb - self.lcb(x,0),0,jnp.inf))

        # Differential Evolution
        result = differential_evolution(obj_fun,self.bound,constraints=Minimizer_cons)

        return result.x, jnp.array(-result.fun)
    
    def infnorm_mean_grad(self,x,i):
        grad_mean = self.mean_grad_jit(x,i)
        return jnp.max(jnp.abs(grad_mean))
    
    def maximize_infnorm_mean_grad(self,i):
        lcb_infnorm_grad_jit = jit(self.infnorm_mean_grad)
        obj_fun = lambda x: -lcb_infnorm_grad_jit(x,i)
        result = differential_evolution(obj_fun,self.bound)
        return -result.fun

    def Lipschitz_continuity_constraint(self,x,i,max_infnorm_mean_constraints):
        ucb_value = self.ucb(x[:self.nx_dim],i)       
        value = ucb_value - max_infnorm_mean_constraints*cdist(x[:self.nx_dim].reshape(1,-1),x[self.nx_dim:].reshape(1,-1))   
        return value.item()
    
    def lcb_constraint_min(self,x):
        lcb_values = []
        for i in range(1,self.n_fun):
            lcb_values.append(self.lcb(x,i))
        return max(lcb_values)
    
    def Expander(self):
        eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
        obj_fun = lambda x: -self.GP_inference_jit(x[:self.nx_dim],self.inference_datasets)[1][0] # objective function is -1*variance as differential equation finds min (convert to max)
        bound = jnp.vstack((self.bound,self.bound))
        safe_unsafe_cons = []
        for i in range(1, self.n_fun):
            safe_unsafe_cons.append(NonlinearConstraint(lambda x: self.lcb(x[:self.nx_dim],i),0.,jnp.inf)) 
        safe_unsafe_cons.append(NonlinearConstraint(lambda x: self.lcb_constraint_min(x[self.nx_dim:]),-jnp.inf,0.))
        
        # Find expander for each constraint
        for index in range(1,self.n_fun):
            def callback(x,convergence):
                lipschitz_constraint = self.Lipschitz_continuity_constraint(x,index,max_infnorm_mean_constraints)
                if lipschitz_constraint >= 0.:
                    self.feasible_found = True
                    self.iterations_without_feasible = 0
                else:
                    self.iterations_without_feasible += 1
                # print(self.iterations_without_feasible)
                # Stop optimization if no feasible solution has been found after a certain number of iterations
                if not self.feasible_found and self.iterations_without_feasible >= self.max_iter_without_feasible:
                    print("Terminating early: No feasible solution found.")
                    return True  # Returning True stops the optimization

            Expander_cons = copy.deepcopy(safe_unsafe_cons)
            Expander_cons.append(NonlinearConstraint(lambda x: self.lcb(x[:self.nx_dim],index),0,eps))
            max_infnorm_mean_constraints = self.maximize_infnorm_mean_grad(index)
            Expander_cons.append(NonlinearConstraint(lambda x: self.Lipschitz_continuity_constraint(x,index,max_infnorm_mean_constraints),0.,jnp.inf))
            
            # Feasibility Checker
            self.feasible_found = False
            self.iterations_without_feasible = 0
            self.max_iter_without_feasible = 100
            satisfied = False
            count = 0
    
            while not satisfied and count < 10:       
                result = differential_evolution(obj_fun,bound,constraints=Expander_cons,callback=callback)
                lcb_constraint = self.lcb(result.x[:self.nx_dim],index)
                
                if lcb_constraint < 0.:
                    self.feasible_found = False
                    self.iterations_without_feasible = 0
                    self.max_iter_without_feasible = 100
                else:
                    satisfied = True
                
                count += 1

        return result.x[:self.nx_dim], jnp.sqrt(-result.fun),satisfied, count

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