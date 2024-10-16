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

    # def Expander_constraint(self,x,unsafe_sobol_sample,maximum_maxnorm_mean_constraints,eps):
    #     min_val = jnp.inf
    #     lcb_non_expander = []
    #     for i in range(1,self.n_fun):
    #         lcb_value = self.lcb(x,i)

    #         if lcb_value <= eps and lcb_value >= 0.:
    #             for j in range(len(unsafe_sobol_sample)):
    #                 sobol_point = unsafe_sobol_sample[j]
    #                 value = self.ucb(x,i) - maximum_maxnorm_mean_constraints*cdist(x.reshape(1, -1),sobol_point.reshape(1, -1))

    #                 if value >= 0.:
    #                     if j > int(0.05*len(unsafe_sobol_sample)):
    #                         unsafe_sobol_sample = jnp.vstack((sobol_point,unsafe_sobol_sample[:j],unsafe_sobol_sample[j+1:]))
    #                     print(f"value,self.ucb(x,i),maximum_maxnorm_mean_constraints,sobol_point.reshape(1, -1): {value,self.ucb(x,i),maximum_maxnorm_mean_constraints,sobol_point.reshape(1, -1)}")
    #                     return lcb_value
    #                 else:
    #                     if value < min_val:
    #                         min_val = value.item()
    #         else:
    #             lcb_non_expander.append(-1*lcb_value)
            
    #     if len(lcb_non_expander)==0:
    #         return min_val
    #     else:
    #         lcb_value = max(lcb_non_expander,key=abs)
    #         if abs(lcb_value)>abs(min_val) or min_val==jnp.inf:
    #             return lcb_value
    #         else:
    #             return min_val
    
    # def Expander(self,unsafe_sobol_sample,maximum_maxnorm_mean_constraints):
    #     eps = jnp.sqrt(jnp.finfo(jnp.float64).eps)
    #     obj_fun = lambda x: -self.GP_inference_jit(x,self.inference_datasets)[1][0] # objective function is -1*variance as differential equation finds min (convert to max)
    #     cons = copy.deepcopy(self.safe_set_cons)
    #     cons.append(NonlinearConstraint(lambda x: self.Expander_constraint(x,unsafe_sobol_sample,maximum_maxnorm_mean_constraints,eps),0.,eps))
    #     satisfied = False
    #     while not satisfied:
    #         result = differential_evolution(obj_fun,self.bound,constraints=cons)
    #         for i in range(1, self.n_fun):
    #             lcb_value = self.lcb(result.x,i)
                
    #             if lcb_value < 0.:
    #                 break
            
    #         satisfied = True
    #     return result.x, jnp.sqrt(-result.fun)

    def Boundary_constraint(self,x,eps):
        lcb_value_max_absvalue = 0.

        for i in range(1,self.n_fun):
            lcb_value = self.lcb(x,i)
            
            if lcb_value >= 0. and lcb_value <= eps:
                # print(f'boundary yes: {x,lcb_value.item()}')
                return lcb_value.item()
            
            if abs(lcb_value) > lcb_value_max_absvalue:
                lcb_value_max_absvalue = abs(lcb_value)
                lcb_value_output = lcb_value
        # print(f"boundary no: {lcb_value.item()}")
        return lcb_value_output.item()
    
    # def Lipschitz_continuity_constraint(self,x,maximum_maxnorm_mean_constraints,eps):
    #     # Find constraint that satisfied boundary constraint; if all not satisfied, then use constraint with highest lcb_value
    #     lcb_value_max_absvalue = 0.
    #     for i in range(1,self.n_fun):
    #         lcb_value = self.lcb(x,i)

    #         if lcb_value >= 0. and lcb_value <= eps:
    #             index = i
                
    #             # Find if lipschitz continuity is satisfied (>0); if not, return biggest value of lipschitz continuity equation
    #             max_value = -jnp.inf

    #             for j in range(len(self.unsafe_sobol_sample)):
    #                 sobol_point = self.unsafe_sobol_sample[j]
    #                 value = self.ucb(x,index) - maximum_maxnorm_mean_constraints*cdist(x.reshape(1, -1),sobol_point.reshape(1, -1))

    #                 if value >= 0.:
    #                     if j > int(0.05*len(self.unsafe_sobol_sample)):
    #                         self.unsafe_sobol_sample = jnp.vstack((sobol_point,self.unsafe_sobol_sample[:j],self.unsafe_sobol_sample[j+1:]))
    #                     # print(f"boundary, lipschitz yes; (value, input, unsafepoint): {value.item(),x,sobol_point}")
    #                     return value.item()
                    
    #                 if value > max_value:
    #                     max_unsafe = sobol_point
    #                     max_value = value
    #             # print(f"boundary yes, lipschitz no(input, max unsafe, max value, ucb, maximum_maxnorm_mean): {x,max_value.item(),max_unsafe,self.ucb(x,index),maximum_maxnorm_mean_constraints}")
    #             return max_value.item()

    #         elif abs(lcb_value) > lcb_value_max_absvalue:
    #             index = i

    #     sobol_point = self.unsafe_sobol_sample[0]
    #     value = self.ucb(x,index) - maximum_maxnorm_mean_constraints*cdist(x.reshape(1, -1),sobol_point.reshape(1, -1))
    #     # print(f"boundary no, lipschitz first;(value, input, unsafepoint): {value.item(),x,sobol_point}")
    #     return value.item()

    def Lipschitz_continuity_constraint(self,x,maximum_maxnorm_mean_constraints,eps):
        # Find constraint that satisfied boundary constraint; if all not satisfied, then use constraint with highest lcb_value
        # lcb_value_max_absvalue = 0.
        for i in range(1,self.n_fun):
            lcb_value = self.lcb(x,i)

            if lcb_value >= 0. and lcb_value <= eps:
                index = i
                ucb_value = self.ucb(x,index)
                # Find if lipschitz continuity is satisfied (>0); if not, return biggest value of lipschitz continuity equation
                min_value = jnp.inf

                for j in range(len(self.unsafe_sobol_sample)):
                    sobol_point = self.unsafe_sobol_sample[j]
                    value = ucb_value - maximum_maxnorm_mean_constraints*cdist(x.reshape(1, -1),sobol_point.reshape(1, -1))

                    if value >= 0.:
                        if j > int(0.05*len(self.unsafe_sobol_sample)):
                            self.unsafe_sobol_sample = jnp.vstack((sobol_point,self.unsafe_sobol_sample[:j],self.unsafe_sobol_sample[j+1:]))
                        # print(f"boundary, lipschitz yes; (value, input, unsafepoint): {value.item(),x,sobol_point}")
                        return value.item()
                    
                    if value < min_value:
                        min_unsafe = sobol_point
                        min_value = value
                # print(f"boundary yes, lipschitz no(input, min unsafe, min value, ucb, maximum_maxnorm_mean): {x,min_value.item(),min_unsafe,self.ucb(x,index),maximum_maxnorm_mean_constraints}")
                return min_value.item()

            # elif abs(lcb_value) > lcb_value_max_absvalue:
                
            #     index = i

        # sobol_point = self.unsafe_sobol_sample[0]
        # value = self.ucb(x,index) - maximum_maxnorm_mean_constraints*cdist(x.reshape(1, -1),sobol_point.reshape(1, -1))
        # # print(f"boundary no, lipschitz first;(value, input, unsafepoint): {value.item(),x,sobol_point}")
        # return value.item()
        return -1.
    
    def Expander(self,unsafe_sobol_sample,maximum_maxnorm_mean_constraints):
        self.unsafe_sobol_sample = unsafe_sobol_sample
        eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
        obj_fun = lambda x: -self.GP_inference_jit(x,self.inference_datasets)[1][0] # objective function is -1*variance as differential equation finds min (convert to max)
        cons = copy.deepcopy(self.safe_set_cons)
        cons.append(NonlinearConstraint(lambda x: self.Boundary_constraint(x,eps),0,eps))
        cons.append(NonlinearConstraint(lambda x: self.Lipschitz_continuity_constraint(x,maximum_maxnorm_mean_constraints,eps),0.,jnp.inf))

        # Feasibility Checker
        self.feasible_found = False
        self.iterations_without_feasible = 0
        self.max_iter_without_feasible = 100
        def callback(x,convergence):
            lipschitz_constraint = self.Lipschitz_continuity_constraint(x,maximum_maxnorm_mean_constraints,eps)
            if lipschitz_constraint >= 0:
                self.feasible_found = True
                self.iterations_without_feasible = 0
            else:
                self.iterations_without_feasible += 1
            # print(self.iterations_without_feasible)
            # Stop optimization if no feasible solution has been found after a certain number of iterations
            if not self.feasible_found and self.iterations_without_feasible >= self.max_iter_without_feasible:
                print("Terminating early: No feasible solution found.")
                return True  # Returning True stops the optimization
        
        satisfied = False
        count = 0
        while not satisfied and count < 10:
            lcb_values = []
            result = differential_evolution(obj_fun,self.bound,constraints=cons,callback=callback)
            for i in range(1, self.n_fun):
                lcb_values.append(self.lcb(result.x,i))
            
            lcb_values = jnp.array(lcb_values)
            if jnp.any(lcb_values < 0.):
                self.feasible_found = False
                self.iterations_without_feasible = 0
                self.max_iter_without_feasible = 100
            else:
                satisfied = True
                # print(f"satisfied: {satisfied}")
            
            count += 1
        return result.x, jnp.sqrt(-result.fun), satisfied, count

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