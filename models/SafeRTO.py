import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize
import sobol_seq
from models.GP_SASBO import GP

class SafeOpt(GP):
    def __init__(self,plant_system):
        GP.__init__(self,plant_system)
        self.sample_set = {'all':[],'safe':[],'maximizer':[],'expander':[]}
        self.GP_inference_np_jit = jit(self.GP_inference_np)

    def ucb(self, x, b):
        GP_inference                = self.GP_inference_np_jit(x)
        mean                        = GP_inference[0][0]
        std                         = jnp.sqrt(GP_inference[1][0])

        return mean + b*std
    
    def lcb(self, x, b):
        GP_inference                = self.GP_inference_np_jit(x)
        mean                        = GP_inference[0][0]
        std                         = jnp.sqrt(GP_inference[1][0])

        return mean - b*std
    
    def obj_fun(self,x,b):
        GP_inference                = self.GP_inference_np_jit(x)
        mean                        = GP_inference[0][0]
        std                         = jnp.sqrt(GP_inference[1][0])

        return -2*b*std # = min(-(ucb - lcb)) = max(ucb -lcb)

    def lcb_constraint(self, x, b, index):
        GP_inference                = self.GP_inference_np_jit(x)
        mean                        = GP_inference[0][index]
        std                         = jnp.sqrt(GP_inference[1][index])

        return mean - b*std
    
    def sobol_seq_sampling(self,x_dim,n_sample,bound,skip=0):
        '''
        Description:
        Arguments:
        Results:
        '''
        fraction = sobol_seq.i4_sobol_generate(x_dim,n_sample,skip)
        lb = bound[0]
        ub = bound[1]
        sample = lb + (ub-lb) * fraction
        
        return sample
    
    def Safe_filter(self,sample,b):
        constraint_vmap = vmap(self.lcb_constraint, in_axes=(0,None,None))
        sample_constraint = constraint_vmap(sample,b,1) ##!!!! Need to change 1 to index + need to make for loop for all constraints.
        mask_safe = sample_constraint >= 0.0 
        sample_safe = sample[mask_safe]
        
        return sample_safe
    
    def minimize_ucb(self,set_safe,b,multi_start=5):
        # Initialization
        options = {'disp':False, 'maxiter':10000,'ftol': 1e-12} 
        cons = []
        localsol = []
        localval = []

        # Jit relavent JAX grad
        ucb_jitgrad                 = jit(grad(self.ucb,argnums=0))
        constraint_jitgrad          = jit(grad(self.lcb_constraint,argnums=0))

        for i in range(self.n_fun):
            if i == 0:
                obj_fun = lambda x: self.ucb(x,b)
                obj_grad = lambda x: ucb_jitgrad(x,b)
            else:
                cons.append({'type': 'ineq',
                             'fun': lambda x: self.lcb_constraint(x,b,i),
                             'jac': lambda x: constraint_jitgrad(x,b,i)
                             })
        
        for j in set_safe:
            res = minimize(obj_fun,j,constraints=cons,method='SLSQP',
                           jac=obj_grad,options=options,tol=1e-8)
        
        for con in cons:
            if con['fun'](res.x) < -0.01: # Barrier when minimize significantly fails to satisfy any constraints 
                passed = False 
                break
            else:
                passed = True
        if passed:
            localsol.append(res.x)
            localval.append(res.fun)
        
        localsol                    = jnp.array(localsol)
        localval                    = jnp.array(localval)
        minindex                    = jnp.argmin(localval)
        xopt                        = localsol[minindex]
        minucb                      = localval[minindex]
        
        return xopt, minucb

    def minimizer_constraint(self,x,b,min_ucb):
        GP_inference                = self.GP_inference_np_jit(x)
        mean                        = GP_inference[0][0]
        std                         = jnp.sqrt(GP_inference[1][0])
        lcb                         = mean - b*std

        return min_ucb - lcb

    def minimizer_filter(self,sample_safe,b,minucb):
        minimizer_constraint_vmap = vmap(self.minimizer_constraint, in_axes=(0,None,None))
        sample_minimizer_constraint = minimizer_constraint_vmap(sample_safe,b,minucb)
        mask = sample_minimizer_constraint >= 0.0
        sample_minimizer = sample_safe[mask]

        return sample_minimizer
    
    def expander_constraint(self,x,b,eps):
        GP_inference                = self.GP_inference_np_jit(x)
        mean                        = GP_inference[0][0]
        std                         = jnp.sqrt(GP_inference[1][0])
        lcb                         = mean - b*std

        return eps - lcb

    def expander_filter(self,sample_safe,b,eps:float):
        constraint_vmap = vmap(self.lcb_constraint, in_axes=(0,None,None))
        sample_constraint = constraint_vmap(sample_safe,b,1)
        mask = sample_constraint <= eps
        sample_expander = sample_safe[mask]
        
        return sample_expander
    
    def Set_sampling(self,x_dim,b,multi_start,bound,eps):
        set_minimizer = jnp.empty((0,x_dim))
        set_expander = jnp.empty((0,x_dim))
        n_sample = 100
        skip = 0

        while set_expander.shape[0] < multi_start or set_minimizer.shape[0] < multi_start:        
            sample = self.sobol_seq_sampling(x_dim,n_sample,bound,skip)
            set_safe = self.Safe_filter(sample,b)
            
            if set_safe.shape[0] > multi_start:

                if skip == 0:
                    xopt, minucb = self.minimize_ucb(set_safe[:multi_start],b,multi_start)
                
                # Minimizer
                sample_minimizer = self.minimizer_filter(set_safe,b,minucb)
                set_minimizer= jnp.concatenate((set_minimizer,sample_minimizer),axis=0)

                # Expander
                sample_expander = self.expander_filter(set_safe,b,eps)
                set_expander = jnp.concatenate((set_expander,sample_expander),axis=0)

            skip += n_sample

        return set_minimizer[:multi_start],set_expander[:multi_start]
    
    def minimize_minimizer(self,set_minimizer,b,minucb):
        # Initialization
        options = {'disp':False, 'maxiter':10000,'ftol': 1e-12} 
        cons = []
        localsol = []
        localval = []

        # Jit relavent JAX grad
        obj_fun_jitgrad                 = jit(grad(self.ucb,argnums=0))
        constraint_jitgrad              = jit(grad(self.lcb_constraint,argnums=0))
        minimizer_constraint_jitgrad    = jit(grad(self.minimizer_constraint,argnums=0))

        for i in range(self.n_fun):
            if i == 0:
                obj_fun = lambda x: self.ucb(x,b)
                obj_grad = lambda x: obj_fun_jitgrad(x,b)
            else:
                cons.append({'type': 'ineq',
                             'fun': lambda x: self.lcb_constraint(x,b,i),
                             'jac': lambda x: constraint_jitgrad(x,b,i)
                             })
                
        cons.append({'type': 'ineq',
                     'fun': lambda x: self.minimizer_constraint(x,b,i),
                     'jac': lambda x: minimizer_constraint_jitgrad(x,b,minucb)})
        
        for j in set_minimizer:
            res = minimize(obj_fun,j,constraints=cons,method='SLSQP',
                           jac=obj_grad,options=options,tol=1e-8)
        
        for con in cons:
            if con['fun'](res.x) < -0.01: # Barrier when minimize significantly fails to satisfy any constraints 
                passed = False 
                break
            else:
                passed = True
        if passed:
            localsol.append(res.x)
            localval.append(res.fun)
        
        localsol                    = jnp.array(localsol)
        localval                    = jnp.array(localval)
        minindex                    = jnp.argmin(localval)
        xopt                        = localsol[minindex]
        funopt                      = localval[minindex]
        
        return xopt, funopt
    
    def minimize_expander(self,set_expander,b,eps):
        # Initialization
        options = {'disp':False, 'maxiter':10000,'ftol': 1e-12} 
        cons = []
        localsol = []
        localval = []

        # Jit relavent JAX grad
        obj_fun_jitgrad                 = jit(grad(self.ucb,argnums=0))
        constraint_jitgrad              = jit(grad(self.lcb_constraint,argnums=0))
        minimizer_constraint_jitgrad    = jit(grad(self.minimizer_constraint,argnums=0))

        for i in range(self.n_fun):
            if i == 0:
                obj_fun = lambda x: self.ucb(x,b)
                obj_grad = lambda x: obj_fun_jitgrad(x,b)
            else:
                cons.append({'type': 'ineq',
                             'fun': lambda x: self.lcb_constraint(x,b,i),
                             'jac': lambda x: constraint_jitgrad(x,b,i)
                             })
                
        cons.append({'type': 'ineq',
                     'fun': lambda x: self.expander_constraint(x,b,i),
                     'jac': lambda x: minimizer_constraint_jitgrad(x,b,eps)})
        
        for j in set_expander:
            res = minimize(obj_fun,j,constraints=cons,method='SLSQP',
                           jac=obj_grad,options=options,tol=1e-8)
        
        for con in cons:
            if con['fun'](res.x) < -0.01: # Barrier when minimize significantly fails to satisfy any constraints 
                passed = False 
                break
            else:
                passed = True
        if passed:
            localsol.append(res.x)
            localval.append(res.fun)
        
        localsol                    = jnp.array(localsol)
        localval                    = jnp.array(localval)
        minindex                    = jnp.argmin(localval)
        xopt                        = localsol[minindex]
        funopt                      = localval[minindex]
        
        return xopt, funopt