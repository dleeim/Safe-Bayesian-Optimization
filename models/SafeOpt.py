import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from models.GP_Safe import GP

jax.config.update("jax_enable_x64", True)

class BO(GP):
    def __init__(self,plant_system,bound,b):
        GP.__init__(self,plant_system)
        self.bound = bound
        self.b = b
        self.GP_inference_jit = jit(self.GP_inference)  

    def calculate_plant_outputs(self,x,noise=0):
        plant_output            = []
        for plant in self.plant_system:
            plant_output.append(plant(x,noise)) 

        return jnp.array(plant_output)
    
    def lcb_constraint_min(self,x):
        lcb_values = []
        for i in range(1,self.n_fun):
            lcb_values.append(self.lcb(x,i))
        return max(lcb_values)
    
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
        # print(x,i,value)
        return value
    
    def minimize_obj_ucb(self,safe_set_cons):
        obj_fun = lambda x: self.ucb(x,0)
        result = differential_evolution(obj_fun,self.bound,constraints=safe_set_cons,polish=False)

        return result.x, result.fun
    
    def Minimizer(self):
        # Setting Optimization Problem
        obj_fun = lambda x: -self.GP_inference_jit(x,self.inference_datasets)[1][0] # objective function is -variance as differential equation finds min (convert to max)
        
        Minimizer_cons = []
        for i in range(1, self.n_fun):
            Minimizer_cons.append(NonlinearConstraint(lambda x, i=i: self.lcb(x,i),0.,jnp.inf)) 
             
        _, min_obj_ucb = self.minimize_obj_ucb(safe_set_cons=Minimizer_cons)
        Minimizer_cons.append(NonlinearConstraint(lambda x: min_obj_ucb - self.lcb(x,0),0,jnp.inf))

        # Differential Evolution
        result = differential_evolution(obj_fun,self.bound,constraints=Minimizer_cons,polish=False)
        return result.x, jnp.sqrt(-result.fun)
    
    def infnorm_mean_grad(self,x,i):
        mean_grad_jit = grad(self.mean,argnums=0)
        grad_mean = mean_grad_jit(x,i)
        return jnp.max(jnp.abs(grad_mean))
    
    def lcb_constraint_min(self,x):
        lcb_values = []
        for i in range(1,self.n_fun):
            lcb_values.append(self.lcb(x,i))
        return max(lcb_values)

    def maximize_infnorm_mean_grad(self,i):
        infnorm_mean_grad_jit = jit(self.infnorm_mean_grad)
        obj_fun = lambda x, i=i: -infnorm_mean_grad_jit(x,i)
        result = differential_evolution(obj_fun,self.bound,polish=False)
        return -result.fun

    def Lipschitz_continuity_constraint(self,x,i,max_infnorm_mean_grad):
        ucb_value = self.ucb(x[:self.nx_dim],i)    
        value = ucb_value - max_infnorm_mean_grad*jnp.linalg.norm(x[:self.nx_dim]-x[self.nx_dim:]+1e-8)  
        return value

    def Expander(self):
        eps = jnp.sqrt(jnp.finfo(jnp.float16).eps)
        obj_fun = lambda x: -self.GP_inference_jit(x[:self.nx_dim],self.inference_datasets)[1][0] # objective function is -1*variance as differential equation finds min (convert to max)
        bound = jnp.vstack((self.bound,self.bound)) 
        
        # Find expander for each constraint
        expanders = []
        std_expanders = []

        for index in range(1,self.n_fun):
            Lipschitz_continuity_constraint_jit = jit(self.Lipschitz_continuity_constraint)
            Expander_cons = []

            for i in range(1,self.n_fun):
                if i == index:
                    Expander_cons.append(NonlinearConstraint(lambda x, i=i: self.lcb(x[:self.nx_dim],i),0,jnp.inf))
                else:
                    Expander_cons.append(NonlinearConstraint(lambda x, i=i: self.lcb(x[:self.nx_dim],i),0,jnp.inf))

            Expander_cons.append(NonlinearConstraint(lambda x: self.lcb_constraint_min(x[self.nx_dim:]),-jnp.inf,0.))
            max_inform_mean_grad = self.maximize_infnorm_mean_grad(i)
            Expander_cons.append(NonlinearConstraint(lambda x, index=index: Lipschitz_continuity_constraint_jit(x,index,max_inform_mean_grad),0.,jnp.inf))    
            result = differential_evolution(obj_fun,bound,constraints=Expander_cons,polish=False)
            print(result.x,-result.fun)

            # Collect optimal point and standard deviation
            expanders.append(result.x[:self.nx_dim])
            std_expanders.append(jnp.sqrt(-result.fun))

        # Find most uncertain expander
        max_std = max(std_expanders)
        max_index = std_expanders.index(max_std)
        argmax_x = expanders[max_index]

        return argmax_x, max_std
