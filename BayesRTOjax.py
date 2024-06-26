import jax.numpy as jnp
import jax
import random
from jax import grad, value_and_grad, jit, vmap
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import BayesOptjax
import Benoit_Problem

class Bayesian_RTO():
    
    def __init__(self) -> None:
        '''
        Global Variables:```
            n_sample                        : number of sample given
            u_dim                           : dimension of input
            n_fun                           : number of obj functions and constraints we want to measure
            plant_system                    : array with obj functions and constraints for plant system
            model                           : array with obj functions and constraints for model
            input_sample                    : sampled input data made from Ball_sampling and used for GP initialization
        '''
        self.n_sample                       = 0
        self.u_dim                          = 0
        self.n_fun                          = 0
        self.plant_system                   = 0
        self.model                          = 0
        self.input_sample                   = 0
    
    ###########################################
    #######______GP Initialization______#######
    ###########################################

    def Ball_sampling(self,ndim,n_sample,r_i,key) -> float:
        '''
        Description:
            This function samples randomly at (0,0) within a ball of radius r_i.
            By adding sampled point onto initial point (u1,u2) you will get 
            randomly sampled points around (u1,u2)
        Arguments:
            ndim                            : no of dimensions required for sampled point
            r_i                             : radius from (0,0) of circle area for sampling
            key                             : key for random seed
        Returns: 
            d_init                          : sampled distances from (0,0)
        '''
        u = jax.random.normal(key, (n_sample,ndim))
        norm = jnp.sqrt(jnp.sum(u**2))
        r = jax.random.uniform(key, (n_sample,1))**(1.0 / ndim)
        d_init = r*u / norm*r_i*2

        return d_init
    
    def WLS(self,theta,input_sample) -> float:
        '''
        Description:
            This function finds the sum of weighted least square between
            plant system and model. This is used as an min objective function
            for parameter estimation of a model using sampled data.
        Argument: 
            theta                           : parameter used in model obj func and cons.
            u_sample                        : sample data 
        Returns:
            error                           : weighted least square between plant and model with input as sample data
        '''
        error = jnp.zeros((self.n_sample))
        for i in range(self.n_fun):
            plant_system_value = vmap(self.plant_system[i])(input_sample)
            model_value = vmap(self.model[i],in_axes=(None,0))(theta,input_sample)
            error += (plant_system_value - model_value)**2/jnp.abs(plant_system_value)
            error = jnp.sum(error)

        return error  
    
    def parameter_estimation(self,theta,input_sample):
        '''
        Description:
            Uses scipy minimize to find the optimal parameter theta for model that has 
            smallest difference to plant system using input as sample data. This is used 
            at method GP_initialization to find optimal theta for model using sampled data.
        Argument:
            theta                           : parameter used in model obj func and cons
            u_sample                        : sample data 
        Returns:
            theta_opt                       : parameter theta that makes minimal difference between plant system 
                                                and model using input as sample data
        '''
        WLS_value_and_grad = value_and_grad(self.WLS,argnums=0)
        sol = minimize(WLS_value_and_grad, x0=theta, args=(input_sample), 
                       method='SLSQP', jac=True, tol=jnp.finfo(jnp.float32).eps)

        return sol.x
    
    def modifier_calc(self,theta,input_sample):
        '''
        Description:
            Finds difference between plant systema and model which will be used in 
            various methods such as GP_initialization.
        Argument:
            u_sample                        : sample data 
            theta                           : parameter used in model obj func and cons.
        Returns:
            modifier                        : A matrix as follows
                                            | diff obj func (sample 1) diff cons(sample 1) ... |
                                            | diff obj func (sample 2) diff cons(sample 2) ... |
                                            | diff obj func (sample 3) diff cons(sample 3) ... |
                                            | ...                      ...                 ... |
        '''
        input_sample = jnp.atleast_2d(input_sample)
        n_sample = input_sample.shape[0]
        modifier = jnp.zeros((n_sample,self.n_fun))

        for i in range(self.n_fun):
            plant_system_value = vmap(self.plant_system[i])(input_sample)
            model_value = vmap(self.model[i],in_axes=(None,0))(theta,input_sample)
            modifier = modifier.at[:,i].set(plant_system_value-model_value)

        return modifier
    
    def GP_Initialization(self,n_sample,u_0,theta_0,r,plant_system,model):
        '''
        Description:
            Initialize GP model by:
                1) Create a sample data in matrix
                2) Find a parameter theta of a model that has minimal difference 
                    between plant system and model
                3) Then find a modifier 
                    (= matrix of difference between plant and model with 
                    the optimal theta in all sampled data)
                4) Finally, use modifier to initialize GP using BayesOpt
        Argument:
            n_sample                        : number of sample points that can be collected
            u_0                             : initial input
            theta_0                         : initial estimated parameter
            r                               : radius of area where samples are collected in Ball_sampling
            plant_system                    : numpy array with objective functions and constraints for plant system
            model                           : numpy array with objective functions and constraints for model
        Returns:
            theta                           : parameter theta for model that makes minimal difference between plant and model
            GP_m                            : GP model that is initialized using random sampling around initial input
        '''
       # === Define relavent parameters and arrays === #
        self.n_sample                       = n_sample
        self.u_dim                          = jnp.shape(u_0)[0]
        self.n_fun                          = len(plant_system)
        self.plant_system                   = plant_system
        self.model                          = model
        key = jax.random.PRNGKey(42)

        # === Collect Training Dataset (Input) === #
        u_trial = self.Ball_sampling(self.u_dim,n_sample,r,key)
        u_trial += u_0
        
        # To store the sampled input data and to see when using the class
        self.input_sample = u_trial

        # === Estimate the parameter theta === #
        theta = self.parameter_estimation(theta_0,self.input_sample)

        # === Collect Training Dataset === #
        modifier = self.modifier_calc(theta,self.input_sample)

        # === Initialize GP with modifier === #
        GP_m = BayesOptjax.BayesianOpt(self.input_sample, modifier, 'RBF', 
                                       multi_hyper=2, var_out=True)
        
        return theta, GP_m
    
    def objective_function(self,d, theta, u_0, GP_m, b, model_func):
        value = model_func(theta, u_0 + d) + GP_m.GP_inference_np(u_0 + d)[0][0] - b * jnp.sqrt(GP_m.GP_inference_np(u_0 + d)[1][0])

        return value
    
    def obj_value_and_grad(self,d, theta, u_0, GP_m, b, model_func):
        fun_grad = value_and_grad(self.objective_function,argnums=0)
        value = fun_grad(d, theta, u_0, GP_m, b, model_func)
                
        jax.debug.print("d_new, {}", d)
        jax.debug.print("u_new, {}", u_0 + d)
        jax.debug.print("obj, {}", value)

        return value
    

    def model_constraint(self,d, theta, u_0, GP_m, b, model_func, index):
        value = model_func(theta, u_0 + d) + GP_m.GP_inference_np(u_0 + d)[0][index] - b * jnp.sqrt(GP_m.GP_inference_np(u_0 + d)[1][index])
        jax.debug.print("model constraint, {}", value)
        return value

    def trust_region_constraint(self,d, r):
        jax.debug.print("trust region, {}", r - jnp.linalg.norm(d))
        return r - jnp.linalg.norm(d)
    
    ####################################################
    ############______New Observation______#############
    ####################################################
    def optimize_acquisition(self,r,u_0,theta,GP_m,b=0):
        '''
        Description:
        Argument:
            r: radius of trust region area
            u_0: previous input observed
            GP_m: Gaussian Process Model
        Results:
            result.x: a distance from input u_0 to observe the corresponding output of function
        '''
        d0 = jnp.array([0.,0.])
        cons = []

        # Collect All objective function and constraints(model constraint + trust region)
        for i in range(self.n_fun):
            if i == 0:
                model_func = self.model[0]
                obj_fun = lambda d: self.objective_function(d, theta, u_0, GP_m, b, model_func)
                obj_grad = grad(obj_fun)
            else:
                model_func = self.model[i]
                cons.append({'type': 'ineq',
                             'fun': lambda d: self.objective_function(d, theta, u_0, GP_m, b, model_func)})
                
        cons.append({'type': 'ineq',
                    'fun': lambda d: self.trust_region_constraint(d,r)})
        
        bounds = [(-r, r), (-r, r)]

        result = minimize(obj_fun, d0, constraints=cons, method='SLSQP', 
                          jac=obj_grad,options={'maxiter':3}, bounds=bounds)
        
        return result.x,result.fun



if __name__ == '__main__':
    
    ##################################################
    ##### Test Case 1: Test on GP_Initialization #####
    ##################################################

    ## Initial Parameters
    BRTO = Bayesian_RTO()
    theta_0 = jnp.array([1.,1.,1.,1.])
    u_0 = jnp.array([4.,-1.])
    u_dim = u_0.shape[0]
    r_i = 1.
    n_s = 4
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system]

    model = [Benoit_Problem.Benoit_Model_1,
            Benoit_Problem.con1_Model]
    
    ## GP Initialization
    print("#######______Test Case: GP_Initialization______#######")
    theta, GP_m = BRTO.GP_Initialization(n_s,u_0,theta_0,r_i,plant_system,model)
    print("sampled input:")
    print(BRTO.input_sample)

    print("\n#######______Test Case: WLS______#######")
    print(BRTO.WLS(theta_0,BRTO.input_sample))
    WLS_grad = grad(BRTO.WLS,argnums=0)
    print(WLS_grad(theta_0,BRTO.input_sample))

    print("\n#######______Test Case: parameter_estimation______#######")
    theta = BRTO.parameter_estimation(theta_0,BRTO.input_sample)
    print(theta)

    print("\n#######______Test Case: modifier_calc______#######")
    print(BRTO.modifier_calc(theta,BRTO.input_sample))

    print("\n#######______Test if GP model is working fine______#######")
    u_0 = jnp.array([2.15956118, -1.42712019])
    d_0 = jnp.array([0,0])
    print(f"input = {u_0}")

    ## Test if GP model is working fine
    GP_modifier = GP_m.GP_inference_np(u_0)

    ## Check if plant and model provides same output using sampled data as input
    print("\n #___Check if plant and model provides similar output using sampled data as input___#")
    print(f"plant obj: {plant_system[0](u_0)}")
    print(f"model obj: {model[0](theta,u_0)+GP_modifier[0][0]}")
    print(f"plant con: {plant_system[1](u_0)}")
    print(f"model con: {model[1](theta,u_0)+GP_modifier[0][1]}")

    ## Check if variance is approx 0 at sampled input
    print("\n #___Check variance at sampled input___#")
    print(f"variance: {GP_modifier[1]}")

    ######################################################
    #### Test Case 2: Test on observation_trustregion ####
    ######################################################
    
    print("\n #######______Test Case: optimization aquisition______#######")
    ## Find info on old observation
    print("#___Find info on old observation___#")
    u_0 = jnp.array([4.,-1.])
    d_new, obj = BRTO.optimize_acquisition(r_i,u_0,theta,GP_m)
    print(d_new)
    print(obj)
    # print(f"optimal old input(model): {u_0}")
    # print(f"optimal old output(model): {model[0](theta,u_0)+GP_modifier[0][0]}")
    # print(f"old constraint(model): {model[1](theta,u_0)+GP_modifier[0][1]}")
    # print(f"GP model: {GP_m.GP_inference_np(u_0)}")

    # ## Find info on new observation
    # print(d_new)
    # GP_modifier = GP_m.GP_inference_np(u_0+d_new)
    # print("\n #___Find info on new observation and see if improved___#")
    # print(f"optimal new input(model): {u_0+d_new}")
    # print(f"Euclidean norm of d_new(model): {jnp.linalg.norm(d_new)}")
    # print(f"optimal new output(model): {model[0](theta,u_0+d_new)+GP_modifier[0][0]}")
    # print(f"new constraint(model): {model[1](theta,u_0+d_new)+GP_modifier[0][1]}")
    # print(f"GP model: {GP_m.GP_inference_np(u_0+d_new)}")


















    


