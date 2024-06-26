import jax.numpy as jnp
from jax import random
from jax import grad, jit
from jax.scipy.optimize import minimize
from jax.scipy.optimize import fsolve
import matplotlib.pyplot as plt
import BayesOpt
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

    def Ball_sampling(self,ndim,r_i, key) -> float:
        '''
        Description:
            This function samples randomly at (0,0) within a ball of radius r_i.
            By adding sampled point onto initial point (u1,u2) you will get 
            randomly sampled points around (u1,u2)
        Arguments:
            ndim                            : no of dimensions required for sampled point
            r_i                             : radius from (0,0) of circle area for sampling
        Returns: 
            d_init                          : sampled distances from (0,0)
        '''
        key, subkey = random.split(key)
        u = random.normal(subkey, (ndim,))
        norm = jnp.sqrt(jnp.sum(u**2))
        r = random.uniform(subkey) ** (1.0 / ndim)
        d_init = r * u / norm * r_i * 2 

        return d_init, key

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
        error = 0
        for i in range(self.n_sample): 
            u = input_sample[i,:]
            for j in range(self.n_fun): 
                error += (self.plant_system[j](u) - self.model[j](theta,u))**2 / jnp.abs(self.plant_system[j](u))
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
        sol = minimize(self.WLS, x0=theta, args=(input_sample), method='SLSQP')
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

        for i in range(n_sample):
            u = input_sample[i]
            for j in range(self.n_fun):
                modifier = modifier.at[i, j].set(self.plant_system[j](u) - self.model[j](theta,u))

        return modifier

    def GP_Initialization(self,n_sample,u_0,theta_0,r,plant_system,model, key):
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
        input_sample                        = jnp.zeros((self.n_sample,self.u_dim))

        # === Collect Training Dataset (Input) === #
        for sample_i in range(n_sample):
            u_trial, key = self.Ball_sampling(self.u_dim, r, key)
            input_sample = input_sample.at[sample_i].set(u_0 + u_trial)
        
        # To store the sampled input data and to see when using the class
        self.input_sample = input_sample

        # === Estimate the parameter theta === #
        theta = self.parameter_estimation(theta_0,input_sample)

        # === Collect Training Dataset === #
        modifier = self.modifier_calc(theta,input_sample)

        # === Initialize GP with modifier === #
        GP_m = BayesOpt.BayesianOpt(input_sample, modifier, 'RBF', 
                                    multi_hyper=2, var_out=True)

        return theta, GP_m


    ####################################################
    #######______New Observation______#######
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
        d0 = jnp.array([0,0])
        cons = []

        # Collect All objective function and constraints(model constraint + trust region)
        for i in range(self.n_fun):
            if i == 0:
                obj_fun = lambda d: (self.model[0](theta, u_0+d) 
                                     + GP_m.GP_inference_np(u_0+d)[0][0] # mean
                                     - b*jnp.sqrt(GP_m.GP_inference_np(u_0+d)[1][0])) # std
            else:
                cons.append({'type': 'ineq',
                             'fun': lambda d: (self.model[i](theta, u_0+d) 
                                               + GP_m.GP_inference_np(u_0+d)[0][i] # mean 
                                               - b*jnp.sqrt(GP_m.GP_inference_np(u_0+d)[1][i]))}) # std
 
        cons.append({'type': 'ineq',
                     'fun': lambda d: r - jnp.linalg.norm(d)})
        
        cons = tuple(cons)

        result = minimize((obj_fun),
                        d0,
                        constraints = cons,
                        method      ='SLSQP',
                        jac         = '3-point',
                        options     = {'ftol': 1e-9})

        return result.x


    #############################################
    #######______Trust Region Update______#######
    #############################################
    def update_trustregion(self):
        pass


if __name__ == '__main__':
    # Test Case 1: Test on GP_Initialization
    ## Initial Parameters
    key = random.PRNGKey(42)
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
    theta,GP_m = BRTO.GP_Initialization(n_s,u_0,theta_0,r_i,plant_system,model,key)
    print("sampled input:")
    print(BRTO.input_sample)

    print("#######______Test Case: WLS______#######")
    print(BRTO.WLS(theta_0,BRTO.input_sample))

    # print(f'parameter theta: {theta}')
    # u_0 = jnp.array([2.15956118, -1.42712019])
    # d_0 = jnp.array([0,0])
    # print(f"input = {u_0}")
    # ## Test if GP model is working fine
    # GP_modifier = GP_m.GP_inference_np(u_0)

    # ## Check if plant and model provides same output using sampled data as input
    # print("\n #___Check if plant and model provides similar output using sampled data as input___#")
    # print(f"plant obj: {plant_system[0](u_0)}")
    # print(f"model obj: {model[0](theta,u_0)+GP_modifier[0][0]}")

    # print(f"plant con: {plant_system[1](u_0)}")
    # print(f"model con: {model[1](theta,u_0)+GP_modifier[0][1]}")
    
    # ## Check if variance is approx 0 at sampled input
    # print(f"variance: {GP_modifier[1]}")

    # # Test Case 2: Test on observation_trustregion
    # print("\n #######______Test Case: optimization aquisition______#######")
    # ## Find info on old observation
    # print("#___Find info on old observation___#")
    # d_new = BRTO.optimize_acquisition(r_i,u_0,theta,GP_m)
    # print(f"optimal old input(model): {u_0}")
    # print(f"optimal old output(model): {model[0](theta,u_0)+GP_modifier[0][0]}")
    # print(f"old constraint(model): {model[1](theta,u_0)+GP_modifier[0][1]}")
    # print(f"GP model: {GP_m.GP_inference_np(u_0)}")


    # ## Find info on new observation
    # GP_modifier = GP_m.GP_inference_np(u_0+d_new)
    # print("\n #___Find info on new observation and see if improved___#")
    # print(f"optimal new input(model): {u_0+d_new}")
    # print(f"Euclidean norm of d_new(model): {jnp.linalg.norm(d_new)}")
    # print(f"optimal new output(model): {model[0](theta,u_0+d_new)+GP_modifier[0][0]}")
    # print(f"new constraint(model): {model[1](theta,u_0+d_new)+GP_modifier[0][1]}")
    # print(f"GP model: {GP_m.GP_inference_np(u_0+d_new)}")

    # ## Check if new observation provides min in trust region
    # print("\n#___Check if plant system agrees with new observation___#")
    # cons = []
    # cons.append({'type': 'ineq',
    #              'fun': lambda u: plant_system[1](u)})
    # cons.append({'type': 'ineq',
    #              'fun': lambda u: r_i - jnp.linalg.norm(u-u_0)})
    # result = minimize((plant_system[0]),
    #             u_0,
    #             constraints = cons,
    #             method      ='SLSQP',
    #             options     = {'ftol': 1e-9})
    # print(f"optimal new input(plant system): {result.x}")
    # print(f"Euclidean norm of new input(plant system): {jnp.linalg.norm(result.x-u_0)}")
    # print(f"optimal new output(plant system): {result.fun}")
    # print(f"new constraint(plant system): {plant_system[1](result.x)}")

    # ########_________Test case 3: Real Time Optimization_________########
    # print("\n ########_________Test case 3: Real Time Optimization_________########")
    # n_sample            = 4
    # u_0                 = jnp.array([4.,-1.])
    # theta_0             = jnp.array([1.,1.,1.,1.])
    # r                   = 1
    # n_iter = 2
    # for i in range(n_iter):
    #     print(f"####___Iteration: {i}___####")
    #     # New observation
    #     d_new = BRTO.optimize_acquisition(r,u_0,theta,GP_m,b=0.1)
    #     # Collect data on new observation
    #     u_new = u_0 + d_new
    #     modifier = BRTO.modifier_calc(theta,u_new)

    #     # Improve GP model using collected data on new observation
    #     GP_m.add_sample(u_new,modifier)

    #     # for next iter + data collection
    #     u_0 = u_new

    #     print(f"u_new: {u_new}")
    #     print(f"d_new: {d_new}")
    #     print(f"mag d_new: {jnp.linalg.norm(d_new)}")
    #     GP_modifier = GP_m.GP_inference_np(u_new)
    #     print(f"obj func after RTO : {Benoit_Problem.Benoit_Model_1(theta,u_new)+GP_modifier[0][0]}")
    #     print(f"const after RTO : {Benoit_Problem.con1_Model(theta,u_new)+GP_modifier[0][1]}")
    #     print(f"plant obj func after RTO : {Benoit_Problem.Benoit_System_1(u_new)}")
    #     print(f"plant const after RTO: {Benoit_Problem.con1_system(u_new)}")
