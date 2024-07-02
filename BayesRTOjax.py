import jax.numpy as jnp
import jax
from jax import grad, value_and_grad, jit, vmap
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import BayesOptjax
import Benoit_Problem
import sobol_seq


class Bayesian_RTO():
    
    def __init__(self) -> None:
        '''
        Global Variables:
            - n_sample                      : number of sample given
            - intput_dim                    : dimension of input
            - n_fun                         : number of obj functions and constraints we want to measure
            - plant_system                  : array with obj functions and constraints for plant system
            - model                         : array with obj functions and constraints for model
            - input_sample                  : sampled input data made from Ball_sampling and used for GP initialization
        '''
        self.n_sample                       = 0
        self.input_dim                      = 0
        self.n_fun                          = 0
        self.plant_system                   = 0
        self.model                          = 0
        self.input_sample                   = 0
        self.key                            = jax.random.PRNGKey(40)

    ###########################################
    #######______GP Initialization______#######
    ###########################################

    def Ball_sampling(self,input_dim,n_sample,r_i,key):
        '''
        Description:
            This function samples randomly at (0,0) within a ball of radius r_i.
            By adding sampled point onto initial point (u1,u2) you will get 
            randomly sampled points around (u1,u2)
        Arguments:
            - input_dim                     : no of dimensions required for sampled point
            - n_sample                      : number of sample required to create
            - r_i                           : radius from (0,0) of circle area for sampling
        Returns: 
            - d_init                        : sampled distances from (0,0)
        '''
        u                                   = jax.random.normal(key, (n_sample,input_dim))
        norm                                = jnp.sqrt(jnp.sum(u**2))
        r                                   = jax.random.uniform(key, (n_sample,1))**(1.0 /input_dim)
        d_init                              = r*u / norm*r_i*2

        return d_init
    
    def GP_Initialization(self,n_sample,input_0,r,plant_system,multi_start=2):
        '''
        Description:
            Initialize GP model by:
                1) Create a sample data in matrix
                4) Use data to initialize GP using BayesOpt
        Argument:
            - n_sample                      : number of sample points that can be collected
            - input_0                       : initial input
            - theta_0                       : initial estimated parameter
            - r                             : radius of area where samples are collected in Ball_sampling
            - plant_system                  : numpy array with objective functions and constraints for plant system
        Returns
            - GP_m                          : GP model that is initialized using random sampling around initial input
        '''
       # === Define relavent parameters and arrays === #
        self.n_sample                       = n_sample
        self.input_dim                      = jnp.shape(input_0)[0]
        self.n_fun                          = len(plant_system)
        self.plant_system                   = plant_system

        # === Collect Training Dataset (Input) === #
        self.key, subkey                    = jax.random.split(self.key)                              
        u_trial                             = self.Ball_sampling(self.input_dim,n_sample,r,subkey)
        u_trial                             += input_0
        
        # To store the sampled input data and to see when using the class
        self.input_sample                   = u_trial

        # === Collect Training Dataset === #
        output                              = jnp.zeros((n_sample,self.n_fun))
        for i in range(self.n_fun):
            output                          = output.at[:,i].set(vmap(self.plant_system[i])(self.input_sample))
        jax.debug.print("output: {}", output)
        # === Initialize GP with modifier === #
        GP_m                                = BayesOptjax.BayesianOpt(self.input_sample, output, 'RBF', 
                                                                      multi_hyper=multi_start, var_out=True)
        
        return GP_m
    
    ####################################################
    ############______New Observation______#############
    ####################################################

    ############______methods for obj_fun, constraint______############
    def obj_fun(self, d, input_0, GP_m, b):
        GP_inference                        = GP_m.GP_inference_np(input_0 + d)
        mean                                = GP_inference[0][0]
        std                                 = jnp.sqrt(GP_inference[1][0])
        value                               = mean - b*std
        jax.debug.print("objfun: {}", value)
        return value

    def obj_fun_grad(self, d, input_0, GP_m, b):
        value                               = grad(self.obj_fun,argnums=0)(d, input_0, GP_m, b)
        jax.debug.print("objfungrad: {}", value)
        return value 

    def constraint(self, d, input_0, GP_m, index, b):
        GP_inference                        = GP_m.GP_inference_np(input_0 + d)
        mean                                = GP_inference[0][index]
        std                                 = jnp.sqrt(GP_inference[1][index])
        value                               = mean - b*std
        jax.debug.print("const: {}", value)

        return value
    
    def constraint_grad(self, d, input_0, GP_m, index, b):
        value                               = grad(self.constraint,argnums=0)(d, input_0, GP_m, index, b)
        jax.debug.print("constgrad: {}", value)

        return value
    
    def TR_constraint(self,d,r):
        value = r - jnp.linalg.norm(d)
        jax.debug.print("TR: {}", value)

        return value

    def TR_constraint_grad(self,d,r):
        value                               = grad(self.TR_constraint,argnums=0)(d,r)
        jax.debug.print("Trgrad: {}", value)

        return value
    
    ############______optimization of acquisition function______############
    def optimize_acquisition(self,r,input_0,GP_m,b=0,multi_start=1):
        '''
        Description:
            Find minimizer x* in a Gaussian Process Lower Confidence Bound
        Argument:
            - r                             : radius of trust region area
            - input_0                       : previous input observed
            - GP_m                          : Gaussian Process Model
            - b                             : parameter for exploration
        Results:
            - result.x                      : a distance from input input_0 to observe the 
                                              minimal output of function
            - result.fun                    : minimal output of function
        '''
        options                             = {'disp':False, 'maxiter':10000} 
        cons                                = []
        localsol                            = [0.]*multi_start
        localval                            = jnp.zeros((multi_start))

        # Collect All objective function and constraints(model constraint + trust region)
        for i in range(self.n_fun):
            if i == 0:
                obj_fun                     = lambda d: self.obj_fun(d, input_0, GP_m, b)
                obj_grad                    = lambda d: self.obj_fun_grad(d, input_0, GP_m, b)
            else: 
                cons.append({'type'         : 'ineq',
                             'fun'          : lambda d: self.constraint(d, input_0, GP_m, i, b),
                             'jac'          : lambda d: self.constraint_grad(d, input_0, GP_m, i, b)
                             })
                
        cons.append({'type'                 : 'ineq',
                     'fun'                  : lambda d: self.TR_constraint(d,r),
                     'jac'                  : lambda d: self.TR_constraint_grad(d,r)
                     })
        
        d0                                  = self.Ball_sampling(self.input_dim,multi_start,r)
        for j in range(multi_start):
            d0_j                            = d0[j,:]
            print(f"iteration: {j}")
            print(f"initial d0: {d0_j}")

            res                             = minimize(obj_fun, d0_j, constraints=cons, method='SLSQP', 
                                                       jac=obj_grad,options=options)
            localsol[j] = res.x
            localval = localval.at[j].set(res.fun)

            print(f"final d0: {res.x}")
            print(f"final fun: {res.fun}")

        minindex = jnp.argmin(localval)
        xopt = localsol[minindex]
        funopt = localval[minindex]
        
        return xopt, funopt
    



if __name__ == '__main__':
    
    ##########################################
    ##### Test Case 1: GP_Initialization #####
    ##########################################
    print("################################################")
    print("####______Test Case: GP_Initialization______####")
    print("################################################")

    ## Initial Parameters
    BRTO = Bayesian_RTO()
    u_0 = jnp.array([1.4,-0.8])
    r_i = 1.
    n_s = 4
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system]
    
    ## GP Initialization
    GP_m = BRTO.GP_Initialization(n_s,u_0,r_i,plant_system)
    print(f"initial input: {u_0}")
    print("sampled input:")
    print(BRTO.input_sample)

    # Test Gaussian Process Model inference
    GP_inference = GP_m.GP_inference_np(u_0)

    ## Check if plant and model provides similar output using sampled data as input
    print("\n#___Check if plant and model provides similar output using sampled data as input___#")
    print(f"plant obj: {plant_system[0](u_0)}")
    print(f"model obj: {GP_inference[0][0]}")
    print(f"plant con: {plant_system[1](u_0)}")
    print(f"model con: {GP_inference[0][1]}")

    ## Check if variance is approx 0 at sampled input
    print("\n#___Check variance at sampled input___#")
    print(f"variance: {GP_inference[1]}")

    ## Check if gradient of GP_inference works. 
    print("\n#___Check if gradient of GP_inference works___#")
    u_0 = jnp.array([1.1304393,  -0.67907566])
    print(f"chose new distance and input as: {u_0}")
    GP_inference = GP_m.GP_inference_np(u_0)
    print(f"new output: {GP_inference[0][0]}")
    
    def GP_infergrad(u):
        return GP_m.GP_inference_np(u)[0][0]
    GP_inferencegrad = grad(GP_infergrad,argnums=0)
    
    print(f"JAX grad of GP_inference w.r.t u as: {GP_inferencegrad(u_0)}")
    
    u_1 = jnp.array([1.130,  -0.67907566])
    gradient = (GP_m.GP_inference_np(u_1)[0][0]-GP_inference[0][0])/(u_1[0]-u_0[0])
    print(f"output with different 1st dimension: {GP_m.GP_inference_np(u_1)[0][0]}")
    print(f"estimated gradient with 1st dimension: {gradient}")

    u_2 = jnp.array([1.1304393,  -0.68])
    gradient = (GP_m.GP_inference_np(u_2)[0][0]-GP_inference[0][0])/(u_2[1]-u_0[1])
    print(f"output with different 2st dimension: {GP_m.GP_inference_np(u_2)[0][0]}")
    print(f"estimated gradient with 2st dimension: {gradient}")
    


    #############################################################
    #### Test Case 2: Optimization of Lower Confidence Bound ####
    #############################################################
    print("\n")
    print("#####################################################################")
    print("####______Test Case: Optimization of Lower Confidence Bound______####")
    print("#####################################################################")

    d_new, obj = BRTO.optimize_acquisition(r_i,u_0,GP_m,b=0,multi_start=5)
    print(f"optimal new input(model): {u_0+d_new}")
    print(f"corresponding new output(model): {obj}")
    print(f"Euclidean norm of d_new(model): {jnp.linalg.norm(d_new)}")

    print(f"\n#___Check Gaussian Process Output and Plant system output___#")
    GP_inference = GP_m.GP_inference_np(u_0+d_new)
    print(f"new output(model): {GP_inference[0][0]}")
    print(f"new constraint(model): {GP_inference[0][1]}")
    print(f"new variances(model): {GP_inference[1]}")
    print(f"new plant output: {plant_system[0](u_0+d_new)}")
    print(f"new plant constraint: {plant_system[1](u_0+d_new)}")
    
    print(f"\n#___Check if model output is actually minimial by comparing with the following value___#")
    GP_inference = GP_m.GP_inference_np(jnp.array([3.,-1.]))
    print(f"output at [3.,-1.](model): {GP_inference[0][0]}")

    print(f"\n#___Check Add Sample___#")
    output_new = []
    for plant in plant_system:
        output_new.append(plant(u_0+d_new))
    print(f"add sample: {u_0+d_new, output_new}")
    GP_m.add_sample(u_0+d_new,output_new)
    GP_inference = GP_m.GP_inference_np(u_0+d_new)
    print(f"output after add sample(model): {GP_inference[0][0]}")
    print(f"constraint after add sample(model): {GP_inference[0][1]}")
    print(f"new variances(model): {GP_inference[1]}")
    print(f"plant output: {plant_system[0](u_0+d_new)}")
    print(f"plant constraint: {plant_system[1](u_0+d_new)}")

    #############################################
    #### Test Case 3: Real Time Optimization ####
    #############################################
    print("\n")
    print("#####################################################")
    print("####______Test Case: Real Time Optimization______####")
    print("#####################################################")

    # GP Initialization
    BRTO = Bayesian_RTO()
    u_0 = jnp.array([1.4,-0.8])
    r_i = 1.
    n_s = 4
    n_iter = 10
    b = 1
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system]
    GP_m = BRTO.GP_Initialization(n_s,u_0,r_i,plant_system,)
    
    for i in range(n_iter):
        print(f"\n####___Iteration: {i}___####")

        # 1. New observation 
        d_new, obj = BRTO.optimize_acquisition(r_i,u_0,GP_m,b,multi_start=5)

        # 2. Collect data
        u_new = u_0+d_new
        output_new = []
        for plant in plant_system:
            output_new.append(plant(u_new))
        
        # 3. Re-initialize Gaussian Process with new data
        GP_m.add_sample(u_new,output_new)

        # 4. Preparation for next iteration:
        u_0 = u_new

        # 5. Store Data
        print(f"\n#___Check Gaussian Process Output and Plant system output___#")
        GP_inference = GP_m.GP_inference_np(u_new)
        print(f"optimal new input(model): {u_new}")
        print(f"new output(model): {GP_inference[0][0]}")
        print(f"new constraint(model): {GP_inference[0][1]}")
        print(f"new variances(model): {GP_inference[1]}")
        for plant in plant_system:
            print(f"new plant output: {plant(u_0+d_new)}")
            print(f"new plant constraint: {plant(u_0+d_new)}")








    

