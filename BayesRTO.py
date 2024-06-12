import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
import random
import BayesOpt
import Benoit_Problem
from dataclasses import dataclass

np.random.seed(42)

class Bayesian_RTO():

    def __init__(self) -> None:
        '''
        Global Variables:
            n_sample                        : number of sample given
            u_dim                           : dimension of input
            n_function                      : number of obj functions and constraints we want to measure
            plant_system                    : array with obj functions and constraints for plant system
            model                           : array with obj functions and constraints for model
        '''
        self.n_sample                       = 0
        self.u_dim                          = 0
        self.n_function                     = 0
        self.plant_system                   = 0
        self.model                          = 0

    ###########################################
    #######______GP Initialization______#######
    ###########################################

    def Ball_sampling(self,ndim,r_i) -> float:
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
        u                                   = np.random.normal(0,1,ndim)  
        norm                                = np.sum(u**2)**(0.5)
        d_init                              = u/norm*r_i      

        return d_init

    def WLS(self,theta,u_sample) -> float:
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
            u                               = u_sample[i,:]

            for j in range(self.n_function):
                error                       += (self.plant_system[j](u) - self.model[j](theta,u))**2/np.abs(self.plant_system[j](u))

        return error     

    def parameter_estimation(self,u_sample,theta):
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
        sol                                 = minimize(self.WLS, args=(u_sample), x0=theta, method='SLSQP')
        return sol.x

    def modifier_calc(self,u_sample,theta):
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
        modifier                            = np.zeros((self.n_sample,self.n_function))

        for i in range(self.n_sample):
            u                               = u_sample[i,:]

            for j in range(self.n_function):
                modifier[i][j]              = self.plant_system[j](u) - self.model[j](theta,u)

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
            GP_m                            : GP model that is initialized using random sampling around initial input
        '''
        # === Define relavent parameters and arrays === #
        self.n_sample                       = n_sample
        self.u_dim                          = np.shape(u_0)[0]
        self.n_function                     = len(plant_system)
        self.plant_system                   = plant_system
        self.model                          = model
        u_sample                            = np.zeros((self.n_sample,self.u_dim))

        # === Collect Training Dataset (Input) === #
        for sample_i in range(n_sample):
            u_trial                         = u_0 + self.Ball_sampling(self.u_dim,r)
            u_sample[sample_i]              = u_trial
        
        # === Estimate the parameter theta === #
        theta                               = self.parameter_estimation(u_sample,theta_0)

        # === Collect Training Dataset === #
        modifier                            = self.modifier_calc(u_sample,theta)

        # === Initialize GP with modifier === #
        GP_m                                = BayesOpt.BayesianOpt(u_sample, modifier, 'RBF', multi_hyper=1, var_out=True)

        return u_sample,theta,GP_m

    # 2. Cost Function Optimization
    # 3. Trust Region Update
    # 4. 


if __name__ == '__main__':
    # Test Case 1: Test on GP Initialization
    BRTO = Bayesian_RTO()
    theta_0 = [1,1,1,1]
    u_0 = [4,-1]
    r_i = 1
    n_s = 4
    plant_system = [Benoit_Problem.Benoit_System_1
                    ,Benoit_Problem.con1_system]

    model = [Benoit_Problem.Benoit_Model_1,
            Benoit_Problem.con1_Model]

    u_sample,theta,GP_m = BRTO.GP_Initialization(n_s,u_0,theta_0,r_i,plant_system,model)
    theta = [ 0.91169766, -0.70738436, 1.51734629, -0.26877567]
    
    u = u_sample[0]
    modifier = GP_m.GP_inference_np(u)

    print(f"plant obj: {plant_system[0](u)}")
    print(f"model obj: {model[0](theta,u)+modifier[0][0]}")
    print(f"var obj: {model[0](theta,u)+modifier[1][0]}")

    print(f"plant con: {plant_system[1](u)}")
    print(f"model con: {model[1](theta,u)+modifier[0][1]}")
    print(f"var con: {model[1](theta,u)+modifier[1][1]}")
