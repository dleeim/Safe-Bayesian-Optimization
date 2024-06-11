import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
import random
import BayesOpt
from dataclasses import dataclass

@dataclass
class Point:
    x : float
    y : float

class Bayesian_RTO(BayesOpt.BayesianOpt):

    "It inherits BayesianOpt class (more general model)" 
    "to create a Bayesian Optimization for Real Time Optimization" 

    def __init__(self):
        """
        Initiallize newly created object.

        Global Variables:
        ...
        """        

        BayesOpt.BayesianOpt.__init__(self)

    #######______GP Initialization______#######

    def Ball_sampling(ndim ,r_i):
        '''
        This function samples randomly at (0,0) within a ball of radius r_i.
        By adding sampled point onto initial point (u1,u2) you will get 
        randomly sampled points around (u1,u2)

        Arguments:
            ndim: no of dimensions required for sampled point
            r_i: radius from (0,0) of circle area for sampling

        Returns: 
            d_init: sampled distances from (0,0)
        '''

        u      = np.random.normal(0,1,ndim)  # random sampling in a ball
        norm   = np.sum(u**2)**(0.5)
        r      = random.random()**(1.0/ndim)
        d_init = r*u/norm*r_i*2      # random sampling in a ball

        return d_init


    # 2. Cost Function Optimization
    # 3. Trust Region Update
    # 4. 


if __name__ == '__main__':
    pass