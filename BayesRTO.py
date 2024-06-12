import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
import random
import BayesOpt
from dataclasses import dataclass

class Bayesian_RTO(BayesOpt.BayesianOpt):

    def __init__(self) -> None:
        """
        Global Variables:
            X: input data for training GP
            Y: output data for training GP 
            kernel: 
            multi_hyper:
            var_out: 
        """        
        BayesOpt.BayesianOpt.__init__(self)

    ###########################################
    #######______GP Initialization______#######
    ###########################################

    # 2. Cost Function Optimization
    # 3. Trust Region Update
    # 4. 


if __name__ == '__main__':
    pass
