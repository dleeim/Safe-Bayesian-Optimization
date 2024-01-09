import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt


def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

###########################################
# --- Weighted Least Squares function --- #
###########################################

def Wighted_LS(params, X, Y):
    '''
    X: matrix: [x^(1),...,x^(n_d)]
    Y: vecor:  [f(x^(1)),...,f(x^(n_d))]
    '''
    # renaming parameters for clarity (this can be avoided)
    a = params[0]; b = params[1]; c = params[2]
    # number of datapoints
    n_d = Y.shape[0] 
    # weighted least squares
    WLS = np.sum([(a*X[i][0]+b*X[i][1]+c*X[i][0]*X[i][1] - Y[i])**2/Y[i]**2 for i in range(n_d)])/n_d

    return WLS