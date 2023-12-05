import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def sample(bounds,n):
    '''
    Make a sample of X,Y with n data points under bounds boundary
    '''
    d = len(bounds)
    b_range = bounds[:,1] - bounds[:,0]
    X = (np.random.uniform(0,1,(n,d))*b_range) + bounds[:,0]
    Y = np.array([[f(xi)] for xi in X])
    return X,Y

def create_gp(X,Y):
    '''
    create a Gaussian Process Model by training the sample X,Y
    '''
    kernel = GPy.kern.RBF(input_dim=d,ARD=True)
    m = GPy.models.GPRegression(X,Y,kernel)
    m.optimize_restarts(10)
    return m

def aquisition_function(x,m,b):
    '''
    return an output of predicted objective function 
    using Lower Confidence Bound
    '''
    mean, var = m.predict(x)
    output = mean - b * var
    return output

def optimize_aquisition(m,aquisition_function,bounds,x0):
    '''
    Find an input x that its output is at minimal point
    Arguments: bounds = input boundary, x0 = initial guess
    '''
    res = minimize(aquisition_function,x0,args=(m),bounds=bounds)
    optimal_input = res.x
    return optimal_input

def add_sample(optimal_input,X,Y):
    '''
    Use new input from optimize_aquisition to find corresponding output
    and insert into matrices of all the sampled data'''
    new_input = np.array([[optimal_input]])
    new_output = np.array([[f(optimal_input)]])
    new_X = np.append(X,new_input,axis=0)
    new_Y = np.append(Y,new_output,axis=0)
    return new_X,new_Y
