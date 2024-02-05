import numpy as np
import random
from scipy.optimize import minimize

# Source: ML4ChemEng_DataDrivenOpt.ipynb from Dr Antonio Del Rio Chanona
def Ball_sampling(ndim, r_i):
    '''
    This function samples randomly withing a ball of radius r_i
    '''
    u      = np.random.normal(0,1,ndim)  # random sampling in a ball
    norm   = np.sum(u**2)**(0.5)
    r      = random.random()**(1.0/ndim)
    d_init = r*u/norm*r_i*2      # random sampling in a ball

    return d_init

# Plant Model 
def Benoit_Model_nomodifier(u,theta):
    f = theta[0] * u[0] ** 2 + theta[1] * u[1] ** 2
    return f

def con1_model_nomodifier(u,theta):
    g1 = 1. - theta[2]*u[0] + theta[3]*u[1] ** 2 
    return -g1

# Plant Model 
def Benoit_Model_nomodifier_1(theta,u):
    '''
    Benoit Model without modifier but parameter 'theta' is 
    in front for scipy minimize function based on theta
    '''
    f = theta[0] * u[0] ** 2 + theta[1] * u[1] ** 2
    return f

def con1_model_nomodifier_1(theta,u):
    '''
    Benoit Model without modifier but parameter 'theta' is 
    in front for scipy minimize function based on theta
    '''
    g1 = 1. - theta[2]*u[0] + theta[3]*u[1] ** 2 
    return -g1

# Actual Plant System
def Benoit_System_1(u):
    f = u[0] ** 2 + u[1] ** 2 + u[0] * u[1] + np.random.normal(0., np.sqrt(1e-3))
    return f

def con1_system_tight(u):
    g1 = 1. - u[0] + u[1] ** 2 + 2. * u[1] + np.random.normal(0., np.sqrt(1e-3))
    return -g1

# Weight Least Square for Benoit Model without any modifier
def WLS_nomodifier(theta,U,plant_func,plant_constraint):
    '''
    This function finds the sum of weight least square between 
    plant and model function and constraints without modifiers
    '''
    n_s = U.shape[0]
    error = 0

    for i in range(n_s):
        plant_func_val = plant_func(U[i])
        plant_constraint_val = plant_constraint(U[i])
        error += (Benoit_Model_nomodifier_1(theta,U[i,:]) - plant_func_val)**2/np.abs(plant_func_val)
        error += (con1_model_nomodifier_1(theta,U[i,:]) - plant_constraint_val)**2/np.abs(plant_constraint_val)

    return error

def parameter_estimation_nomodifier(U,theta_0,plant_func,plant_constraint):
    '''
    U: matrix: [x^(1),...,x^(n_d)]
    Y: vecor:  [f(x^(1)),...,f(x^(n_d))]

    This function estimates parameters of the model using sample U,Y
    '''
    # minimizing benoit's model and constraint with scipy minimize
    res         = minimize(WLS_nomodifier, args=(U,plant_func,plant_constraint), x0=theta_0, method='COBYLA')
    # obtaining solution
    params      = res.x
    #f_val  = res.fun

    return params

u_0 = [2,-1]
n_s = 5
r = 0.2
# === first sampling inside the radius === #
u_dim                           = np.shape(u_0)[0]                         # dimension for input
u_sample                        = np.zeros((n_s,u_dim))
modifier_sample                 = np.zeros((n_s,2))          # sample for plant output and constraint output

# === Collect Training Dataset (Input) === #
for sample_i in range(n_s):                       
    u_trial                     = u_0 + Ball_sampling(u_dim, r)
    u_sample[sample_i]          = u_trial
theta = [1,1,1,1]
params = parameter_estimation_nomodifier(u_sample,theta,Benoit_System_1,con1_system_tight)

print(params)
