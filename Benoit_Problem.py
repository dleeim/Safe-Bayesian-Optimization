import numpy as np

# Actual Plant System (if noise exists it equals to np.sqrt(1e-3))
def Benoit_System_1(u, noise = 0):

    f = u[0] ** 2 + u[1] ** 2 + u[0] * u[1]
    if noise: 
        f += np.random.normal(0., np.sqrt(noise))

    return f

def Benoit_System_2(u, noise = 0):

    f = u[0] ** 2 + u[1] ** 2 + (1 - u[0] * u[1])**2
    if noise: 
        f += np.random.normal(0., np.sqrt(noise))

    return f


def con1_system(u, noise = 0):

    g1 = 1. - u[0] + u[1] ** 2 + 2. * u[1] - 2.
    if noise:
        g1 += np.random.normal(0., np.sqrt(noise))

    return -g1


def con1_system_tight(u, noise = 0):
    
    g1 = 1. - u[0] + u[1] ** 2 + 2. * u[1] 
    if noise:
        g1 += np.random.normal(0., np.sqrt(noise))

    return -g1


# Model of Plant System
def Benoit_Model_1(theta, u):
    f = theta[0] * u[0] ** 2 + theta[1] * u[1] ** 2
    return f

def con1_Model(theta, u):
    g1 = 1. - theta[2]*u[0] + theta[3]*u[1] ** 2
    return -g1