import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit, random
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
from models import StableOpt
from problems import W_shape_Problem, WilliamOttoReactor_Problem
import warnings
from utils import utils_SafeOpt
jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# Class Initialization
Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor(measure_disturbance=True)
plant_system = [Reactor.get_objective,
                Reactor.get_constraint1,
                Reactor.get_constraint2]
bound = jnp.array([[4.,7.],[70.,100.]])
b = 2.
n_start = 1
data = {}
noise = 0.001
Fa = 1.8275
bound_d = jnp.array([[Fa-jnp.sqrt(noise)*b,Fa+jnp.sqrt(noise)*b]])
GP_m = StableOpt.BO(plant_system,bound,bound_d,b)

x_i = jnp.array([6.8,80.])
r = 1.
n_sample = 5
X_sample = jnp.empty((0,len(bound)+len(bound_d)))
Y_sample = jnp.empty((0,len(plant_system)))
for count in range(n_sample):
    X,Y,D = GP_m.Data_sampling_output_and_perturbation(1,x_i,r,noise)
    Reactor.noise_generator()
    X_sample=jnp.append(X_sample,jnp.hstack((X,D)),axis=0)
    Y_sample=jnp.append(Y_sample,Y,axis=0)

GP_m.GP_initialization(X_sample, Y_sample, 'RBF', multi_hyper=5, var_out=True)

print(f"\n")
print(f"Data Sample Input:")
print(f"{X_sample}")
print(f"Data Sample Output:")
print(f"{Y_sample}")
print(f"")

# Tests
def test_GP_inference():
    x = X_sample[1]
    xc = x[:GP_m.nxc_dim]
    d = x[GP_m.nxc_dim:]
    plant = GP_m.calculate_plant_outputs(xc,noise)
    print(f"Test: GP Inference; check if Gp inference well with low var")
    print(f"x input: {xc}")
    print(f"GP Inference: {GP_m.GP_inference(x,GP_m.inference_datasets)}")
    print(f"Actual plant: {plant}")
    print(f"")

def test_lcb():
    x = X_sample[1]
    xc = x[:GP_m.nxc_dim]
    d = x[GP_m.nxc_dim:]
    print(f"Test: lcb; check if lcb are in appropriate value")
    print(f"x input: {xc}")
    print(f"disturbance: {d}")
    print(f"GP Inference: {GP_m.lcb(xc,d,0)}")
    print(f"")

def test_Maximize_d():
    x = X_sample[1]
    xc = x[:GP_m.nxc_dim]
    d = x[GP_m.nxc_dim:]
    max_robust_f = GP_m.Maximise_d(GP_m.lcb,xc,0)
    print(f"Test: robust filter; check if robust filter is in appropriate value")
    print(f"xc input: {xc}")
    print(f"max_robust_f: {max_robust_f}")

    d = jnp.linspace(bound_d[:,0],bound_d[:,1],100)
    max_output = -jnp.inf
    for i in d:
        output = GP_m.lcb(xc,i,0)
        if output>max_output:
            max_d = i
            max_output = output
    print(f"result from random search: max_d: {max_d}")
    print(f"research from random search: actual max_output: {max_output}")
    print(f"")

def test_Minimize_d():
    x = X_sample[1]
    xc = x[:GP_m.nxc_dim]
    d = x[GP_m.nxc_dim:]
    max_robust_f = GP_m.Minimise_d(GP_m.lcb,xc,0)
    print(f"Test: robust filter; check if robust filter is in appropriate value")
    print(f"xc input: {xc}")
    print(f"min_robust_f: {max_robust_f}")

    d = jnp.linspace(bound_d[:,0],bound_d[:,1],100)
    min_output = np.inf
    for i in d:
        output = GP_m.lcb(xc,i,0)
        if output<min_output:
            min_d = i
            min_output = output
    print(f"result from random search: min_d: {min_d}")
    print(f"research from random search: actual min_output: {min_output}")
    print(f"")

def test_Minimize_Maximise():
    xcmin, fmin = GP_m.Minimize_Maximise(GP_m.lcb)
    print(f"Test: Minimize Maximise; min max lcb")
    print(f"xmin, fmin: {xcmin, fmin} \n")

def test_Maximize_d_with_constraints():
    x = X_sample[1]
    xcmin = x[:GP_m.nxc_dim]
    d_max,output = GP_m.Maximise_d_with_constraints(GP_m.ucb,xcmin)

    d = jnp.linspace(bound_d[:,0],bound_d[:,1],100)
    max_output = -jnp.inf
    for i in d:
        output = GP_m.lcb(xcmin,i,0)
        if output>max_output:
            max_d = i
            max_output = output
    constraint1 = GP_m.lcb(xcmin,d_max,1)
    constraint2 = GP_m.lcb(xcmin,d_max,2)    
    constraint1_randomsearch = GP_m.lcb(xcmin,max_d,1)
    constraint2_randomsearch = GP_m.lcb(xcmin,max_d,2)
    print(f"Test: Maximize_d_with_constriants; d = argmax ucb s.t constraints")
    print(f"Result form Maximize_d_with_constraints d_max, output,constraint1,constraint2 {d_max,output,constraint1,constraint2}")
    print(f"result from random search: min_d: max_d, output, constraint1, constraint2: {max_d, max_output,constraint1_randomsearch,constraint2_randomsearch} \n")
    print(f"")


def test_multiple_WilliamOttoReactor():
    # Class Initialization
    Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor(measure_disturbance=True)
    plant_system = [Reactor.get_objective,
                    Reactor.get_constraint1,
                    Reactor.get_constraint2]
    bound = jnp.array([[4.,7.],[70.,100.]])
    b = 2.
    n_start = 1
    data = {}
    noise = 0.001
    Fa = 1.8275
    bound_d = jnp.array([[Fa-jnp.sqrt(noise)*b,Fa+jnp.sqrt(noise)*b]])
    GP_m = StableOpt.BO(plant_system,bound,bound_d,b)

    for i in range(n_start):
        print(f"iteration: {i}")
        # Data Storage
        data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

        # GP Initialization: 
        x_i = jnp.array([6.8,80.])
        r = 1.
        n_sample = 5
        X_sample = jnp.empty((0,len(bound)+len(bound_d)))
        Y_sample = jnp.empty((0,len(plant_system)))
        for count in range(n_sample):
            X,Y,D = GP_m.Data_sampling_output_and_perturbation(1,x_i,r,noise)
            Reactor.noise_generator()
            X_sample=jnp.append(X_sample,jnp.hstack((X,D)),axis=0)
            Y_sample=jnp.append(Y_sample,Y,axis=0)

        GP_m.GP_initialization(X_sample, Y_sample, 'RBF', multi_hyper=5, var_out=True)
        
        data[f'{i}']['sampled_x'] = X_sample
        data[f'{i}']['sampled_output'] = Y_sample

        print(f"\n")
        print(f"Data Sample Input:")
        print(f"{X_sample}")
        print(f"Data Sample Output:")
        print(f"{Y_sample}")
        print(f"")

        # StableOpt
        n_iter = 1

        for i in range(n_iter):
            xcmin, fmin = GP_m.Minimize_Maximise(GP_m.lcb)
            print(f"xcmin, fmin: {xcmin, fmin}")
            dmax, fdmax = GP_m.Maximise_d_with_constraints(GP_m.ucb,xcmin)
            print(f"dmax, fdmax: {dmax, fdmax}")
            plant_output,disturbance = GP_m.calculate_plant_outputs(xcmin,noise)

            # Add sample into GP
            x = jnp.concatenate((xcmin,disturbance))
            GP_m.add_sample(x,plant_output)

            # Store data
            data['observed_x'].append(xcmin)
            data['observed_output'].append(plant_output)
        
    jnp.savez('data/data_StableOpt_WilliamOttoReactor.npz', **data)
        

if __name__ == "__main__":
    # test_GP_inference()
    # test_lcb()
    # test_Maximize_d()
    # test_Minimize_d()
    # test_Minimize_Maximise()
    test_Maximize_d_with_constraints()
    # test_multiple_WilliamOttoReactor()
    pass

