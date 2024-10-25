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

##########_____W_Shape_____###########

# Class Initialization
plant_system = [W_shape_Problem.W_shape]
bound = jnp.array([[-1,2]])
bound_d = jnp.array([[2,4]])
b = 2.
GP_m = StableOpt.BO(plant_system,bound,bound_d,b)

# GP Initialization:
n_sample = 3
x_samples, plant_output = GP_m.Data_sampling_with_perbutation(n_sample)
GP_m.GP_initialization(x_samples,plant_output,'RBF',multi_hyper=5,var_out=True)

print(f"Data Sample Input:")
print(f"{x_samples[:,:GP_m.nxc_dim]}")
print(f"Data Sample Perbutation:")
print(f"{x_samples[:,GP_m.nxc_dim:GP_m.nd_dim+1]}")
print(f"Data Sample Output:")
print(f"{plant_output} \n")

# Tests
def test_GP_inference():
    x = x_samples[1]
    xc = x[:GP_m.nxc_dim]
    d = x[GP_m.nxc_dim:GP_m.nd_dim+1]
    plant = GP_m.calculate_plant_outputs(xc,d)
    print(f"Test: GP Inference; check if Gp inference well with low var")
    print(f"x input: {x}")
    print(f"GP Inference: {GP_m.GP_inference(x,GP_m.inference_datasets)}")
    print(f"Actual plant: {plant}")
    print(f"")

def test_lcb():
    x = x_samples[1]
    xc = x[:GP_m.nxc_dim]
    d = x[GP_m.nxc_dim:GP_m.nd_dim+1]
    print(f"Test: lcb; check if lcb are in appropriate value")
    print(f"x input: {xc,d}")
    print(f"GP Inference: {GP_m.lcb(xc,d,0)}")
    print(f"")

def test_Maximize_d():
    xc = jnp.array([1.8,0.4])
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
    print(f"actual max_output: {max_output}")
    print(f"actual max_x: {max_d}")
    print(f"")

def test_Minimize_Maximise():
    xc0_sample = GP_m.xc0_sampling(n_sample=5)
    xcmin, fmin = GP_m.Minimize_Maximise(GP_m.lcb,xc0_sample)
    print(f"Test: Minimize Maximise; min max lcb")
    print(f"xmin, fmin: {xcmin, fmin} \n")

def test_Maximize_d_with_constraints():
    xcmin = jnp.array([-1.])
    d_max,output = GP_m.Maximise_d_with_constraints(GP_m.ucb,xcmin)
    print(f"Test: Maximize_d_with_constriants; d = argmax ucb s.t constraints")
    print(f"xmin, d_max, output: {xcmin, d_max, output} \n")

def test_StableOpt_W_shape():
    # Class Initialization
    plant_system = [W_shape_Problem.W_shape]
    bound = jnp.array([[-1,2]])
    bound_d = jnp.array([[2,4]])
    b = 2.
    GP_m = StableOpt.BO(plant_system,bound,bound_d,b)

    # Data Storage
    data = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

    # GP Initialization:
    n_sample = 3
    x_samples, plant_output = GP_m.Data_sampling_with_perbutation(n_sample)
    GP_m.GP_initialization(x_samples,plant_output,'RBF',multi_hyper=5,var_out=True)
    data['sampled_x']=x_samples
    data['sampled_output']=plant_output

    # StableOpt
    n_iter = 15

    for i in range(n_iter):
        # Find x and d for sample
        xcmin, fmin = GP_m.Minimize_Maximise(GP_m.lcb)
        print(f"xcmin, fmin: {xcmin, fmin}")
        dmax, fdmax = GP_m.Maximise_d_with_constraints(GP_m.ucb,xcmin)
        print(f"dmax, fdmax: {dmax, fdmax}")
        plant_output = GP_m.calculate_plant_outputs(xcmin,dmax)[0]

        # Add sample into GP
        x = jnp.concatenate((xcmin,dmax))
        GP_m.add_sample(x,plant_output)

        # Store data
        data['observed_x'].append(x)
        data['observed_output'].append(plant_output)

    jnp.savez('data/data_StableOpt_W_shape.npz', **data)

def test_draw_robust(): 
    x = jnp.linspace(bound[:,0],bound[:,1],100)
    outputs = []
    for i in x:
        print(i)
        output = GP_m.Maximise_d(GP_m.lcb,i,0)
        outputs.append(output)
    plt.figure()
    plt.plot(x,outputs)
    plt.show()

def test_multiple_WilliamOttoReactor():
    # Class Initialization
    Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()
    plant_system = [Reactor.get_objective,
                    Reactor.get_constraint1,
                    Reactor.get_constraint2]
    bound = jnp.array([[4.,7.],[70.,100.]])
    b = 2.
    n_start = 1
    data = {}
    noise = 0.001
    Fa = 1.8275
    bound_d = jnp.array([[Fa-noise*b,Fa+noise*b]])
    GP_m = StableOpt.BO(plant_system,bound,bound_d,b)

    for i in range(n_start):
        print(f"iteration: {i}")
        # Data Storage
        data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

        # GP Initialization: 
        x_i = jnp.array([6.8,80.])
        r = 0.3
        n_sample = 5
        X_sample = jnp.empty((0,len(x_i)))
        Y_sample = jnp.empty((0,len(plant_system)))
        for count in range(n_sample):
            X,Y = GP_m.Data_sampling(1,x_i,r,noise)
            Reactor.noise_generator
            X_sample=jnp.append(X_sample,X,axis=0)
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
        n_iter = 10

        for i in range(n_iter):
            xcmin, fmin = GP_m.Minimize_Maximise(GP_m.lcb)
            print(f"xcmin, fmin: {xcmin, fmin}")
            dmax, fdmax = GP_m.Maximise_d_with_constraints(GP_m.ucb,xcmin)
            print(f"dmax, fdmax: {dmax, fdmax}")
            plant_output = GP_m.calculate_plant_outputs(xcmin,dmax)[0]

            # Add sample into GP
            x = jnp.concatenate((xcmin,dmax))
            GP_m.add_sample(x,plant_output)

            # Store data
            data['observed_x'].append(xcmin)
            data['observed_output'].append(plant_output)
        
        jnp.savez('data/data_StableOpt_WilliamOttoReactor.npz', **data)
        

if __name__ == "__main__":
    # test_GP_inference()
    # test_lcb()
    # test_Maximize_d()
    # test_Minimize_Maximise()
    # test_Maximize_d_with_constraints()
    # test_StableOpt_W_shape()
    # test_draw_robust()
    test_multiple_WilliamOttoReactor()
    pass

