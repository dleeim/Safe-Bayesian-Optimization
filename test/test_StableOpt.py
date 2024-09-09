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
from problems import W_shape_Problem
import warnings
from utils import utils_SafeOpt
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# Class Initialization
plant_system = [W_shape_Problem.W_shape]
bound = jnp.array([[-1,2]])
bound_d = jnp.array([[2,4]])
b = 3.
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
    xc = jnp.array([1.99855295])
    max_robust_f = GP_m.Maximise_d(GP_m.lcb,xc,0)
    print(f"Test: robust filter; check if robust filter is in appropriate value")
    print(f"xc input: {xc}")
    print(f"max_robust_f: {max_robust_f}")
    print(f"")

def test_Minimize_Maximise():
    xcmin, fmin = GP_m.Minimize_Maximise(GP_m.lcb)
    print(f"Test: Minimize Maximise; min max lcb")
    print(f"xmin, fmin: {xcmin, fmin} \n")

def test_Maximize_d_with_constraints():
    xcmin = GP_m.xc_min[0]
    d_max,output = GP_m.Maximise_d_with_constraints(GP_m.ucb,xcmin)
    print(f"Test: Maximize_d_with_constriants; d = argmax ucb s.t constraints")
    print(f"xmin, d_max, output: {xcmin, d_max, output} \n")

def test_Best_xc_Guess():
    xc_min_of_min = GP_m.Best_xc_Guess(GP_m.ucb)
    print(f"Test: Best_xc_Guess; check if Best variable without perbutation guess is made")
    print(f"xcmin: {xc_min_of_min} \n")

def test_StageOpt():
    # Class Initialization
    plant_system = [W_shape_Problem.W_shape]
    bound = jnp.array([[-1,2]])
    bound_d = jnp.array([[2,4]])
    b = 3.
    GP_m = StableOpt.BO(plant_system,bound,bound_d,b)

    # GP Initialization:
    n_sample = 3
    x_samples, plant_output = GP_m.Data_sampling_with_perbutation(n_sample)
    GP_m.GP_initialization(x_samples,plant_output,'RBF',multi_hyper=5,var_out=True)

    # StageOpt
    n_iter = 10
    data = {'xc_min_of_min':[],'output':[],'gridworld_input':[],'gridworld_output':[]}
    x_grid = jnp.linspace(bound[:,0],bound[:,1],100)
    data['gridworld_input'].append(x_grid)

    for i in range(n_iter):
        xcmin, fmin = GP_m.Minimize_Maximise(GP_m.mean)
        dmax, dmax_output = GP_m.Maximise_d_with_constraints(GP_m.mean,xcmin)
        plant_output = GP_m.calculate_plant_outputs(xcmin,dmax)[0]
        
        x = jnp.concatenate((xcmin,dmax))
        print(f"x, plantoutput for add sample: {x,plant_output}")
        GP_m.add_sample(x,plant_output)
        
        xc_min_of_min,d = GP_m.Best_xc_Guess(GP_m.mean)
        plant_output_min = GP_m.calculate_plant_outputs(xc_min_of_min,d)
        print(f"xc_min_of_min, plant output{xc_min_of_min, plant_output_min}")

        # Store data
        data['xc_min_of_min'].append(xc_min_of_min), data['output'].append(plant_output_min)
        outputs = []
        for i in x_grid:
            output = GP_m.Maximise_d(GP_m.lcb,i,0)
            outputs.append(output)
        data['gridworld_output'].append(outputs)

    jnp.savez('data/data_StableOpt.npz', **data)


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

if __name__ == "__main__":
    # test_GP_inference()
    # test_lcb()
    # test_Maximize_d()
    # test_Minimize_Maximise()
    # test_Maximize_d_with_constraints()
    # test_Best_xc_Guess()
    test_StageOpt()
    # test_draw_robust()
    pass