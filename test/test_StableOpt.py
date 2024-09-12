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
b = 3
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
    xcmin, fmin = GP_m.Minimize_Maximise(GP_m.lcb)
    print(f"Test: Minimize Maximise; min max lcb")
    print(f"xmin, fmin: {xcmin, fmin} \n")

    # # Create meshgrid from the boundaries
    # xc = jnp.linspace(bound[:, 0], bound[:, 1], 100)
    # d = jnp.linspace(bound_d[:, 0], bound_d[:, 1], 100)
    # xc, d = jnp.meshgrid(xc.flatten(), d.flatten())

    # # Initialize the outputs array with correct shape (100, 100)
    # outputs = []

    # # Loop over the meshgrid and update outputs
    # for i in range(jnp.shape(xc)[0]):
    #     for j in range(jnp.shape(xc)[1]):  # Use jnp.shape(xc)[1] for columns
    #         # Debugging the values before passing to GP_m.lcb
    #         xc_val = jnp.array([xc[i, j]])
    #         d_val = jnp.array([d[i, j]])

    #         # Call GP_m.lcb and store the result in a variable to inspect it
    #         lcb_result = GP_m.lcb(xc_val, d_val, 0)

    #         # Correctly update 'outputs' with re-assignment
    #         outputs.append(lcb_result)
    # outputs = jnp.array(outputs)
    # outputs = outputs.reshape(jnp.shape(xc)[0],jnp.shape(xc)[1])

    # plt.figure()
    # plt.contourf(xc,d,outputs)
    # plt.colorbar()
    # plt.show()

def test_Maximize_d_with_constraints():
    xcmin = jnp.array([-1.])
    print(xcmin)
    d_max,output = GP_m.Maximise_d_with_constraints(GP_m.ucb,xcmin)
    print(f"Test: Maximize_d_with_constriants; d = argmax ucb s.t constraints")
    print(f"xmin, d_max, output: {xcmin, d_max, output} \n")

    # # Create meshgrid from the boundaries
    # xc = jnp.linspace(bound[:, 0], bound[:, 1], 100)
    # d = jnp.linspace(bound_d[:, 0], bound_d[:, 1], 100)
    # xc, d = jnp.meshgrid(xc.flatten(), d.flatten())

    # # Initialize the outputs array with correct shape (100, 100)
    # outputs = []

    # # Loop over the meshgrid and update outputs
    # for i in range(jnp.shape(xc)[0]):
    #     for j in range(jnp.shape(xc)[1]):  # Use jnp.shape(xc)[1] for columns
    #         # Debugging the values before passing to GP_m.lcb
    #         xc_val = jnp.array([xc[i, j]])
    #         d_val = jnp.array([d[i, j]])

    #         # Call GP_m.lcb and store the result in a variable to inspect it
    #         lcb_result = GP_m.ucb(xc_val, d_val, 0)

    #         # Correctly update 'outputs' with re-assignment
    #         outputs.append(lcb_result)
    # outputs = jnp.array(outputs)
    # outputs = outputs.reshape(jnp.shape(xc)[0],jnp.shape(xc)[1])

    # plt.figure()
    # plt.contourf(xc,d,outputs)
    # plt.colorbar()
    # plt.show()

def test_StageOpt():
    # Class Initialization
    plant_system = [W_shape_Problem.W_shape]
    bound = jnp.array([[-1,2]])
    bound_d = jnp.array([[2,4]])
    b = 3.
    GP_m = StableOpt.BO(plant_system,bound,bound_d,b)

    # Data Storage
    data = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

    # GP Initialization:
    n_sample = 3
    x_samples, plant_output = GP_m.Data_sampling_with_perbutation(n_sample)
    GP_m.GP_initialization(x_samples,plant_output,'RBF',multi_hyper=5,var_out=True)
    data['sampled_x']=x_samples
    data['sampled_output']=plant_output

    # StageOpt
    n_iter = 10

    for i in range(n_iter):
        # Find x and d for sample
        xcmin, fmin = GP_m.Minimize_Maximise(GP_m.lcb)
        dmax, fdmax = GP_m.Maximise_d_with_constraints(GP_m.ucb,xcmin)
        plant_output = GP_m.calculate_plant_outputs(xcmin,dmax)[0]

        # Add sample into GP
        x = jnp.concatenate((xcmin,dmax))
        GP_m.add_sample(x,plant_output)

        # Store data
        data['observed_x'].append(x)
        data['observed_output'].append(plant_output)

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

# xc = jnp.array([-1.])
# d = jnp.array([4])
# print(GP_m.lcb(xc,d,0))

if __name__ == "__main__":
    # test_GP_inference()
    # test_lcb()
    # test_Maximize_d()
    # test_Minimize_Maximise()
    # test_Maximize_d_with_constraints()
    test_StageOpt()
    # test_draw_robust()
    pass