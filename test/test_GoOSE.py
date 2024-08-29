import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import time
from models import GoOSE
from problems import Benoit_Problem
import warnings
from utils import utils_SafeOpt
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# Class Initialization
jax.config.update("jax_enable_x64", True)
plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system_tight]
bound = jnp.array([[-.6,1.5],[-1.,1.]])
b = 3.
GP_m = GoOSE.BO(plant_system,bound,b)

# GP Initialization: 
n_sample = 4
x_i = jnp.array([1.4,-.8])
r = 0.3
X,Y = GP_m.Data_sampling(n_sample,x_i,r)
GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)
# Create sobol_seq sample for Expander
n_sample = 1000
safe_sobol_sample = GP_m.safe_sobol_seq_sampling(GP_m.nx_dim,n_sample,GP_m.bound)

print(f"\n")
print(f"Data Sample Input:")
print(f"{X}")
print(f"Data Sample Output:")
print(f"{Y}")
print(f"")
print

# Tests
def test_GP_inference():
    i = 0
    x = jnp.array([1.45698204, -0.76514894])
    plant = GP_m.calculate_plant_outputs(x)
    print(f"Test: GP Inference; check if Gp inference well with low var")
    print(f"x input: {x}")
    print(f"GP Inference: {GP_m.GP_inference(x,GP_m.inference_datasets)}")
    print(f"Actual plant: {plant}")
    print(f"")


def test_ucb():
    i = 0
    x = jnp.array([1.45698204, -0.76514894])
    ucb = GP_m.ucb(x,i)
    obj_fun = GP_m.calculate_plant_outputs(x)[0]

    print(f"Test: ucb; check if objective ucb value at b=3 and actual objective value ares similar")
    print(f"x input: {x}")
    print(f"ucb: {ucb}")
    print(f"Actual obj fun: {obj_fun}")
    print(f"")

def test_lcb():
    i = 1
    x = jnp.array([1.45698204, -0.76514894])
    lcb = GP_m.lcb(x,i)
    constraint = GP_m.calculate_plant_outputs(x)[1]

    print(f"Test: lcb; check if constraint lcb value at b=3 is bigger than 0")
    print(f"x input: {x}")
    print(f"lcb: {lcb}")
    print(f"Actual constraint: {constraint}")
    print(f"")

def test_maxmimize_maxnorm_mean_grad():
    maximum_maxnorm_mean_constraints = GP_m.maxmimize_maxnorm_mean_grad()
    print(maximum_maxnorm_mean_constraints)

def test_optimistic_safeset_constraint():
    x = jnp.array([1.09323804, -0.59091112])
    maximum_maxnorm_mean_constraints = GP_m.maxmimize_maxnorm_mean_grad()
    start = time.time()
    indicator = GP_m.optimistic_safeset_constraint(x,safe_sobol_sample,maximum_maxnorm_mean_constraints)
    end = time.time()
    print(f"Test: optimisitic safeset constraint; outputs indicator: 0 - not satisfied condition, 1 - satisfied condition")
    print(f"x input: {x}")
    print(f"indicator: {indicator}")
    print(f"time_taken: {end-start}")

def test_minimize_obj_lcb():
    x_min,min_lcb = GP_m.minimize_obj_lcb()
    print(f"Test: minimize objective function lcb")
    print(f"x_min: {x_min}")
    print(f"min_lcb: {min_lcb}")

def test_Target():
    maximum_maxnorm_mean_constraints = GP_m.maxmimize_maxnorm_mean_grad()
    start = time.time()
    target,target_min = GP_m.Target(safe_sobol_sample,maximum_maxnorm_mean_constraints)
    end = time.time()
    print(f"Test: find target in optimistic safe set")
    print(f"x_min: {target}")
    print(f"min_lcb: {target_min}")
    print(f"time spent: {end-start}")

def test_explore_safeset():
    target = jnp.array([0.98963043, -0.42596487])
    x_observe = GP_m.explore_safeset(target)
    print(f"Test: find point to observe in safe set given target")
    print(f"target: {target}")
    print(f"x_observe: {x_observe}")

def test_GoOSE():
    
    # Preparation for plot
    filenames = []
    data = {'i':[],'obj':[],'con':[],'x_0':[],'x_1':[], 'x_target_0':[], 'x_target_1':[]}
    points = jnp.array([[1.,-0.7],[0.44,-1.]])

    # GoOSE
    n_iteration = 10

    for i in range(n_iteration):
        x_safe_min,min_safe_lcb = GP_m.minimize_obj_lcb()
        maximum_maxnorm_mean_constraints = GP_m.maxmimize_maxnorm_mean_grad()
        x_target_min,min_target_lcb = GP_m.Target(safe_sobol_sample,maximum_maxnorm_mean_constraints)

        if min_safe_lcb <= min_target_lcb:
            x_new = x_safe_min
            x_target_min = jnp.array([None]*GP_m.nx_dim)
        else:
            x_safe_observe = GP_m.explore_safeset(x_target_min)
            x_new = x_safe_observe

        plant_output = GP_m.calculate_plant_outputs(x_new)

        # Create frame
        data['i'].append(i)
        data['obj'].append(plant_output[0])
        data['con'].append(plant_output[1])
        data['x_0'].append(x_new[0])
        data['x_1'].append(x_new[1])
        data['x_target_0'].append(x_target_min[0])
        data['x_target_1'].append(x_target_min[1])
        t = i*0.1
        filename = f'frame_{i:02d}.png'
        X_0, X_1, mask_safe, obj = create_data_for_plot()
        utils_SafeOpt.create_frame(utils_SafeOpt.plot_safe_region_Benoit(X,X_0,X_1, mask_safe,obj,bound,data),filename)
        filenames.append(filename)

        GP_m.add_sample(x_new,plant_output)

    # Create GIF
    frame_duration = 700
    GIFname = 'Benoit_GoOSE_Outputs.gif'
    utils_SafeOpt.create_GIF(frame_duration,filenames,GIFname)
    # Create plot for outputs
    utils_SafeOpt.plant_outputs_drawing(data['i'],data['obj'],data['con'],'Benoit_GoOSE_Outputs.png')

def create_data_for_plot():
    x_0 = jnp.linspace(-0.6, 1.5, 400)
    x_1 = jnp.linspace(-1.0, 1.0, 400)
    X_0, X_1 = jnp.meshgrid(x_0, x_1)

    # Flatten the meshgrid arrays
    X_0_flat = X_0.ravel()
    X_1_flat = X_1.ravel()

    # Stack them to create points for lcb function
    points = jnp.column_stack((X_0_flat, X_1_flat))

    # Apply lcb function using vmap
    lcb_vmap = vmap(GP_m.lcb, in_axes=(0, None))
    mask_safe = lcb_vmap(points, 1).reshape(X_0.shape) > 0.

    # Create points for plant system
    plant_obj_vmap = vmap(plant_system[0])
    plant_con_vmap = vmap(plant_system[1])
    obj = jnp.array(plant_obj_vmap(points)).reshape(X_0.shape)

    return X_0, X_1, mask_safe, obj


if __name__ == "__main__":
    # test_GP_inference()
    # test_ucb()
    # test_lcb()
    # test_maxmimize_maxnorm_mean_grad()
    # test_optimistic_safeset_constraint()
    # test_minimize_obj_lcb()
    # test_Target()
    # test_explore_safeset()


