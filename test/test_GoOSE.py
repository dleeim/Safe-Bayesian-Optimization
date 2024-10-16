import jax
import jax.numpy as jnp
import numpy as np
import random
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import time
from models import GoOSE
from problems import Benoit_Problem
import warnings
from utils import utils_GoOSE
from scipy.spatial.distance import cdist
jax.config.update("jax_enable_x64", True)
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
noise = 0.
X,Y = GP_m.Data_sampling(n_sample,x_i,r,noise)
GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)
# Create sobol_seq sample for Expander
n_sample = 1000
safe_sobol_sample = GP_m.safe_sobol_seq_sampling(GP_m.nx_dim,n_sample,GP_m.bound)

print(f"\n")
print(f"Data Sample Input:")
print(f"{X}")
print(f"Data Sample Output:")
print(f"{Y}")
print(f"Y norm")
print(f"{GP_m.Y_norm}")
print()

# Tests
def test_GP_inference():
    i = 0
    x = jnp.array([0.,0.])
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

def test_min_lcb_cons():
    x = jnp.array([1.4, -0.8])
    min_value = GP_m.min_lcb_cons(x)
    print(f"Test minvalue of all lcb cons:")
    print(f"min value: {min_value} \n")

def test_maxmimize_maxnorm_mean_grad():
    print(f"Test: lcb: check if maximum of max-norm mean gradient")
    maximum_maxnorm_mean_constraints = GP_m.maxmimize_maxnorm_mean_grad()
    print(f"maximum_maxnorm_mean_constraints: {maximum_maxnorm_mean_constraints} \n")

def test_optimistic_safeset_constraint():
    x = jnp.array([1.4, -0.59091112])
    maximum_maxnorm_mean_constraints = GP_m.maxmimize_maxnorm_mean_grad()
    start = time.time()
    indicator = GP_m.optimistic_safeset_constraint(x,safe_sobol_sample,maximum_maxnorm_mean_constraints)
    end = time.time()
    print(f"Test: optimisitic safeset constraint; outputs indicator: 0 - not satisfied condition, 10 - satisfied condition")
    print(f"x input: {x}")
    print(f"indicator: {indicator}")
    print(f"time_taken: {end-start}")

def test_minimize_obj_lcb():
    x_min,min_lcb = GP_m.minimize_obj_lcb()
    print(f"Test: minimize objective function lcb")
    print(f"x_min: {x_min}")
    print(f"min_lcb: {min_lcb} \n")

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

def test():
    x_1 = jnp.array([0.98963043, -0.42596487])
    plant_output = GP_m.calculate_plant_outputs(x_1)
    GP_m.add_sample(x_1,plant_output)
    safe_sobol_sample = GP_m.safe_sobol_seq_sampling(GP_m.nx_dim,n_sample,GP_m.bound)
    maximum_maxnorm_mean_constraints = GP_m.maxmimize_maxnorm_mean_grad()
    # print(maximum_maxnorm_mean_constraints)
    # x_target = jnp.array([0.97, -0.42596487])
    # indicator = GP_m.optimistic_safeset_constraint(x_target,safe_sobol_sample,maximum_maxnorm_mean_constraints)
    # print(indicator)
    start = time.time()
    x_target_min,min_target_lcb = GP_m.Target(safe_sobol_sample,maximum_maxnorm_mean_constraints)
    end = time.time()
    print(f"x_target: {x_target_min}")
    print(f"min_target_lcb: {min_target_lcb}")
    print(f"time: {end-start}")

def test_GoOSE():
    
    # Preparation for plot
    filenames = []
    data = {'i':[],'obj':[],'con':[],'x_0':[],'x_1':[], 'x_target_0':[], 'x_target_1':[]}

    # GoOSE
    n_iteration = 20

    for i in range(n_iteration):
        print(f"Iteration: {i}")
        x_safe_min,min_safe_lcb = GP_m.minimize_obj_lcb()
        print(f"x_safe_min, min_safe_lcb: {x_safe_min,min_safe_lcb}")
        safe_sobol_sample = GP_m.safe_sobol_seq_sampling(GP_m.nx_dim,n_sample,GP_m.bound)
        maximum_maxnorm_mean_constraints = GP_m.maxmimize_maxnorm_mean_grad()
        x_target_min,min_target_lcb = GP_m.Target(safe_sobol_sample,maximum_maxnorm_mean_constraints)
        print(f"x_target_min,min_target_lcb: {x_target_min,min_target_lcb}")

        # check if target satisfied optimistic safe set condition:
        value = GP_m.optimistic_safeset_constraint(x_target_min,safe_sobol_sample,maximum_maxnorm_mean_constraints)

        if value > 0. or min_safe_lcb <= min_target_lcb:
            x_new = x_safe_min
            x_target_min = jnp.array([jnp.nan]*GP_m.nx_dim)
        else:
            x_safe_observe = GP_m.explore_safeset(x_target_min)
            x_new = x_safe_observe

        plant_output = GP_m.calculate_plant_outputs(x_new,noise)

        # Create frame
        data['i'].append(i)
        data['obj'].append(plant_output[0])
        data['con'].append(plant_output[1])
        data['x_0'].append(x_new[0])
        data['x_1'].append(x_new[1])
        data['x_target_0']=x_target_min[0]
        data['x_target_1']=x_target_min[1]
        t = i*0.1
        filename = f'frame_{i:02d}.png'
        X_0, X_1, mask_safe, obj = create_data_for_plot()
        utils_GoOSE.create_frame(utils_GoOSE.plot_safe_region_Benoit(X,X_0,X_1, mask_safe,obj,bound,data),filename)
        filenames.append(filename)

        GP_m.add_sample(x_new,plant_output)
        print(plant_output[0])

        if abs(plant_system[0](x_new) - 0.145249) <= 0.005:
            break

    # Create GIF
    frame_duration = 700
    GIFname = 'Benoit_GoOSE_Outputs.gif'
    utils_GoOSE.create_GIF(frame_duration,filenames,GIFname)
    # Create plot for outputs
    utils_GoOSE.plant_outputs_drawing(data['i'],data['obj'],data['con'],'Benoit_GoOSE_Outputs.png')

def test_multiple_Benoit():
    # Class Initialization
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system_tight]
    bound = jnp.array([[-.6,1.5],[-1.,1.]])
    b = 2.
    GP_m = GoOSE.BO(plant_system,bound,b)
    n_start = 5
    data = {}
    noise = 0.005

    for i in range(n_start):
        print(f"iteration: {i}")
        # Data Storage
        data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}
        random_number = random.randint(1, 100)
        GP_m.key = jax.random.PRNGKey(random_number)

        # GP Initialization: 
        n_sample = 4
        x_i = jnp.array([1.4,-.8])
        r = 0.3
        X,Y = GP_m.Data_sampling(n_sample,x_i,r,noise)
        GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)
        data[f'{i}']['sampled_x'] = X
        data[f'{i}']['sampled_output'] = Y

        print(f"\n")
        print(f"Data Sample Input:")
        print(f"{X}")
        print(f"Data Sample Output:")
        print(f"{Y}")
        print(f"")

        # GoOSE
        n_iteration = 10

        for j in range(n_iteration):
            x_safe_min,min_safe_lcb = GP_m.minimize_obj_lcb()
            
            # Create sobol_seq sample for Optimistic Safe Set
            sobol_seq_n_sample = 1000
            safe_sobol_sample = GP_m.safe_sobol_seq_sampling(GP_m.nx_dim,sobol_seq_n_sample,GP_m.bound)
            maximum_maxnorm_mean_constraints = GP_m.maxmimize_maxnorm_mean_grad()
            x_target_min,min_target_lcb = GP_m.Target(safe_sobol_sample,maximum_maxnorm_mean_constraints)

            # check if target satisfied optimistic safe set condition:
            value = GP_m.optimistic_safeset_constraint(x_target_min,safe_sobol_sample,maximum_maxnorm_mean_constraints)

            if value > 0. or min_safe_lcb <= min_target_lcb:
                x_new = x_safe_min
                x_target_min = jnp.array([jnp.nan]*GP_m.nx_dim)
            else:
                x_safe_observe = GP_m.explore_safeset(x_target_min)
                x_new = x_safe_observe

            plant_output = GP_m.calculate_plant_outputs(x_new,noise)
            GP_m.add_sample(x_new,plant_output)

            # Store Data
            data[f'{i}']['observed_x'].append(x_new)
            data[f'{i}']['observed_output'].append(plant_output)

            if abs(plant_system[0](x_new) - 0.145249) <= 0.005:
                break
    
    jnp.savez('data/data_multi_GoOSE_Benoit.npz',**data)


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

def test_GIF():
    # Preparation for plot
    filenames = []
    data = {'i':[],'obj':[],'con':[],'x_0':[],'x_1':[], 'x_target_0':[], 'x_target_1':[]}
    points = jnp.array([[0.98963043, -0.42596487],[0.44,-1.]])
    targets = jnp.array([[1.,0.8],[jnp.nan,jnp.nan]])

    for i in range(2):
        x_new = points[i]
        target = targets[i]
        plant_output = GP_m.calculate_plant_outputs(x_new)

        # Create frame
        data['i'].append(i)
        data['obj'].append(plant_output[0])
        data['con'].append(plant_output[1])
        data['x_0'].append(x_new[0])
        data['x_1'].append(x_new[1])
        data['x_target_0']=target[0]
        data['x_target_1']=target[1]

        t = i*0.1
        filename = f'frame_{i:02d}.png'
        X_0, X_1, mask_safe, obj = create_data_for_plot()
        utils_GoOSE.create_frame(utils_GoOSE.plot_safe_region_Benoit(X,X_0,X_1, mask_safe,obj,bound,data),filename)
        filenames.append(filename)

        GP_m.add_sample(x_new,plant_output)

    # Create GIF
    frame_duration = 700
    GIFname = 'Benoit_GoOSE_Outputs.gif'
    utils_GoOSE.create_GIF(frame_duration,filenames,GIFname)
    # Create plot for outputs
    utils_GoOSE.plant_outputs_drawing(data['i'],data['obj'],data['con'],'Benoit_GoOSE_Outputs.png')
    


if __name__ == "__main__":
    # test_GP_inference()
    # test_ucb()
    # test_lcb()
    # test_min_lcb_cons()
    # test_maxmimize_maxnorm_mean_grad()
    # test_optimistic_safeset_constraint()
    # test_minimize_obj_lcb()
    # test_Target()
    # test_explore_safeset()
    test_GoOSE()
    # test_multiple_Benoit()
    # test_GIF()
    pass


