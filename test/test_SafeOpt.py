import jax
import jax.numpy as jnp
import numpy as np
import random
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import time
from models import SafeOpt
from problems import Benoit_Problem, WilliamOttoReactor_Problem
import warnings
from utils import utils_SafeOpt
jax.config.update("jax_enable_x64", True)

warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# Class Initialization
jax.config.update("jax_enable_x64", True)
plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system_tight]
bound = jnp.array([[-.6,1.5],[-1.,1.]])
b = 3.
GP_m = SafeOpt.BO(plant_system,bound,b)

# GP Initialization: 
n_sample = 4
x_i = jnp.array([1.4,-.8])
r = 0.3
noise = 0.
X,Y = GP_m.Data_sampling(n_sample,x_i,r,noise)
GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)

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

def test_minimize_obj_ucb():
    x_min, ucb_min = GP_m.minimize_obj_ucb()

    print(f"Test: minimize objective ucb")
    print(f"x_min, ucb_min: {x_min, ucb_min}")
    print(f"")

def test_Minimizer():   
    # x_new = jnp.array([1., -0.74191489])
    # y_new = GP_m.calculate_plant_outputs(x_new)
    # GP_m.add_sample(x_new,y_new)
    minimizer,std_minimizer = GP_m.Minimizer()
    print(f"Test: Minimizer")
    print(f"minimizer, std_minimizer: {minimizer,std_minimizer}")
    print(f"Check if the input is in safe set: {GP_m.GP_inference(minimizer,GP_m.inference_datasets)}")
    print(f"")

def test_mean_grad_jit():
    x = jnp.array([1.45698204, -0.76514894])
    delta = 0.0001
    xdelta_0 = x + jnp.array([delta,0.])
    xdelta_1 = x + jnp.array([0.,delta])

    mean = GP_m.mean(x,1)
    mean_new_0 = GP_m.mean(xdelta_0,1)
    mean_new_1 = GP_m.mean(xdelta_1,1)

    mean_grad = GP_m.mean_grad_jit(x,1)
    predicted_changed = mean_grad*delta
    predicted_mean_new_0 = mean + predicted_changed[0]
    predicted_mean_new_1 = mean + predicted_changed[1]
    
    print(f"Test: mean grad for 2 dimensions")
    print(f"mean_new_0: {mean_new_0}")
    print(f"mean_new_1: {mean_new_1}")
    print(f"predicted_mean_new_0: {predicted_mean_new_0}")
    print(f"predicted_mean_new_1: {predicted_mean_new_1} \n")

def test_maximize_infnorm_mean_grad():
    print(f"Test: lcb: check if maximum of max-norm mean gradient")
    max_infnorm_mean_constraints = GP_m.maximize_infnorm_mean_grad(1)
    print(f"max_infnorm_mean_constraints: {max_infnorm_mean_constraints} \n")

def test_Expander():
    start = time.time()
    expander, std_expander,satisfied, count = GP_m.Expander()
    end = time.time()
    print(f"Test: Expander")
    print(f"expander, std_expander,satisfied, count: {expander, std_expander,satisfied, count}")
    print(f"time: {end-start} \n")

    plant_output = GP_m.calculate_plant_outputs(expander)
    GP_m.add_sample(expander,plant_output)
    print(f"updated lcb?: {GP_m.lcb(expander,1)}")
    
def test_SafeOpt_Benoit():
    
    # Preparation for plot
    filenames = []
    data = {'i':[],'obj':[],'con':[],'x_0':[],'x_1':[]}

    # SafeOpt
    n_iteration = 10
    print("Im here")
    for i in range(n_iteration):
        # Create sobol_seq sample for Expander
        start = time.time()
        minimizer,std_minimizer = GP_m.Minimizer()
        print(f"minimizer:{minimizer},std: {std_minimizer}")
        expander,std_expander = GP_m.Expander()
        print(f"expander: {expander},std: {std_expander}")
        end = time.time()

        if std_minimizer > std_expander:
            x_new = minimizer
            print(f"I am minimizing!: {x_new}")
        else:
            x_new = expander
            print(f"I am Expanding!: {x_new}")
        print(f"time taken: {end-start}")
        
        plant_output = GP_m.calculate_plant_outputs(x_new,noise)

        # Create frame
        data['i'].append(i)
        data['obj'].append(plant_output[0])
        data['con'].append(plant_output[1])
        data['x_0'].append(x_new[0])
        data['x_1'].append(x_new[1])
        t = i*0.1
        filename = f'frame_{i:02d}.png'
        X_0, X_1, mask_safe, obj = create_data_for_plot()
        utils_SafeOpt.create_frame(utils_SafeOpt.plot_safe_region_Benoit(X,X_0,X_1, mask_safe,obj,bound,data),filename)
        filenames.append(filename)

        GP_m.add_sample(x_new,plant_output)

        # Finish the iteration early 
        if std_expander < 0.01 and std_minimizer < 0.01:
            break
    
    # Create GIF
    frame_duration = 700
    GIFname = 'Benoit_SafeOpt_Outputs.gif'
    utils_SafeOpt.create_GIF(frame_duration,filenames,GIFname)
    # Create plot for outputs
    utils_SafeOpt.plant_outputs_drawing(data['i'],data['obj'],data['con'],'Benoit_SafeOpt_Outputs.png')

def test_multiple_Benoit():
    # Class Initialization
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system_tight]
    bound = jnp.array([[-.6,1.5],[-1.,1.]])
    b = 2.
    GP_m = SafeOpt.BO(plant_system,bound,b)
    n_start = 5
    data = {}

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
        noise = 0.005
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

        # SafeOpt
        n_iteration = 20

        for j in range(n_iteration):
            # Create sobol_seq sample for Expander
            minimizer,std_minimizer = GP_m.Minimizer()
            expander,std_expander = GP_m.Expander()
            lipschitz_continuous = False

            for j in range(1,GP_m.n_fun):
                lipschitz_constraint = GP_m.ucb(expander,j)
                if lipschitz_constraint >= 0.:
                    lipschitz_continuous = True
                    break

            if std_minimizer > std_expander or lipschitz_continuous == False:
                x_new = minimizer
            else:
                x_new = expander

            plant_output = GP_m.calculate_plant_outputs(x_new,noise)
            GP_m.add_sample(x_new,plant_output)

            # Store Data
            data[f'{i}']['observed_x'].append(x_new)
            data[f'{i}']['observed_output'].append(plant_output)

            # Finish the iteration early 
            if std_expander < 0.01 and std_minimizer < 0.01:
                break
    
    jnp.savez('data/data_multi_SafeOpt_Benoit.npz',**data)

def test_multiple_WilliamOttoReactor():
    # Class Initialization
    Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()
    plant_system = [Reactor.get_objective,
                    Reactor.get_constraint1,
                    Reactor.get_constraint2]
    bound = jnp.array([[4.,7.],[70.,100.]])
    b = 2.
    GP_m = SafeOpt.BO(plant_system,bound,b)
    n_start = 10
    data = {}
    noise = 0.001

    for i in range(n_start):
        print(f"iteration: {i}")

        # Data Storage
        data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

        # GP Initialization: 
        x_i = jnp.array([6.8,80.])
        r = 1.
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

        # SafeOpt
        n_iteration = 20

        for j in range(n_iteration):
            # Create sobol_seq sample for Expander
            minimizer,std_minimizer = GP_m.Minimizer()
            print(f"minimizer,std_minimizer: {minimizer,std_minimizer}")
            expander,std_expander = GP_m.Expander()
            print(f"expander,std_expander: {expander,std_expander}")
            print(type(std_expander))

            if std_minimizer > std_expander or std_expander == np.inf or std_expander == np.nan or std_expander == np.inf or std_expander == np.nan:
                x_new = minimizer
            else:
                x_new = expander
            print(f"x_new: {x_new}")
            
            Reactor.noise_generator()
            plant_output = GP_m.calculate_plant_outputs(x_new,noise)
            GP_m.add_sample(x_new,plant_output)

            # Store Data
            data[f'{i}']['observed_x'].append(x_new)
            data[f'{i}']['observed_output'].append(plant_output)
    
    jnp.savez('data/data_multi_SafeOpt_WilliamOttoReactor.npz',**data)

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
    data = {'i':[],'obj':[],'con':[],'x_0':[],'x_1':[]}
    points = jnp.array([[1.,-0.7],[0.44,-1.]])

    for i in range(2):
        x_new = points[i]
        plant_output = GP_m.calculate_plant_outputs(x_new)

        # Create frame
        data['i'].append(i)
        data['obj'].append(plant_output[0])
        data['con'].append(plant_output[1])
        data['x_0'].append(x_new[0])
        data['x_1'].append(x_new[1])
        t = i*0.1
        filename = f'frame_{i:02d}.png'
        X_0, X_1, mask_safe, obj = create_data_for_plot()
        utils_SafeOpt.create_frame(utils_SafeOpt.plot_safe_region_Benoit(X,X_0,X_1, mask_safe,obj,bound,data),filename)
        filenames.append(filename)

        GP_m.add_sample(x_new,plant_output)
    
    # Create GIF
    frame_duration = 700
    GIFname = 'Benoit_SafeOpt_Outputs.gif'
    utils_SafeOpt.create_GIF(frame_duration,filenames,GIFname)
    # Create plot for outputs
    utils_SafeOpt.plant_outputs_drawing(data['i'],data['obj'],data['con'],'Benoit_SafeOpt_Outputs.png')

    
if __name__ == "__main__":
#     test_GP_inference()
#     test_ucb()
#     test_lcb()
#     test_minimize_obj_ucb()
    # test_Minimizer()
    # test_mean_grad_jit()
    # test_maxmimize_maxnorm_mean_grad()
    # test_unsafe_sobol_seq_sampling()
    # test_Expander() 
    # test_SafeOpt_Benoit() 
    # test_multiple_Benoit()
    test_multiple_WilliamOttoReactor()
    # test_GIF()
    pass

