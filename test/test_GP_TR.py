import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import random
from models import GP_TR
from problems import Benoit_Problem, WilliamOttoReactor_Problem
from problems import Rosenbrock_Problem
from utils import utils_SafeOpt

import warnings
jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system_tight]
bound = jnp.array([[-.6,1.5],[-1.,1.]])
b = 3.
TR_parameters = {
    'radius': 0.5,
    'radius_max': 1,
    'radius_red': 0.8,
    'radius_inc': 1.1,
    'rho_lb': 0.2,
    'rho_ub': 0.8
}
GP_m = GP_TR.BO(plant_system,bound,b,TR_parameters)

# GP Initialization: 
n_sample = 4
x_old = jnp.array([1.4,-.8])
r_old = 0.3
noise = 0.
X,Y = GP_m.Data_sampling(n_sample,x_old,r_old,noise)
GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)


print(f"\n")
print(f"Data Sample Input:")
print(f"{X}")
print(f"Data Sample Output:")
print(f"{Y}")
print()

# Tests
def test_minimize_obj_lcb():
    min_x, min_f = GP_m.minimize_obj_lcb(r_old,x_old)
    print(min_x,min_f)

def test_update_TR():
    plant_oldoutput = GP_m.calculate_plant_outputs(x_old)
    min_x, min_f = GP_m.minimize_obj_lcb(r_old,x_old)
    plant_newoutput = GP_m.calculate_plant_outputs(min_x)
    x_new, radius_new = GP_m.update_TR(x_old,min_x,r_old,plant_oldoutput,plant_newoutput)
    print(x_new,radius_new)

def test_GP_TR_Benoit():

    filenames = []
    data = {'i':[],'obj':[],'con':[],'x_0':[],'x_1':[]}

    # Initial Phase
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system_tight]
    bound = jnp.array([[-.6,1.5],[-1.,1.]])
    b = 2.
    TR_parameters = {
        'radius_max': 1,
        'radius_red': 0.8,
        'radius_inc': 1.1,
        'rho_lb': 0.2,
        'rho_ub': 0.8
    }
    GP_m = GP_TR.BO(plant_system,bound,b,TR_parameters)

    # GP Initialization: 
    n_sample = 4
    x_old = jnp.array([1.4,-.8])
    r_old = 0.3
    X,Y = GP_m.Data_sampling(n_sample,x_old,r_old)
    GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)

    plant_oldoutput = GP_m.calculate_plant_outputs(x_old,noise)
    n_iteration = 10

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

    # Iteration
    for i in range(n_iteration):
        x_new, obj = GP_m.minimize_obj_lcb(r_old,x_old)
        for k in range(1,GP_m.n_fun):
            print("constraint:",GP_m.lcb(x_new,k))
        plant_newoutput = GP_m.calculate_plant_outputs(x_new,noise)
        x_update, r_new = GP_m.update_TR(x_old,x_new,r_old,plant_oldoutput,plant_newoutput)
        for k in range(1,GP_m.n_fun):
                    print("constraint:",GP_m.lcb(x_update,k))
        # Preparation for next iter:
        x_old = x_update
        r_old = r_new
        if jnp.all(x_update == x_new):
            plant_oldoutput = plant_newoutput
        print("x_new,plant_newoutput,x_update,r_new")
        print(x_new,plant_newoutput,x_update,r_new)

        data['i'].append(i)
        data['obj'].append(plant_newoutput[0])
        data['con'].append(plant_newoutput[1])
        data['x_0'].append(x_new[0])
        data['x_1'].append(x_new[1])
        t = i*0.1
        filename = f'frame_{i:02d}.png'
        X_0, X_1, mask_safe, obj = create_data_for_plot()
        utils_SafeOpt.create_frame(utils_SafeOpt.plot_safe_region_Benoit(X,X_0,X_1, mask_safe,obj,bound,data),filename)
        filenames.append(filename)

        GP_m.add_sample(x_new,plant_newoutput)

        if abs(plant_system[0](x_new) - 0.145249) <= 0.005:
            break
    
    # Create GIF
    frame_duration = 700
    GIFname = 'Benoit_GP_TR_Outputs.gif'
    utils_SafeOpt.create_GIF(frame_duration,filenames,GIFname)
    # Create plot for outputs
    utils_SafeOpt.plant_outputs_drawing(data['i'],data['obj'],data['con'],'Benoit_GP_TR_Outputs.png')

def test_multiple_Benoit():
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system_tight]
    bound = jnp.array([[-.6,1.5],[-1.,1.]])
    b = 2.
    TR_parameters = {
        'radius_max': 1,
        'radius_red': 0.8,
        'radius_inc': 1.1,
        'rho_lb': 0.2,
        'rho_ub': 0.8
    }
    GP_m = GP_TR.BO(plant_system,bound,b,TR_parameters)
    n_start = 10
    data = {}
    noise = 0.

    for i in range(n_start):
        print(f"iteration: {i}")
        # Data Storage
        data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}
        random_number = random.randint(1, 100)
        GP_m.key = jax.random.PRNGKey(random_number)

        # GP Initialization:
        n_sample = 4
        x_old = jnp.array([1.4,-0.8])
        r_old = 0.3
        X,Y = GP_m.Data_sampling(n_sample,x_old,r_old,noise)
        GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)
        plant_oldoutput = GP_m.calculate_plant_outputs(x_old,noise)
        data[f'{i}']['sampled_x'] = X
        data[f'{i}']['sampled_output'] = Y

        print(f"\n")
        print(f"Data Sample Input:")
        print(f"{X}")
        print(f"Data Sample Output:")
        print(f"{Y}")
        print(f"")

        # GP_TR
        n_iteration = 10

        for j in range(n_iteration):
            x_new, obj = GP_m.minimize_obj_lcb(r_old,x_old)
            plant_newoutput = GP_m.calculate_plant_outputs(x_new,noise)
            x_update, r_new = GP_m.update_TR(x_old,x_new,r_old,plant_oldoutput,plant_newoutput)

            # Preparation for next iter:
            x_old = x_update
            r_old = r_new
            if jnp.all(x_update == x_new):
                plant_oldoutput = plant_newoutput

            GP_m.add_sample(x_new,plant_newoutput)

            # Store Data
            data[f'{i}']['observed_x'].append(x_new)
            data[f'{i}']['observed_output'].append(plant_newoutput)

            if abs(plant_system[0](x_new) - 0.145249) <= 0.005:
                break
        
    jnp.savez('data/data_multi_GP_TR_Benoit.npz',**data)

def test_multiple_WilliamOttoReactor():
    # Class Initialization
    Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()
    plant_system = [Reactor.get_objective,
                    Reactor.get_constraint1,
                    Reactor.get_constraint2]
    bound = jnp.array([[4.,7.],[70.,100.]])
    b = 2.
    TR_parameters = {
        'radius_max': 1,
        'radius_red': 0.8,
        'radius_inc': 1.1,
        'rho_lb': 0.2,
        'rho_ub': 0.8
    }
    GP_m = GP_TR.BO(plant_system,bound,b,TR_parameters)
    n_start = 1
    data = {}
    noise = 0.001

    for i in range(n_start):
        print(f"iteration: {i}")
        # Data Storage
        data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

        # GP Initialization: 
        x_old = jnp.array([6.8,80.])
        r_old = 0.3
        n_sample = 5
        X_sample = jnp.empty((0,len(x_old)))
        Y_sample = jnp.empty((0,len(plant_system)))
        for count in range(n_sample):
            X,Y = GP_m.Data_sampling(1,x_old,r_old,noise)
            Reactor.noise_generator
            X_sample=jnp.append(X_sample,X,axis=0)
            Y_sample=jnp.append(Y_sample,Y,axis=0)
        
        GP_m.GP_initialization(X_sample, Y_sample, 'RBF', multi_hyper=5, var_out=True)
        plant_oldoutput = GP_m.calculate_plant_outputs(x_old,noise)
        data[f'{i}']['sampled_x'] = X_sample
        data[f'{i}']['sampled_output'] = Y_sample

        print(f"\n")
        print(f"Data Sample Input:")
        print(f"{X_sample}")
        print(f"Data Sample Output:")
        print(f"{Y_sample}")
        print(f"")

        # GP_TR
        n_iteration = 20

        for j in range(n_iteration):
            x_new, obj = GP_m.minimize_obj_lcb(r_old,x_old)
            Reactor.noise_generator()
            plant_newoutput = GP_m.calculate_plant_outputs(x_new,noise)
            x_update, r_new = GP_m.update_TR(x_old,x_new,r_old,plant_oldoutput,plant_newoutput)
            
            # Preparation for next iter:
            x_old = x_update
            r_old = r_new
            if jnp.all(x_update == x_new):
                plant_oldoutput = plant_newoutput
            
            GP_m.add_sample(x_new,plant_newoutput)

            # Store Data
            data[f'{i}']['observed_x'].append(x_new)
            data[f'{i}']['observed_output'].append(plant_newoutput)
            
            # print(x_new,plant_newoutput,r_new)
    jnp.savez('data/data_multi_GP_TR_WilliamOttoReactor.npz',**data)


if __name__ == "__main__":
    # test_minimize_obj_lcb()
    # test_update_TR()
    # test_GP_TR_Benoit()
    # test_multiple_Benoit()
    test_multiple_WilliamOttoReactor()

