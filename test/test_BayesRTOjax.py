import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
jax.config.update("jax_enable_x64", True)

from models import BayesRTOjax
from problems import Benoit_Problem
from problems import Rosenbrock_Problem

def test_RTOminimize_Benoit():

    # Initialization
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system_tight]
    
    GP_m = BayesRTOjax.BayesianOpt(plant_system)

    # Data Sampling
    n_sample = 4
    x_i = jnp.array([1.1,-0.8])
    r = 0.5

    X,Y = GP_m.Data_sampling(n_sample,x_i,r)

    # GP initialization
    GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=10, var_out=True)

    # Real-Time Optimization 
    n_iter = 10
    multi_start= 5
    b = 3.
    TR_parameters = {
        'radius': 0.5,
        'radius_max': 1,
        'radius_red': 0.8,
        'radius_inc': 1.1,
        'rho_lb': 0.2,
        'rho_ub': 0.8
    }

    data = GP_m.RTOminimize(n_iter=n_iter,x_initial=x_i,TR_parameters=TR_parameters,
                            multi_start=multi_start,b=b)
    
    # Data Processing
    processed_data = {
        'i': data['i'],
        'x_initial_0': data['x_initial'][:,0],
        'x_initial_1': data['x_initial'][:,1],
        'x_new_0': data['x_new'][:, 0],
        'x_new_1': data['x_new'][:, 1],
        'plant_output': data['plant_output'][:, 0],
        'plant_constraint': data['plant_output'][:, 1],
        'GP_constraint': data['GP_cons'][:,0],
        'GP_constraint_safe': data['GP_cons_safe'][:,0],
        'TR_radius': data['TR_radius']
    }

    # print Data
    df = pd.DataFrame(processed_data)
    print("\n", df)

    # Plot Real-Time Optimization Step
    filenames = []
    for i in range(processed_data['x_new_0'].shape[0]):
        # Create Frame
        t = i * 0.1
        filename = f'frame_{i:02d}.png'
        Benoit_Problem.create_frame(Benoit_Problem.BRTO_Benoit_drawing(processed_data,i),filename)
        filenames.append(filename)

    ## Create GIF
    frame_duration = 1000
    GIFname = 'Benoit_Problem.gif'
    Benoit_Problem.create_GIF(frame_duration,filenames,GIFname)

    # Plot plant output and its constraint
    figname = 'Benoit_Problem_Outputs.png'
    Benoit_Problem.plant_outputs_drawing(processed_data['i'],
                                         processed_data['plant_output'],
                                         processed_data['plant_constraint'],
                                         figname)

    
def test_RTOminimize_Rosenbrock():
    
    # Initialization
    plant_system = [Rosenbrock_Problem.Rosenbrock_f,
                    Rosenbrock_Problem.con1_system]
    
    GP_m = BayesRTOjax.BayesianOpt(plant_system)

    # Data Sampling:
    X = jnp.array([[-1.5,-0.6],
                   [-1.8,-0.65],
                   [-1.9,-0.5],
                   [-1.7,-0.7],
                   [-1.95,-0.8],
                   [-1.5,-1]])
    n_sample = X.shape[0]
    n_fun = 2
    Y = jnp.zeros((n_sample,n_fun))
    for i in range(n_fun):
        Y = Y.at[:,i].set(vmap(plant_system[i])(X))

    # GP initialization
    GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=10, var_out=True)

    # Real-Time Optimization 
    x_i = jnp.array([-1.8,-.8])
    n_iter = 20
    multi_start = 5
    b = 3
    TR_parameters = {
        'radius': 0.25,
        'radius_max': 0.7,
        'radius_red': 0.8,
        'radius_inc': 1.1,
        'rho_lb': 0.2,
        'rho_ub': 0.8
    }
    data = GP_m.RTOminimize(n_iter=n_iter,x_initial=x_i,TR_parameters=TR_parameters,
                            multi_start=multi_start,b=b)
    
    # Data Processing 
    processed_data = {
        'i': data['i'],
        'x_initial_0': data['x_initial'][:,0],
        'x_initial_1': data['x_initial'][:,1],
        'x_new_0': data['x_new'][:, 0],
        'x_new_1': data['x_new'][:, 1],
        'plant_output': data['plant_output'][:, 0],
        'plant_constraint': data['plant_output'][:, 1],
        'TR_radius': data['TR_radius']
    }

    # print Data
    df = pd.DataFrame(processed_data)
    print("\n", df)

    # Plot Real-Time Optimization Step
    filenames = []
    for i in range(processed_data['x_new_0'].shape[0]):
        # Create Frame
        t = i * 0.1
        filename = f'frame_{i:02d}.png'
        Rosenbrock_Problem.create_frame(Rosenbrock_Problem.BRTO_Rosenbrock_drawing(processed_data,i),filename)
        filenames.append(filename)

    ## Create GIF
    frame_duration = 100
    GIFname = 'Rosenbrock_Problem.gif'
    Rosenbrock_Problem.create_GIF(frame_duration,filenames,GIFname)

    # Plot plant output and its constraint
    figname = 'Rosenbrock_Problem_Outputs.png'
    Rosenbrock_Problem.plant_outputs_drawing(processed_data['i'],
                                             processed_data['plant_output'],
                                             processed_data['plant_constraint'],
                                             figname)

if __name__ == "__main__":
    test_RTOminimize_Benoit()

        



