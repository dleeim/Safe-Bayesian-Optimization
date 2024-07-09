import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
from jax import grad, vmap, jit
import pandas as pd
import BayesRTOjax
import Benoit_Problem
import Rosenbrock_Problem


# --- Preparation --- #
plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system]
GP_m = BayesRTOjax.BayesianOpt(plant_system)

def func_mean(x,index):
    return GP_m.GP_inference_np(x)[0][index]

def func_var(x,index):
    return GP_m.GP_inference_np(x)[1][index]

def variances(x):
    return GP_m.GP_inference_np(x)[1]

def change_jaxgrad(x,delta,func,index):
    func_grad = grad(func,argnums=0)(x,index)
    predicted_change = func_grad * delta
    return predicted_change

def check_jaxgrad(x,delta,func,index):
    xdelta_0 = x + jnp.array([delta,0.])
    xdelta_1 = x + jnp.array([0.,delta])

    func_new_0 = func(xdelta_0,index)
    func_new_1 = func(xdelta_1,index)

    func_grad = grad(func,argnums=0)(x,index)
    predicted_change = func_grad * delta    

    func_newgrad_0 = func(x,index) + predicted_change[0]
    func_newgrad_1 = func(x,index) + predicted_change[1]

    print(f"\n check for accuracy for jax grad in {func}")
    print(f"input test: {x}")
    print(f"jaxgrad: {func_grad}")
    print(f"Actual new function after increase in 1st dim: {func_new_0}")
    print(f"Actual new function after increase in 2st dim: {func_new_1}")
    print(f"Predicted(jaxgrad) new function after increase in 1st dim: {func_newgrad_0}")
    print(f"Predicted(jaxgrad) new function after increase in 2st dim: {func_newgrad_1}")
    print(f"Difference between actual and predicted in 1st dim: {func_new_0 - func_newgrad_0}")
    print(f"Difference between actual and predicted in 1st dim: {func_new_1 - func_newgrad_1}")

#######################################
#### Test Case 1: Gaussian Process ####
#######################################

def test_GP_initialization():
    # --- Initialization --- #
    x_0 = jnp.array([1.4,-0.8])
    n_sample = 4
    r = 0.5
    
    print("\n# --- Data Sampling --- #")
    X,Y = GP_m.Data_sampling(n_sample,x_0,r)
    print(f'X: \n{X}')
    print(f"Y: \n{Y}")

    print("\n# --- GP initialization --- #")
    GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=2, var_out=True)
    print(f"GP mean: {GP_m.Y_mean}")

def test_NLL():
    print("\n# --- negative log likelihood --- #")
    hyper = jnp.array([[ 0.,  0.,  0., -5.],
                    [ 2.,  -2.,   2.,  -6.5],])
    for i in range(hyper.shape[0]):
        NLL = GP_m.negative_loglikelihood(hyper[i],GP_m.X_norm,GP_m.Y_norm[:,i:i+1])
        print(f"NLL for func index {i}: {NLL}")

def test_GP_inference():
    # --- Initialization --- #
    x_1 = jnp.array([1.4,-0.8])

    print("\n#___Check similarity in plant and model output on sampled data___#")
    print(f"test input: {x_1}")
    print(f"plant obj: {plant_system[0](x_1)}")
    print(f"model obj: {func_mean(x_1,index=0)}")
    print(f"plant con: {plant_system[1](x_1)}")
    print(f"model con: {func_mean(x_1,index=1)}")

    print("\n#___Check variance at sampled input___#")
    print(f"variance: {variances(x_1)}")

def test_GP_inference_grad():
    # --- Initialization --- #
    x_2 = jnp.array([1.4,-0.8])
    delta = 0.0001

    print("\n# --- GP inference grad --- #")
    for i in range(len(plant_system)):
        check_jaxgrad(x_2,delta,func_mean,index=i)

##################################################
#### Test Case 2: Optimization of Acquisition ####
##################################################

def test_minimize_acquisition():
    # --- Initialization --- #
    x_0 = jnp.array([1.4,-0.8])
    r_i = 0.5

    print('\n# --- minimize acquisiton')
    d_new, obj = GP_m.minimize_acquisition(r_i,x_0,multi_start=1)
    print(f"optimal new input(model): {x_0+d_new}")
    print(f"corresponding new output(model): {obj}")
    print(f"Euclidean norm of d_new(model): {jnp.linalg.norm(d_new)}")


    print(f"\n#___Check Add Sample___#")
    output_new = []
    for plant in plant_system:
        output_new.append(plant(x_0+d_new))
    print(output_new)
    output_new = jnp.array(output_new)
    print(f"add sample: {x_0+d_new, output_new}")
    GP_m.add_sample(x_0+d_new,output_new)

    print(f"output after add sample(model): {func_mean(x_0+d_new,0)}")
    print(f"constraint after add sample(model): {func_mean(x_0+d_new,1)}")
    print(f"new variances(model): {variances(x_0+d_new)}")
    print(f"plant output: {plant_system[0](x_0+d_new)}")
    print(f"plant constraint: {plant_system[1](x_0+d_new)}")

#############################################
#### Test Case 3: Real Time Optimization ####
#############################################

def test_RealTimeOptimization():

    print("# --- Real Time Optimization --- #")
    # --- Initialization --- #
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system,
                    Benoit_Problem.con1_system_tight]
    GP_m = BayesRTOjax.BayesianOpt(plant_system)
    x_i = jnp.array([1.4,-0.8])
    print(f"initial x: {x_i}")
    n_sample = 4
    n_iter = 5
    r = 0.5
    b = 0.

    # GP initialization
    X,Y = GP_m.Data_sampling(n_sample,x_i,r)
    GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=2, var_out=True)

    for i in range(n_iter):
        # Bayesian Optimization
        d_new, obj = GP_m.minimize_acquisition(r_i,x_i,multi_start=5,b=b)

        # Collect Data
        x_new = x_i + d_new  
        output_new = []
        for plant in plant_system:
            output_new.append(plant(x_new))  
        output_new = jnp.array(output_new)

        # Add sample
        GP_m.add_sample(x_new,output_new)

        # Preparation for next iter:
        x_i = x_new

        # Print
        print(f"iter: {i}")
        print(f"d_new: {d_new}")
        print(f"Euclidean norm of d_new(model): {jnp.linalg.norm(d_new)}")
        print(f"x_new: {x_new}")
        print(f"mean: {obj}")
        print(f"output: {output_new}")
        print(f"check add sample: {GP_m.GP_inference_np(x_i)}")

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
    multi_start=5
    b = 0.
    TR_parameters = {
        'radius': 0.5,
        'radius_max': 2,
        'radius_red': 0.8,
        'radius_inc': 1.2,
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
        Benoit_Problem.create_frame(Benoit_Problem.BRTO_Benoit_drawing(processed_data,i),filename)
        filenames.append(filename)

    ## Create GIF
    frame_duration = 1000
    GIFname = 'BRTO_Benoit.gif'
    Benoit_Problem.create_GIF(frame_duration,filenames,GIFname)

    # Plot plant output and its constraint
    figname = 'Plant_Output_Constraint_png'
    Benoit_Problem.plant_outputs_drawing(processed_data['i'],
                                         processed_data['plant_output'],
                                         processed_data['plant_constraint'],
                                         figname)

    
def test_RTOminimize_Rosenbrock():
    
    # Initialization
    plant_system = [Rosenbrock_Problem.Rosenbrock_f,
                    Rosenbrock_Problem.con2contour_plot]
    
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

    pass
        



