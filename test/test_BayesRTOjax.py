import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
from jax import grad
import BayesRTOjax
import Benoit_Problem


# --- Preparation --- #
plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system]
GP_m = BayesRTOjax.BRTO(plant_system)

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

# --- Initialization --- #
x_0 = jnp.array([1.4,-0.8])
print(f"initial x: {x_0}")
n_sample = 4
r = 0.5

#######################################
#### Test Case 1: Gaussian Process ####
#######################################

def test_GP_initialization():
    
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

# --- Initialization --- #
x_1 = jnp.array([1.4,-0.8])

def test_GP_inference():
    print("\n#___Check similarity in plant and model output on sampled data___#")
    print(f"test input: {x_1}")
    print(f"plant obj: {plant_system[0](x_1)}")
    print(f"model obj: {func_mean(x_1,index=0)}")
    print(f"plant con: {plant_system[1](x_1)}")
    print(f"model con: {func_mean(x_1,index=1)}")

    print("\n#___Check variance at sampled input___#")
    print(f"variance: {variances(x_1)}")

# --- Initialization --- #
x_2 = jnp.array([1.4,-0.8])
delta = 0.0001

def test_GP_inference_grad():
    print("\n# --- GP inference grad --- #")
    for i in range(len(plant_system)):
        check_jaxgrad(x_2,delta,func_mean,index=i)

##################################################
#### Test Case 2: Optimization of Acquisition ####
##################################################

# --- Initialization --- #
r_i = 0.5

def test_optimize_acquisition():
    print('\n# --- optimize acquisiton')
    d_new, obj = GP_m.optimize_acquisition(r_i,x_0,multi_start=1)
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
                    Benoit_Problem.con1_system]
    GP_m = BayesRTOjax.BRTO(plant_system)
    x_i = jnp.array([1.4,-0.8])
    print(f"initial x: {x_0}")
    n_sample = 4
    n_iter = 5
    r = 0.5
    b = 0.

    # GP initialization
    X,Y = GP_m.Data_sampling(n_sample,x_i,r)
    GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=2, var_out=True)

    for i in range(n_iter):
        # Bayesian Optimization
        d_new, obj = GP_m.optimize_acquisition(r_i,x_i,multi_start=5,b=b)

        # Collect Data
        x_new = x_0 + d_new  
        output_new = []
        for plant in plant_system:
            output_new.append(plant(x_new))  
        output_new = jnp.array(output_new)

        GP_m.add_sample(x_new,output_new)

        # Preparation for next iter:
        x_i = x_new

        # Print
        print(f"iter: {i}")
        print(f"x_new: {x_new}")
        print(f"mean: {obj}")
        print(f"output: {output_new}")
        



