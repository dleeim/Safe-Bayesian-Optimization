import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
from models.GP_Classic import GP
from problems import Benoit_Problem
from problems import Rosenbrock_Problem 

# --- Preparation --- #
plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system_tight]

GP_m = GP(plant_system)

def func_mean(x,index):
    return GP_m.GP_inference(x,GP_m.inference_datasets)[0][index]

def func_var(x,index):
    return GP_m.GP_inference(x,GP_m.inference_datasets)[1][index]

def variances(x):
    return GP_m.GP_inference(x,GP_m.inference_datasets)[1]

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

if __name__ == "__main__":
    test_GP_initialization()
    test_NLL()
    test_GP_inference()
    test_GP_inference_grad()