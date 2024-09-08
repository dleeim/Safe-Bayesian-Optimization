import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit, random
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import pandas as pd
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

def test_robust_filter():
    x = jnp.array([0.7224331])
    max_robust_f = GP_m.robust_filter(GP_m.lcb,x,0)
    print(f"Test: robust filter; check if robust filter is in appropriate value")
    print(f"x input: {x}")
    print(f"max_robust_f: {max_robust_f}")
    print(f"")

def test_Minimize_Robust():
    xmin, fmin = GP_m.Minimize_Robust()
    print(f"Test: Minimize Robust; check if Minimize Robust is in appropriate value")
    print(f"xmin, fmin: {xmin, fmin}")

def test_

def test_draw_robust(): 
    x = jnp.linspace(bound[:,0],bound[:,1],100)
    outputs = []
    for i in x:
        output = GP_m.robust_filter(GP_m.lcb,i,0)
        outputs.append(output)
    plt.figure()
    plt.plot(x,outputs)
    plt.show()

if __name__ == "__main__":
    test_GP_inference()
    test_lcb()
    test_robust_filter()
    test_Minimize_Robust()
    test_draw_robust()
    pass