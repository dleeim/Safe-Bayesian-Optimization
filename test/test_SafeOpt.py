import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import time
from models import SafeOpt
from problems import Benoit_Problem
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# Class Initialization
jax.config.update("jax_enable_x64", True)
plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system_tight]
bound = jnp.array([[-.6,1.5],[-1.,1.]])
b = 3.
GP_m = SafeOpt.SafeOpt(plant_system,bound,b)

# GP Initialization: 
n_sample = 4
x_i = jnp.array([1.4,-.8])
r = 0.3
X,Y = GP_m.Data_sampling(n_sample,x_i,r)
GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)

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

def test_minimize_obj_ucb():
    x_min, ucb_min = GP_m.minimize_obj_ucb()

    print(f"Test: minimize objective ucb")
    print(f"x_min, ucb_min: {x_min, ucb_min}")
    print(f"")

def test_Minimizer():   
    minimizer,std_minimizer = GP_m.Minimizer()
    print(f"Test: Minimizer")
    print(f"minimizer, std_minimizer: {minimizer,std_minimizer}")
    print(f"Check if the input is in safe set: {GP_m.GP_inference(minimizer,GP_m.inference_datasets)}")
    print(f"")
    plant_output = GP_m.calculate_plant_outputs(minimizer)
    GP_m.add_sample(minimizer,plant_output)
    plot_safe_region()


def test_create_point_arb():
    x = jnp.array([1.45698204, -0.76514894])
    fun_arb = GP_m.create_point_arb(x)
    print(f"Test: Create aribtrary point; makes point of given x input and ucb of all plant system")
    print(f"x input: {x}")
    print(f"arbitrary ucb for all functions: {fun_arb} \n")

def test_lcb_arb():
    x = jnp.array([1., -0.74191489])
    y = GP_m.create_point_arb(x)
    GP_m.create_GP_arb(x,y)

    value_arb = GP_m.lcb_arb(x,1)
    value = GP_m.lcb(x,1)
    print(f"Test: lcb for arbitrary GP")
    print(f"x for arbitrary GP: {x}")
    print(f"value of lcb for GP arb: {value_arb}")
    print(f"value of lcb for GP: {value}")

def test_Expander_constraint():
    x = jnp.array([1., -0.74191489])
    start = time.time()
    indicator = GP_m.Expander_constraint(x)
    end = time.time()
    print(f"Test: Expander constraint: Result is indicator; 1 means point can be classified as expander")
    print(f"x input: {x}")
    print(f"indicator: {indicator}")
    print(f"time spent: {end-start} \n")

def test_Expander():
    x_old = jnp.array([1.00036285, -0.71751398])
    start = time.time()
    expander, std_expander = GP_m.Expander()
    end = time.time()
    print(f"Test: Expander")
    print(f"expander, std_expander: {expander, std_expander}")
    print(f"time: {end-start} \n")

    plant_output = GP_m.calculate_plant_outputs(expander)
    GP_m.add_sample(expander,plant_output)
    print(f"updated ")
    print(f"updated lcb?: {GP_m.lcb(expander,1)}")

    plot_safe_region()
    
def test_SafeOpt():
    print(f"Test: SafeOpt")
    n_iteration = 1
    for i in range(n_iteration):
        print(f"iteration: {i}")
        minimizer,std_minimizer = GP_m.Minimizer()

        start = time.time()
        expander,std_expander = GP_m.Expander()
        end = time.time()

        if std_minimizer > std_expander:
            x_new = minimizer
        else:
            x_new = expander
        
        plant_output = GP_m.calculate_plant_outputs(x_new)
        GP_m.add_sample(x_new,plant_output)
        print(f"time taken: {end-start}")
        print(f"x_new, plant_output: {x_new, plant_output} \n")
    
    plot_safe_region()

def test_Safeset_cons():
    x_new = jnp.array([1.00036055, -0.71751819])
    plant_output = GP_m.calculate_plant_outputs(x_new)
    GP_m.add_sample(x_new,plant_output)
    plot_safe_region()
        
def plot_safe_region():
    # Assuming GP_m.lcb is already defined and GP_m is properly initialized
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
    plt.figure()
    plt.contourf(X_0, X_1, mask_safe, levels=[0, 0.5, 1], colors=['lightcoral','lightblue'])
    plt.plot(X[:,0],X[:,1],'kx')
    plt.show()
        
    
if __name__ == "__main__":
    # test_GP_inference()
    # test_ucb()
    # test_lcb()
    # test_minimize_obj_ucb()
    # test_Minimizer()
    # test_create_point_arb()
    # test_lcb_arb()
    # test_Expander_constraint()
    # test_Expander() 
    # test_SafeOpt()
    test_Safeset_cons()

