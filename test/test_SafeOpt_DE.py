import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd

from models import SafeOpt
from problems import Benoit_Problem

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
r = 0.5
X,Y = GP_m.Data_sampling(n_sample,x_i,r)
GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)

print(f"\n")
print(f"Data Sample Input:")
print(f"{X}")
print(f"Data Sample Output:")
print(f"{Y}")
print(f"\n")
print

# Tests
def test_GP_inference():
    i = 0
    x = jnp.array([1.08137762, -1.04739777])
    plant = GP_m.calculate_plant_outputs(x)
    print(f"Test: GP Inference; check if Gp inference well with low var")
    print(f"x input: {x}")
    print(f"GP Inference: {GP_m.GP_inference(x)}")
    print(f"Actual plant: {plant}")
    print(f"\n")


def test_ucb():
    i = 0
    x = jnp.array([1.49497006, -0.74191489])
    ucb = GP_m.ucb(x,i)
    obj_fun = GP_m.calculate_plant_outputs(x)[0]

    print(f"Test: ucb; check if objective ucb value at b=3 and actual objective value ares similar")
    print(f"x input: {x}")
    print(f"ucb: {ucb}")
    print(f"Actual obj fun: {obj_fun}")
    print(f"\n")

def test_lcb():
    i = 1
    x = jnp.array([1.49497006, -0.74191489])
    lcb = GP_m.lcb(x,i)
    constraint = GP_m.calculate_plant_outputs(x)[1]

    print(f"Test: lcb; check if constraint lcb value at b=3 is bigger than 0")
    print(f"x input: {x}")
    print(f"lcb: {lcb}")
    print(f"Actual constraint: {constraint}")
    print(f"\n")

def test_minimize_objective_ucb():
    x_min, ucb_min = GP_m.minimize_objective_ucb()

    print(f"Test: minimize objective ucb")
    print(f"x_min, ucb_min: {x_min, ucb_min}")
    print(f"\n")

def test_SafeOpt():
    n_iteration = 10
    for i in range(n_iteration):
        minimizer,std_minimizer = GP_m.Minimizer()
        expander,std_expander = GP_m.Expander()

        if std_minimizer > std_expander:
            x_new = minimizer
        else:
            x_new = expander
        
        plant_output = GP_m.calculate_plant_outputs(x_new)
        GP_m.add_sample(minimizer,plant_output)
    
    print(f"Test: SafeOpt")
    print(f"x_new, plant_output: {x_new, plant_output}")
    print(f"\n")

        
        

        
        

if __name__ == "__main__":
    test_GP_inference()
    test_ucb()
    test_lcb()
    test_minimize_objective_ucb()

