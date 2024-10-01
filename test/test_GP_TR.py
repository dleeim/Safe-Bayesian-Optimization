import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd

from models import GP_TR
from problems import Benoit_Problem
from problems import Rosenbrock_Problem

import warnings

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
X,Y = GP_m.Data_sampling(n_sample,x_old,r_old)
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
def test_minimize_obj_lcb():
    min_x, min_f = GP_m.minimize_obj_lcb(r_old,x_old)
    print(min_x,min_f)

def test_update_TR():
    plant_oldoutput = GP_m.calculate_plant_outputs(x_old)
    min_x, min_f = GP_m.minimize_obj_lcb(r_old,x_old)
    plant_newoutput = GP_m.calculate_plant_outputs(min_x)
    x_new, radius_new = GP_m.update_TR(x_old,min_x,r_old,plant_oldoutput,plant_newoutput)
    print(x_new,radius_new)

def test_GP_TR():

    # Initial Phase
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
    X,Y = GP_m.Data_sampling(n_sample,x_old,r_old)
    GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=5, var_out=True)

    plant_oldoutput = GP_m.calculate_plant_outputs(x_old)
    n_iteration = 10

    # Iteration
    for i in range(n_iteration):
        x_new, obj = GP_m.minimize_obj_lcb(r_old,x_old)
        plant_newoutput = GP_m.calculate_plant_outputs(x_new)
        x_update, r_new = GP_m.update_TR(x_old,x_new,r_old,plant_oldoutput,plant_newoutput)
        GP_m.add_sample(x_new,plant_newoutput)

        # Preparation for next iter:
        x_old = x_update
        r_old = r_new
        plant_oldoutput = plant_newoutput
        print("x_new,plant_newoutput,x_update,r_new")
        print(x_new,plant_newoutput,x_update,r_new)
        

if __name__ == "__main__":
    # test_minimize_obj_lcb()
    # test_update_TR()
    test_GP_TR()