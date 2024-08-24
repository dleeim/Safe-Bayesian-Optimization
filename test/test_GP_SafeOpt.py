import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
from models.GP_SafeOpt import GP
from problems import Benoit_Problem
from problems import Rosenbrock_Problem 

# --- Preparation --- #
jax.config.update("jax_enable_x64", True)
plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system_tight]
GP_m = GP(plant_system)
n_sample = 4
x_i = jnp.array([1.1,-0.8])
r = 0.5
b = 3
X,Y = GP_m.Data_sampling(n_sample,x_i,r)
print(f"initial sample: \n {X}")
print(f"initial output: \n {Y}")
GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=10, var_out=True)

def test_mean_var():
    x = jnp.array([1.19497006, -0.74191489])
    print(GP_m.GP_inference(x))

x_new = jnp.array([0.9, -0.6])
Y_new = jnp.zeros((1,GP_m.n_fun))
for i in range(GP_m.n_fun):
    Y_new = Y_new.at[:,i].set(plant_system[i](x_new))

def test_GP_inference_arbitrary():
    # --- Test --- #
    print(f"Check if GP inference and Data provides same value")
    print(f"x_new: {x_new}")

    GP_m.create_GP_arbitrary(x_new,Y_new)
    print(f"GP inference arbitrary: {GP_m.GP_inference_arbitrary(x_new)[0]}")

    GP_m.add_sample(x_new,Y_new)
    print(f"GP inference: {GP_m.GP_inference(x_new)[0]}")
    print(f"Data: {Y_new}")
    print(f"\n")

def test_check_mean_GP():
    x_test = jnp.array([10,10])
    print(f"check if GP inference at unknown region has constraint mean lower than 0")
    print(f"x_test: {x_test}")
    print(f"GP inference: {GP_m.GP_inference_np(x_test)[0]}")

    pass

if __name__ == "__main__":
    test_mean_var()
    # test_GP_inference_arbitrary()
    # test_check_mean_GP()

