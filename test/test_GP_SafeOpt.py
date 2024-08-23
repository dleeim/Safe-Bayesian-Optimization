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
GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=10, var_out=True)

def test_GP_inference():
    # --- Initialization --- #
    x_1 = jnp.array([1.1,-0.8])
    mean = GP_m.GP_inference(x_1)[0][1]
    std = jnp.sqrt(GP_m.GP_inference(x_1)[1][1])
    print(f"constraint: {mean}")
    print(f"lcbconstraint: {mean-b*std}")
    print(Benoit_Problem.con1_system_tight(x_1))

x_new = jnp.array([0.9, -0.6])
Y_new = jnp.zeros((1,GP_m.n_fun))
for i in range(GP_m.n_fun):
    Y_new = Y_new.at[:,i].set(plant_system[i](x_new))

def test_GP_inference_arbitrary():
    GP_m.create_GP_arbitrary(x_new,Y_new)
    print(f"Check if GP inference and Data provides same value")
    print(f"GP inference: {GP_m.GP_inference_arbitrary(x_new)[0]}")
    print(f"Data: {Y_new}")

    GP_m.add_sample(x_new,Y_new)
    print(f"GP inference: {GP_m.GP_inference(x_new)[0]}")
    print(f"Data: {Y_new}")

if __name__ == "__main__":
    test_GP_inference()
    test_GP_inference_arbitrary()

