import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.optimize import differential_evolution, NonlinearConstraint
from models import GP_TR
from problems import Benoit_Problem, WilliamOttoReactor_Problem
from problems import Rosenbrock_Problem
from utils import utils_SafeOpt
Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()
plant_system = [Reactor.get_objective,
                Reactor.get_constraint1,
                Reactor.get_constraint2]
bound = jnp.array([[4.,7.],[70.,100.]])

obj_fun = lambda x: plant_system[0](x)
cons = [NonlinearConstraint(lambda x: plant_system[1](x),0.,jnp.inf),
        NonlinearConstraint(lambda x: plant_system[2](x),0.,jnp.inf)]
result = differential_evolution(obj_fun,bound,constraints=cons)
print(result.x,result.fun)

result = jnp.array([4.38893395, 80.64981093])
for i in range(1,len(plant_system)):
    print(plant_system[i](result))


