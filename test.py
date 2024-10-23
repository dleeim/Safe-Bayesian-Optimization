import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
import random
from scipy.optimize import differential_evolution, NonlinearConstraint
from models import GP_TR
from problems import Benoit_Problem, WilliamOttoReactor_Problem
from problems import Rosenbrock_Problem
from utils import utils_SafeOpt

a = jnp.array([[1,2]])
b = jnp.array([[3,4]])
print(cdist(a,b))
print(jnp.linalg.norm(a-b))
