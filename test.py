import jax
import numpy as np
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
key = jax.random.PRNGKey(0)
print(int(key))