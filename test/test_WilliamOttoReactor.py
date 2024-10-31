import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.optimize import differential_evolution, NonlinearConstraint
from models import GP_TR, SafeOpt,GoOSE, StableOpt
from problems import Benoit_Problem, WilliamOttoReactor_Problem
from problems import Rosenbrock_Problem
from utils import utils_SafeOpt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Values in x were outside bounds during a minimize step, clipping to bounds")

# Class Initialization
Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()
plant_system = [Reactor.get_objective,
                Reactor.get_constraint1,
                Reactor.get_constraint2]
bound = jnp.array([[4.,7.],[70.,100.]])
b = 2.
GP_m = SafeOpt.BO(plant_system,bound,b)
n_start = 10
data = {}
noise = 0.01

for i in range(n_start):
    print(f"iteration: {i}")
    # Data Storage
    data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

    # GP Initialization: 
    x_i = jnp.array([6.8,80.])
    r = 0.3
    n_sample = 5
    X_sample = jnp.empty((0,len(x_i)))
    Y_sample = jnp.empty((0,len(plant_system)))
    for count in range(n_sample):
        X,Y = GP_m.Data_sampling(1,x_i,r,noise)
        Reactor.noise_generator
        X_sample=jnp.append(X_sample,X,axis=0)
        Y_sample=jnp.append(Y_sample,Y,axis=0)


def test_multiple_WilliamOttoReactor_GP_TR():
    # Class Initialization
    Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()
    plant_system = [Reactor.get_objective,
                    Reactor.get_constraint1,
                    Reactor.get_constraint2]
    bound = jnp.array([[4.,7.],[70.,100.]])
    b = 2.
    TR_parameters = {
        'radius_max': 1,
        'radius_red': 0.8,
        'radius_inc': 1.1,
        'rho_lb': 0.2,
        'rho_ub': 0.8
    }
    GP_m = GP_TR.BO(plant_system,bound,b,TR_parameters)
    n_start = 10
    data = {}
    noise = 0.001

    for i in range(n_start):
        print(f"iteration: {i}")
        # Data Storage
        data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

        # GP Initialization: 
        x_old = jnp.array([6.8,80.])
        r_old = 0.3
        n_sample = 5
        X_sample = jnp.empty((0,len(x_old)))
        Y_sample = jnp.empty((0,len(plant_system)))
        for count in range(n_sample):
            X,Y = GP_m.Data_sampling(1,x_old,r_old,noise)
            Reactor.noise_generator
            X_sample=jnp.append(X_sample,X,axis=0)
            Y_sample=jnp.append(Y_sample,Y,axis=0)
        
        GP_m.GP_initialization(X_sample, Y_sample, 'RBF', multi_hyper=5, var_out=True)
        plant_oldoutput = GP_m.calculate_plant_outputs(x_old,noise)
        data[f'{i}']['sampled_x'] = X_sample
        data[f'{i}']['sampled_output'] = Y_sample

        print(f"\n")
        print(f"Data Sample Input:")
        print(f"{X_sample}")
        print(f"Data Sample Output:")
        print(f"{Y_sample}")
        print(f"")
        # GP_TR
        n_iteration = 40

        for j in range(n_iteration):
            x_new, obj = GP_m.minimize_obj_lcb(r_old,x_old)
            Reactor.noise_generator()
            plant_newoutput = GP_m.calculate_plant_outputs(x_new,noise)
            x_update, r_new = GP_m.update_TR(x_old,x_new,r_old,plant_oldoutput,plant_newoutput)
            
            # Preparation for next iter:
            x_old = x_update
            r_old = r_new
            if jnp.all(x_update == x_new):
                plant_oldoutput = plant_newoutput
            
            GP_m.add_sample(x_new,plant_newoutput)

            # Store Data
            data[f'{i}']['observed_x'].append(x_new)
            data[f'{i}']['observed_output'].append(plant_newoutput)
            
    jnp.savez('data/data_multi_GP_TR_WilliamOttoReactor.npz',**data)

def test_multiple_WilliamOttoReactor_SafeOpt():
    # Class Initialization
    Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()
    plant_system = [Reactor.get_objective,
                    Reactor.get_constraint1,
                    Reactor.get_constraint2]
    bound = jnp.array([[4.,7.],[70.,100.]])
    b = 2.
    GP_m = SafeOpt.BO(plant_system,bound,b)
    n_start = 10
    data = {}
    noise = 0.001

    for i in range(n_start):
        print(f"iteration: {i}")
        # Data Storage
        data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

        # GP Initialization: 
        x_i = jnp.array([6.8,80.])
        r = 0.3
        n_sample = 5
        X_sample = jnp.empty((0,len(x_i)))
        Y_sample = jnp.empty((0,len(plant_system)))
        for count in range(n_sample):
            X,Y = GP_m.Data_sampling(1,x_i,r,noise)
            Reactor.noise_generator
            X_sample=jnp.append(X_sample,X,axis=0)
            Y_sample=jnp.append(Y_sample,Y,axis=0)
        
        GP_m.GP_initialization(X_sample, Y_sample, 'RBF', multi_hyper=5, var_out=True)
        
        data[f'{i}']['sampled_x'] = X_sample
        data[f'{i}']['sampled_output'] = Y_sample

        print(f"\n")
        print(f"Data Sample Input:")
        print(f"{X_sample}")
        print(f"Data Sample Output:")
        print(f"{Y_sample}")
        print(f"")

        # SafeOpt
        n_iteration = 40

        for j in range(n_iteration):
            # Create sobol_seq sample for Expander
            minimizer,std_minimizer = GP_m.Minimizer()
            print(f"minimizer,std_minimizer: {minimizer,std_minimizer}")
            expander,std_expander = GP_m.Expander()
            lipschitz_continuous = False
            print(f"expander,std_expander: {expander,std_expander}")

            if std_minimizer > std_expander or lipschitz_continuous == False:
                x_new = minimizer
            else:
                x_new = expander
            print(f"x_new: {x_new}")
            Reactor.noise_generator()
            plant_output = GP_m.calculate_plant_outputs(x_new,noise)
            GP_m.add_sample(x_new,plant_output)

            # Store Data
            data[f'{i}']['observed_x'].append(x_new)
            data[f'{i}']['observed_output'].append(plant_output)
    
    jnp.savez('data/data_multi_SafeOpt_WilliamOttoReactor.npz',**data)

def test_multiple_WilliamOttoReactor_GoOSE():
    # Class Initialization
    Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()
    plant_system = [Reactor.get_objective,
                    Reactor.get_constraint1,
                    Reactor.get_constraint2]
    bound = jnp.array([[4.,7.],[70.,100.]])
    b = 2.
    GP_m = GoOSE.BO(plant_system,bound,b)
    n_start = 10
    data = {}
    noise = 0.001
    # noise = 0.

    for i in range(n_start):
        print(f"iteration: {i}")
        # Data Storage
        data[f'{i}'] = {'sampled_x':[],'sampled_output':[],'observed_x':[],'observed_output':[]}

        # GP Initialization: 
        x_i = jnp.array([6.8,80.])
        r = 0.3
        n_sample = 5
        X_sample = jnp.empty((0,len(x_i)))
        Y_sample = jnp.empty((0,len(plant_system)))
        for count in range(n_sample):
            X,Y = GP_m.Data_sampling(1,x_i,r,noise)
            # Reactor.noise_generator
            X_sample=jnp.append(X_sample,X,axis=0)
            Y_sample=jnp.append(Y_sample,Y,axis=0)
        
        GP_m.GP_initialization(X_sample, Y_sample, 'RBF', multi_hyper=5, var_out=True)
        
        data[f'{i}']['sampled_x'] = X_sample
        data[f'{i}']['sampled_output'] = Y_sample

        print(f"\n")
        print(f"Data Sample Input:")
        print(f"{X_sample}")
        print(f"Data Sample Output:")
        print(f"{Y_sample}")
        print(f"")

        # GoOSE
        n_iteration = 20
        for j in range(n_iteration):
            x_safe_min,min_safe_lcb = GP_m.minimize_obj_lcb()
            print(f"x_safe_min,min_safe_lcb: {x_safe_min,min_safe_lcb}")
            x_target,target_lcb = GP_m.Target()
            print(f"x_target,target_lcb: {x_target,target_lcb}")
            if min_safe_lcb <= target_lcb or target_lcb == np.inf:
                x_new = x_safe_min
            else:
                x_safe_observe = GP_m.explore_safeset(x_target)
                x_new = x_safe_observe
            print(f"x_new: {x_new}")

            Reactor.noise_generator()
            plant_output = GP_m.calculate_plant_outputs(x_new,noise)
            GP_m.add_sample(x_new,plant_output)

            # Store Data
            data[f'{i}']['observed_x'].append(x_new)
            data[f'{i}']['observed_output'].append(plant_output)

            if abs(plant_system[0](x_new) + 76.03600894648002) <= 0.001:
                break
    
    jnp.savez('data/data_multi_GoOSE_WilliamOttoReactor.npz',**data)


# test_multiple_WilliamOttoReactor_GP_TR()
test_multiple_WilliamOttoReactor_GoOSE()
test_multiple_WilliamOttoReactor_SafeOpt()