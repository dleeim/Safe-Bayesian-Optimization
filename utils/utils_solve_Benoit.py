import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import t
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import time
import imageio.v2 as imageio
from IPython.display import Image
import os
from models import SafeOpt
from problems import Benoit_Problem
jax.config.update("jax_enable_x64", True)

def plot_obj_con_outputs(data):
    n_samples = []
    obj_fun_outputs = []
    con_fun_outputs = []
    n_start = len(data.items())

    for i in range(n_start):
        observed_output = jnp.array(data[f'{i}'].item()['observed_output'])
        n_samples.append(observed_output.shape[0])
    
    max_n_sample = max(n_samples)
    
    for i in range(n_start):
        observed_output = jnp.array(data[f'{i}'].item()['observed_output'])
        obj_fun_outputs.append(observed_output[:,0])
        output_nan = jnp.array((max_n_sample-len(obj_fun_outputs[i]))*[jnp.nan])
        obj_fun_outputs[i] = jnp.append(obj_fun_outputs[i],output_nan)
        con_fun_outputs.append(observed_output[:,1])
        con_fun_outputs[i] = jnp.append(con_fun_outputs[i],output_nan)

    obj_fun_outputs = jnp.array(obj_fun_outputs)
    con_fun_outputs = jnp.array(con_fun_outputs)

    # Find mean and std
    obj_fun_mean = jnp.nanmean(obj_fun_outputs,axis=0)
    obj_fun_std = jnp.nanstd(obj_fun_outputs,axis=0,ddof=1)

    con_fun_mean = jnp.nanmean(con_fun_outputs,axis=0)
    con_fun_std = jnp.nanstd(con_fun_outputs,axis=0,ddof=2)
        
    # Find confidence interval
    degree_of_freedom = n_start-1
    confidence_interval = 0.975
    t_value = t.ppf(confidence_interval,degree_of_freedom)

    obj_fun_upper = obj_fun_mean + t_value*obj_fun_std/jnp.sqrt(n_start)
    obj_fun_lower = obj_fun_mean - t_value*obj_fun_std/jnp.sqrt(n_start)

    con_fun_upper = con_fun_mean + t_value*con_fun_std/jnp.sqrt(n_start)
    con_fun_lower = con_fun_mean - t_value*con_fun_std/jnp.sqrt(n_start)

    n_iter = len(obj_fun_mean)
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].plot(range(n_iter),obj_fun_mean)
    axs[0].fill_between(range(n_iter),obj_fun_upper,obj_fun_lower,alpha=0.3)
    axs[1].plot(range(n_iter),con_fun_mean)
    axs[1].fill_between(range(n_iter),con_fun_lower,con_fun_upper,alpha=0.3)
    axs[1].plot(np.array([0.]*n_iter),'r--',label='safety threshold')
    plt.show()
    
def plot_all_obj_fun():
    data_SafeOpt = jnp.load('data/data_multi_SafeOpt_Benoit.npz', allow_pickle=True)
    data_GoOSE = jnp.load('data/data_multi_GoOSE_Benoit.npz', allow_pickle=True)
    data_GP_TR = jnp.load('data/data_multi_GP_TR_Benoit.npz', allow_pickle=True)

    data_array = [data_SafeOpt,data_GoOSE,data_GP_TR]
    data_array_name = ["SafeOpt","GoOSE","GP_TR"]
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    
    for i in range(len(data_array)):
        data = data_array[i]
        n_samples = []
        obj_fun_outputs = []
        con_fun_outputs = []
        n_start = len(data.items())

        for j in range(n_start):
            observed_output = jnp.array(data[f'{j}'].item()['observed_output'])
            n_samples.append(observed_output.shape[0])
        
        max_n_sample = max(n_samples)

        for k in range(n_start):
            observed_output = jnp.array(data[f'{k}'].item()['observed_output'])
            obj_fun_outputs.append(observed_output[:,0])
            output_nan = jnp.array((max_n_sample-len(obj_fun_outputs[k]))*[jnp.nan])
            obj_fun_outputs[k] = jnp.append(obj_fun_outputs[k],output_nan)
            con_fun_outputs.append(observed_output[:,1])
            con_fun_outputs[k] = jnp.append(con_fun_outputs[k],output_nan)

        obj_fun_outputs = jnp.array(obj_fun_outputs)
        con_fun_outputs = jnp.array(con_fun_outputs)

        # Find mean and std
        obj_fun_mean = jnp.nanmean(obj_fun_outputs,axis=0)
        con_fun_mean = jnp.nanmean(con_fun_outputs,axis=0)

        n_iter = len(obj_fun_mean)

        axs[0].plot(range(n_iter),obj_fun_mean,label=data_array_name[i])
        axs[1].plot(range(n_iter),con_fun_mean,label=data_array_name[i])
        axs[1].plot(np.array([0.]*n_iter),'r--',label='safety threshold')

    axs[0].legend()
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Plant Output')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Constraint Output')
    axs[1].legend()
    plt.show()


plant_system = [Benoit_Problem.Benoit_System_1,
                Benoit_Problem.con1_system_tight]
bound = jnp.array([[-.6,1.5],[-1.,1.]])
b = 2.
GP_m = SafeOpt.BO(plant_system,bound,b)

# Load Data - Either SafeOpt, GoOSE, or GP_TR
data_Safe = jnp.load('data/data_multi_SafeOpt_Benoit.npz', allow_pickle=True)

# Plot data
plot_obj_con_outputs(data_Safe)

# Plot all data together
# plot_all_obj_fun()




