import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import time
from problems import WilliamOttoReactor_Problem
import imageio.v2 as imageio
from IPython.display import Image
import os
jax.config.update("jax_enable_x64", True)
'''
1. retrieve data at iteration = 3
2. draw contour line for WOR and its adversarially robust contour line
3. draw initial sample points 
4. draw points and lines
5.  a. for GoOSE need to draw oracle
    b. for GP-TR need to draw circles
    c. for StableOpt need to draw optimal solution and perturbated optimal solution in worst-case
'''

def generate_contour_contour_WilliamOttoReactor():
    # Reactor 
    Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()

    # Initial Parameters
    n_point = 100
    bound = jnp.array([[4.,7.],[70.,100.]])
    x_0 = jnp.linspace(bound[0,0],bound[0,1], n_point)
    x_1 = jnp.linspace(bound[1,0],bound[1,1], n_point)
    X_0,X_1 = jnp.meshgrid(x_0,x_1)

    # Flatten the meshgrid arrays
    X_0_flat = X_0.ravel()
    X_1_flat = X_1.ravel()

    # Stack them to creat points for lcb function
    X = jnp.column_stack((X_0_flat,X_1_flat))
    Y_objective = jnp.zeros(jnp.shape(X)[0])
    Y_constraint_1 = jnp.zeros(jnp.shape(X)[0])
    Y_constraint_2 = jnp.zeros(jnp.shape(X)[0])
    for i in range(len(X)):
        Y_objective = Y_objective.at[i].add(Reactor.get_objective(X[i]))
        print(i,Y_objective[i], Y_constraint_1[i], Y_constraint_2[i])

    Y_objective_reshaped = Y_objective.reshape(np.shape(X_0))

    data = {}
    data['X_0'] = X_0
    data['X_1'] = X_1
    data['Y_objective'] = Y_objective_reshaped

    jnp.savez('data/data_contour_WilliamOttoReactor.npz',**data)

def plot_WilliamOttoReactor():
    data_objective_contour = jnp.load('data/data_contour_WilliamOttoReactor.npz',allow_pickle=True)
    X_0 = data_objective_contour['X_0']
    X_1 = data_objective_contour['X_1']
    Y_objective = data_objective_contour['Y_objective']
    Y_constraint_1 = data_objective_contour['Y_constraint1']
    Y_constraint_2 = data_objective_contour['Y_constraint2']

    plt.figure(figsize=(8, 6))
    levels = jnp.linspace(-80,300,15)
    contour = plt.contour(X_0, X_1, Y_objective, levels,colors='k',linestyles = "dashed",linewidths = 0.5)
    plt.clabel(contour,inline=True)

    X_0 = X_0.reshape((-1,1))
    X_1 = X_1.reshape((-1,1))
    Y_constraint_1 = Y_constraint_1.reshape((-1,1))
    Y_constraint_2 = Y_constraint_2.reshape((-1,1))
    mask_1 = jnp.abs(Y_constraint_1) < 1e-3
    mask_2 = jnp.abs(Y_constraint_2) < 1e-3
    
    plt.plot(X_0[mask_1],X_1[mask_1],'k-')
    plt.plot(X_0[mask_2],X_1[mask_2],'k-')

    plt.title("Contour Plot of Objective Function")
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()

# generate_contour_contour_WilliamOttoReactor()

def plot_safe_region_Benoit(X,X_0,X_1,mask_safe,obj,bound,data=None):

    plt.figure()
    
    #Plot colors for safe and unsafe region
    plt.contourf(X_0, X_1, mask_safe, levels=[0.,0.5,1.], colors=['lightcoral','lightblue'],alpha=0.4)
    
    # Plot contour for plant objective function
    CS1 = plt.contour(X_0,X_1,obj.reshape(X_0.shape),colors='k',linestyles='dashed',linewidths=0.5)
    plt.clabel(CS1,inline=True)
    
    #Plot line for tight constraint
    x_0 = jnp.linspace(-1.5, 1.5, 400)
    uc0t = 1. +  x_0**2 + 2.*x_0
    plt.plot(uc0t,x_0,'k')

    # Plot point for global minimal point
    plt.plot(0.36845785, -0.39299271,'ro')

    # Plot points for initially sampled points
    plt.plot(X[:,0],X[:,1],'bo')

    # Plot point for new observed point
    if data != None:
        plt.plot(data['x_0'][:],data['x_1'][:],'ko',linewidth=1.,markersize=5)
        plt.plot(data['x_0'][:],data['x_1'][:],'k-',linewidth=0.5,label='_nolegend_')        

    # Set bound for plt plot
    plt.axis((bound[0,0],bound[0,1],bound[1,0],bound[1,1]))

Reactor = WilliamOttoReactor_Problem.WilliamOttoReactor()
u = jnp.array([5.,85.])
Reactor.get_objective(u)
