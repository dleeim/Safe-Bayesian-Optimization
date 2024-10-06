import jax
import numpy as np
import random
import jax.numpy as jnp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os
import pandas as pd
import imageio.v2 as imageio
from IPython.display import Image
jax.config.update("jax_enable_x64", True)

# Actual Plant System (if noise exists it equals to jnp.sqrt(1e-3))
def Benoit_System_1(u, noise = 0):

    f = u[0] ** 2 + u[1] ** 2 + u[0] * u[1]
    if noise: 
        f += random.gauss(0., jnp.sqrt(noise))
    
    return f

def Benoit_System_2(u, noise = 0):

    f = u[0] ** 2 + u[1] ** 2 + (1 - u[0] * u[1])**2
    if noise: 
        f += random.gauss(0., jnp.sqrt(noise))

    return f


def con1_system(u, noise = 0):

    g1 = 1. - u[0] + u[1] ** 2 + 2. * u[1] - 2.
    if noise:
        g1 -= random.gauss(0., jnp.sqrt(noise))

    return -g1


def con1_system_tight(u, noise = 0):
    
    g1 = 1. - u[0] + u[1] ** 2 + 2. * u[1] 
    if noise:
        g1 -= random.gauss(0., jnp.sqrt(noise))

    return -g1


# Model of Plant System
def Benoit_Model_1(theta, u):

    f = theta[0]*(u[0]) ** 2 + theta[1]*(u[1]) ** 2

    return f

def con1_Model(theta, u):

    g1 = 1. - theta[2]*(u[0]) + theta[3]*(u[1]) ** 2
    
    return -g1


# Functions for Drawing both plant system and 
# results of Real-Time Optimization with Bayesian Optimization

# Plot for objective function 
def plant_drawing(is_constraint=True,is_tight_constraint=True):
    delta = 0.01
    x = jnp.arange(-6.5, 6.5, delta)
    y = jnp.arange(-6.5, 6.5, delta)
    u = jnp.meshgrid(x, y)
    levels = jnp.linspace(0,50,26)

    CS1 = plt.contour(u[0],u[1],Benoit_System_1(u),levels,colors='k',linestyles = "dashed",linewidths = 0.5)
    plt.clabel(CS1,inline=True)
    plt.axis((-1.5, 6.0, -1.5, 1.5))

    if is_constraint:
            plt.plot(0,0,'kx')
            
            # Plot for constraint
            uc0 = 1. + x**2 + 2.*x - 2.
            plt.plot(uc0,x,'k')

    if is_tight_constraint:
            # Plot for optimal value for optimization with tightened constraint
            plt.plot(0.36845785, -0.39299271,'ko')

            # Plot for tightened constraint
            uc0t = 1. + x ** 2 + 2. * x
            plt.plot(uc0t,x,'m')

def trustregion_drawing(r,inputold_0,inputold_1):
       
    d_trial_x = jnp.linspace(-r, r, 50)
    d_trial_ypos= []
    d_trial_yneg = []
    equations = lambda d: [r - jnp.linalg.norm(d),d[0] - j]
    for j in d_trial_x: 
            initial_guess = [0,0]
            d = fsolve(equations,initial_guess)
            d_trial_ypos.append(d[1])
            d_trial_yneg.append(-d[1])

    d_trial_ypos = jnp.array(d_trial_ypos)
    d_trial_yneg = jnp.array(d_trial_yneg)

    plt.plot(d_trial_x+inputold_0,d_trial_ypos+inputold_1,'k-',linewidth=0.5)
    plt.plot(d_trial_x+inputold_0,d_trial_yneg+inputold_1,'k-',linewidth=0.5)

def BRTO_Benoit_drawing(data,iter):

    plt.figure()
    # Drawing for Benoit's Problem
    plant_drawing(is_constraint = False,is_tight_constraint = True)

    for i in range(iter+1):
        # Plot points for input observed
        plt.plot(data['x_new_0'][i],data['x_new_1'][i], 'ro')
        trustregion_drawing(data['TR_radius'][i],data['x_initial_0'][i],data['x_initial_1'][i])

        if i != 0:
            old_new_0 = [data['x_initial_0'][i],data['x_new_0'][i]]
            old_new_1 = [data['x_initial_1'][i],data['x_new_1'][i]]
            plt.plot(old_new_0,old_new_1,'b-',linewidth=1,label='_nolegend_')

        
def create_frame(fun_drawing,filename):
    fun_drawing
    plt.savefig(filename)
    plt.close()

def create_GIF(frame_duration,filenames,GIFname,output_dir='output'):
    # create a GIF from saved frames
    gif_path = os.path.join(output_dir, GIFname)
    with imageio.get_writer(gif_path, mode='I', duration=frame_duration, loop=0) as writer:
            for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
    # remove individual frame files
    for filename in filenames:
            os.remove(filename)

def plant_outputs_drawing(iteration,output,constraint,figname,output_dir='output'):
    # Create a figure and a set of subplots
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))

    # First subplot
    axs[0].plot(iteration, output)
    axs[0].set_xlabel('Iteration',fontsize=14)
    axs[0].set_ylabel('Plant Output',fontsize=14)
    axs[0].legend()

    # Second subplot
    axs[1].plot(iteration, constraint)
    axs[1].plot(iteration,np.array([0.]*len(iteration)),'r--',label='safety threshold')
    axs[1].set_xlabel('Iteration',fontsize=14)
    axs[1].set_ylabel('Plant Constraint',fontsize=14)
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, figname)
    plt.savefig(output_path)
    
