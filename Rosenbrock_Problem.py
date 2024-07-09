import numpy as np
import random
import jax
import jax.numpy as jnp
from jax import grad,jit,vmap
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import imageio.v2 as imageio
import pandas as pd
from IPython.display import Image

def Rosenbrock_f(x):
    z = (1.-x[0])**2 + 100*(x[1]-x[0]**2)**2

    return z

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

def plant_drawing(is_constraint=True,is_tight_constraint=False):
    delta = 0.01
    x = jnp.arange(-6., 6., delta)
    y = jnp.arange(-6., 6., delta)
    u = jnp.meshgrid(x, y)
    levels = jnp.logspace(-1,3.3,50)

    CS1 = plt.contour(u[0],u[1],Rosenbrock_f(u),levels,cmap='viridis',norm=LogNorm(),linestyles = "dashed",linewidths = 0.5)
    plt.clabel(CS1,inline=True)
    plt.axis((-2., 2., -1, 2.))

    if is_constraint:
            # Plot for constraint
            uc0 = 1. + x**2 + 2.*x - 2.
            plt.plot(uc0,x,'k')

    if is_tight_constraint:
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

def BRTO_Rosenbrock_drawing(data,iter):

    plt.figure()
    # Drawing for Benoit's Problem
    plant_drawing()

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
    with imageio.get_writer(gif_path, mode='I', duration=frame_duration) as writer:
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