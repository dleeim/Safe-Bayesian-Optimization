import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import time
from problems import Benoit_Problem
import imageio.v2 as imageio
from IPython.display import Image
import os
jax.config.update("jax_enable_x64", True)

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
