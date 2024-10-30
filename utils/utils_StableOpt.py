import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import pandas as pd
import time
import imageio.v2 as imageio
from IPython.display import Image
import os
from models import StableOpt
from problems import W_shape_Problem
jax.config.update("jax_enable_x64", True)

def plot_contour_W_shape(fig,axs,GP_m):
    # Contour for GP_m.lcb and GP_m.ucb
    xc = jnp.linspace(GP_m.bound[:,0],GP_m.bound[:,1],100)
    d = jnp.linspace(GP_m.bound_d[:,0],GP_m.bound_d[:,1],100)
    outputs_lcb = jnp.zeros((jnp.shape(xc)[0],jnp.shape(d)[0]))
    outputs_ucb = jnp.zeros((jnp.shape(xc)[0],jnp.shape(d)[0]))

    for i in range(jnp.shape(xc)[0]):
        for j in range(jnp.shape(d)[0]):
            output_lcb = GP_m.lcb(jnp.array([xc[i,0]]),jnp.array([d[j,0]]),0)
            outputs_lcb = outputs_lcb.at[i,j].set(output_lcb)  
            output_ucb = GP_m.ucb(jnp.array([xc[i,0]]),jnp.array([d[j,0]]),0)
            outputs_ucb = outputs_ucb.at[i,j].set(output_ucb)  

    xc, d = jnp.meshgrid(xc.flatten(),d.flatten())

    contourf1 = axs[0].contourf(xc,d,outputs_lcb,50,)
    contourf2 = axs[1].contourf(xc,d,outputs_ucb,50)
    fig.colorbar(contourf1)
    fig.colorbar(contourf2)


def plot_sampled_points(fig,axs):
    axs[0].plot(data['sampled_x'][:,0],data['sampled_x'][:,1],'rx')
    axs[1].plot(data['sampled_x'][:,0],data['sampled_x'][:,1],'rx')


def plot_observed_points(fig,axs,i):
    axs[0].plot(data['observed_x'][:i+1,0],data['observed_x'][:i+1,1],'ro',markersize=10)
    axs[1].plot(data['observed_x'][:i+1,0],data['observed_x'][:i+1,1],'ro',markersize=10)

def create_frame(i,filename,GP_m,Robust_Regret):
    fig, axs = plt.subplots(1, 2,figsize=(12,6))
    
    # Adjust tick label font size for both subplots
    axs[0].tick_params(axis='both', which='major', labelsize=8)
    axs[1].tick_params(axis='both', which='major', labelsize=8)

    # Plot Relevant Diagrams
    plot_contour_W_shape(fig,axs,GP_m)
    plot_sampled_points(fig,axs)
    plot_observed_points(fig,axs,i)

    # Store data for robust regret
    output_min = jnp.inf
    for j in data['observed_x'][:i+1]:
        xc_min = jnp.array([j[0]])
        dmax, value = GP_m.Maximise_d_with_constraints(GP_m.ucb,xc_min)
        
        output = GP_m.calculate_plant_outputs(xc_min,dmax)
        
        print(f"xc_min,dmax,output:{xc_min,dmax,output}")
        
        if output < output_min:
            output_min = output.item()
            xc_min_of_min = xc_min
            dmax_for_min = dmax
    
    print(f"_____xc_min_of_min,dmax_for_min,output_min {xc_min_of_min,dmax_for_min,output_min}")

    Robust_Regret.append(output_min-(-0.2961))

    plt.savefig(filename)
    plt.close()


def create_GIF(frame_duration, filenames, GIFname, output_dir='output'):
    gif_path = os.path.join(output_dir,GIFname)
    with imageio.get_writer(gif_path, mode='I',duration=frame_duration,loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
        for filename in filenames:
            os.remove(filename)

def create_Robust_Regret_Plot(Robust_Regret):
    plt.figure()
    plt.plot(Robust_Regret)
    plt.plot(jnp.array([0.]*len(Robust_Regret)))
    plt.xlabel("Iteration")
    plt.ylabel("Robust Regret")
    plt.savefig("output/W_shaped_Robust_Regret_Output.png")
    plt.close()


def create_W_Shape_outcomes():
    # Initialize GP and GIF formation
    frame_duration = 700
    filenames = []
    plant_system = [W_shape_Problem.W_shape]
    bound = jnp.array([[-1,2]])
    bound_d = jnp.array([[2,4]])
    b = 2.
    GP_m = StableOpt.BO(plant_system,bound,bound_d,b)
    GP_m.GP_initialization(data['sampled_x'],data['sampled_output'],'RBF',multi_hyper=5,var_out=True)
    Robust_Regret = []

    # Initial frame
    filename = f'frame_{0:02d}.png'
    create_frame(0,filename,GP_m,Robust_Regret)
    filenames.append(filename)

    # Create frame with adding observed points
    for i in range(1,data['observed_x'].shape[0]+1):
        GP_m.add_sample(data['observed_x'][i-1],data['observed_output'][i-1])
        filename = f'frame_{i:02d}.png'
        create_frame(i,filename,GP_m,Robust_Regret)
        filenames.append(filename)

    GIFname = 'W_shaped_Outputs.gif'
    create_GIF(frame_duration,filenames,GIFname)

    # Create plot for robust regret
    create_Robust_Regret_Plot(Robust_Regret)

# Load Data
data = jnp.load('data/data_StableOpt.npz')

# Check data
for key, value in data.items():
    print(f"{key}: {jnp.shape(value)}")

# Check data
for key, value in data.items():
    print(f"{key}: {value}")

# Create diagrams
create_W_Shape_outcomes()
# play() 








