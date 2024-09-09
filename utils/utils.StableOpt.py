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
    
data = jnp.load('data/data_StableOpt.npz')

def plot_W_shape(i):
    plt.figure()
    plt.plot(data['gridworld_input'][0],data['gridworld_output'][i])

def create_frame(function_for_plot,filename):
    function_for_plot
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

frame_duration = 700
filenames = []
for i in range(data['gridworld_output'].shape[0]):
    filename = f'frame_{i:02d}.png'
    create_frame(plot_W_shape(i),filename)
    filenames.append(filename)
GIFname = 'W_shaped_Outputs.gif'
create_GIF(frame_duration,filenames,GIFname)

