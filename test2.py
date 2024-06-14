import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

# Create a function to generate and save plots
def create_frame(t, filename):
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x - t)
    plt.figure()
    plt.plot(x, y)
    plt.title(f'Sine wave at t={t:.2f}')
    plt.xlabel('x')
    plt.ylabel('sin(x - t)')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Generate and save individual frames
filenames = []
for i in range(20):
    t = i * 0.1
    filename = f'frame_{i:02d}.png'
    create_frame(t, filename)
    filenames.append(filename)

# Set the duration for each frame in the GIF (e.g., 0.05 seconds per frame)
frame_duration = 1000  # Adjust this value to control the speed of the GIF

# Create a GIF from the saved frames
with imageio.get_writer('sine_wave.gif', mode='I', duration=frame_duration) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optionally, remove the individual frame files after creating the GIF
for filename in filenames:
    os.remove(filename)

print('GIF created successfully!')
