import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from IPython.display import display, clear_output

# Create the figure and axis
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
line, = ax.plot(x, np.sin(x))

# Update function for the animation
def update(frame):
    line.set_ydata(np.sin(x - 0.1 * frame))
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)

# Control buttons
start_button = widgets.Button(description="Start")
stop_button = widgets.Button(description="Stop")
pause_button = widgets.Button(description="Pause")

# Animation control functions
def start_animation(b):
    ani.event_source.start()

def stop_animation(b):
    ani.event_source.stop()

def pause_animation(b):
    ani.event_source.stop()

# Link buttons to functions
start_button.on_click(start_animation)
stop_button.on_click(stop_animation)
pause_button.on_click(pause_animation)

# Display the plot and buttons
display(start_button, pause_button, stop_button, fig)

# Show the initial plot
plt.show()
