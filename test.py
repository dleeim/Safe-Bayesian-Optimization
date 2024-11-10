import numpy as np
import matplotlib.pyplot as plt

# Define the grid for plotting
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Define the two functions for the contours
Z1 = np.sin(X) * np.cos(Y)  # Example function 1
Z2 = np.exp(-(X**2 + Y**2) / 10)  # Example function 2

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the first contour with a specific color and style
contour1 = plt.contour(X, Y, Z1, levels=10, colors='blue', linestyles='dashed')
# plt.clabel(contour1, inline=True, fontsize=8, fmt="Func 1")  # Label inline

# Plot the second contour with a different color and style
contour2 = plt.contour(X, Y, Z2, levels=10, colors='red', linestyles='dashed')
# plt.clabel(contour2, inline=True, fontsize=8, fmt="Func 2")  # Label inline

# Add a legend
# plt.legend()  # No need to manually reference collections anymore

# Optional: Add titles and labels
plt.title('Contour Plots of Two Functions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()

