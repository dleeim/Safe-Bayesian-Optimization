import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a 2D grid of points
x = np.linspace(-2.0, 2.0, 400)
y = np.linspace(-2.0, 2.0, 400)
X, Y = np.meshgrid(x, y)

# Step 2: Define a function to compute the Z values
Z = np.sin(X**2 + Y**2)

# Step 3: Create the filled contour plot
plt.figure()
contour = plt.contourf(X, Y, Z, cmap='RdYlBu')

# Step 4: Add a colorbar
plt.colorbar(contour)

# Step 5: Show the plot
plt.title('Filled Contour Plot of sin(x^2 + y^2)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
