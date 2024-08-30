import matplotlib.pyplot as plt

# Define the coordinates of the two points
x1, y1 = 0, 0  # First point
x2, y2 = 4, 3  # Second point

# Plot the dashed black line between the two points
plt.plot([x1, x2], [y1, y2], 'k--')

# Add labels and title for clarity
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Dashed Black Line Between Two Points')

# Show the plot
plt.grid(True)
plt.show()
