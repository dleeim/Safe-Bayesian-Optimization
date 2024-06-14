import numpy as np
# u is a list of lists (2D array)
u = np.array([1, 2, 3])
u = np.atleast_2d(u)

# Print all columns of the second row
print(u.shape)  
print(u)
