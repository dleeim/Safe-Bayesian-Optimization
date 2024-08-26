import numpy as np

a = np.array([1,2,3])
mask = a > 1
print(a[not mask])