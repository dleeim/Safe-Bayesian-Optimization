import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve

# 1. Scipy minimize
def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 
               + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

iteration_data = []

def callback_function(x):
    iteration_data.append(x)

res = minimize(rosen, x0, method='nelder-mead',
               options={'xtol': 1e-8, 'disp': True}, callback = callback_function)

optimal_solution = res.x

minimum_value = res.fun

for i, data in enumerate(iteration_data):
    print("Iteration {}: {}".format(i+1,data))


# 2. Scipy fsolve
# Define a system of nonlinear equations
def equations(vars,u):
    
    x, y = vars
    eq1 = x**2 + y**2 - 1
    eq2 = x - y
    return [eq1, eq2]

# Initial guess
initial_guess = [0.5, 0.5]

# Solve the system of nonlinear equations
u = (1,1)
result = fsolve(equations, initial_guess, args=(u))

print("Solution:", result)


