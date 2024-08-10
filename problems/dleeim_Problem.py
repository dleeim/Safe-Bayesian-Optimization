def obj_fun(x):
    return x**8 - x**7 + x**6 - 0.1*x**5 - 3*x**2 + 2

def constraint_1(x):
    return -1.5*(x-0.5)**2 + 3

def constraint_2(x):
    return 1.5*(x-0.5)**2 - 3
