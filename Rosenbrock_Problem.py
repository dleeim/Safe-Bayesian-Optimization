def Rosenbrock_f(x):
    z = (1.-x[0])**2 + 100*(x[1]-x[0]**2)**2

    return z

def con2contour_plot(x):
    return x[0]**2 + 2*x[0] + 1  # = x
