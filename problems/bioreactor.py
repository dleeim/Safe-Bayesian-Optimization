import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import optimize
jax.config.update("jax_enable_x64", True)

def bioreactor_ss_equations(input):
    # constants
    mu_max = 0.5
    Ks = 0.2
    Yxs = 0.5
    si = 20.
    # variables
    x,s,D = input
    mu = ((mu_max*s)/(Ks + s)) 

    # steady state equations
    df = [
        mu - D,
        D*si - D*s - ((mu * x)/(Yxs))    
    ]

    return df

def bioreactor_obj_x(x_sp,s_0,D_0):
    x0 = jnp.array([s_0,D_0])
    bioreactor_ss_equations_const_x = lambda x: bioreactor_ss_equations(jnp.append(x_sp,x))
    s,D = optimize.fsolve(bioreactor_ss_equations_const_x,x0)
    return D*x_sp, s, D

def bioreactor_obj_D(x_0,s_0,D_sp):
    x0 = jnp.array([x_0,s_0])
    bioreactor_ss_equations_const_x = lambda x: bioreactor_ss_equations(jnp.append(x,D_sp))
    x,s = optimize.fsolve(bioreactor_ss_equations_const_x,x0)
    return D_sp*x, x, s

# x_sp = 10.75
# s_0 = 1.
# D_0 = 1.
# val,s,D = bioreactor_obj_x(x_sp,s_0,D_0)
# print(val,s,D)

# obj = []
# D_vals = []
# x_sp_list = jnp.linspace(0,12,1000)
# s_0 = 1.
# D_0 = 1.
# for x_sp in x_sp_list:
#     val, s, D = bioreactor_obj_x(x_sp,s_0,D_0)
#     s_0, D_0 = s,D
#     obj.append(val)
#     D_vals.append(D)

# plt.figure()
# plt.plot(x_sp_list, obj)
# plt.xlabel('biomass concentration')
# plt.ylabel('productivity')
# plt.show()

# D_sp = 0.495
# x_0 = 10.
# s_0 = 0.8
# val,x,s = bioreactor_obj_D(x_0,s_0,D_sp)
# print(val,x,s)

obj = []
D_sp_list = jnp.linspace(0,0.495,1000) # Don't put 0.5 (an asymptote)
x_0 = 0.
s_0 = 0.
for D_sp in D_sp_list:
    val, x, s = bioreactor_obj_D(x_0,s_0,D_sp)
    x_0, s_0 = x,s
    obj.append(val)

plt.figure()
plt.plot(D_sp_list, obj)
plt.xlabel('biomass concentration')
plt.ylabel('productivity')
plt.show()