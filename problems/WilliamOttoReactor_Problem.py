import random 
import jax
from jax import vmap
import jax.numpy as jnp
from scipy.optimize import fsolve, differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

class WilliamOttoReactor():

    def __init__(self,measure_disturbance=False):
        self.key = jax.random.PRNGKey(42)
        self.subkey = self.key
        self.measure_disturbance = measure_disturbance

    def noise_generator(self):
        self.key, self.subkey = jax.random.split(self.key) 

    def odecallback(self,w, x, normal_noise):
        xa, xb, xc, xp, xe, xg = w
        Fa = 1.8275 
        Fb, Tr = x
        Fb += normal_noise
        Fr = Fa + Fb

        Vr = 2105.2 

        k1, k2, k3 = 1.6599e6, 7.2177e8, 2.6745e12
        eta1, eta2, eta3 = 6666.7, 8333.3, 11111

        k1 = k1*jnp.exp(-eta1/(Tr + 273))  
        k2 = k2*jnp.exp(-eta2/(Tr + 273))  
        k3 = k3*jnp.exp(-eta3/(Tr + 273))  

        df = [(Fa - (Fr)*xa - Vr*xa*xb*k1)/Vr,
            (Fb - (Fr)*xb - Vr*xa*xb*k1 - Vr*xb*xc*k2)/Vr,
            -(Fr)*xc/Vr + 2*xa*xb*k1 - 2*xb*xc*k2 - xc*xp*k3,
            -(Fr)*xp/Vr + xb*xc*k2 - 0.5*xp*xc*k3,
            -(Fr)*xe/Vr + 2*xb*xc*k2,
            -(Fr)*xg/Vr + 1.5*xp*xc*k3,
            ]

        return df

    def get_objective(self,u,noise=0.):
        x0 = jnp.array([0.1,0.1,0.1,0.1,0.1,0.1])
        normal_noise = jax.random.normal(self.subkey)
        normal_noise = jnp.clip(normal_noise,-2.05,2.05)
        normal_noise = normal_noise * jnp.sqrt(noise)

        Fa = 1.8275
        Fb, _ = u
        Fb += normal_noise

        fobj = lambda x: self.odecallback(x, u,normal_noise)
        xa, xb, xc, xp, xe, xg = fsolve(func=fobj, x0=x0)
        fx = (1043.38*xp*(Fa+Fb)+20.92*xe*(Fa+Fb) - 79.23*Fa - 118.34*Fb)
        
        if not self.measure_disturbance:
            return -fx
        if self.measure_disturbance:
            return -fx, normal_noise


    def get_constraint1(self,u,noise=0.):
        x0 = jnp.array([0.1,0.1,0.1,0.1,0.1,0.1])
        normal_noise = jax.random.normal(self.subkey)
        normal_noise = jnp.clip(normal_noise,-2.05,2.05)
        normal_noise = normal_noise * jnp.sqrt(noise)

        fobj = lambda x: self.odecallback(x, u,normal_noise)
        xa, xb, xc, xp, xe, xg = fsolve(func=fobj, x0=x0)
        g = jnp.array([0.12-xa])

        if not self.measure_disturbance:
            return g.item()
        if self.measure_disturbance:
            return g.item(), normal_noise
    
    def get_constraint2(self,u,noise=0.):
        x0 = jnp.array([0.1,0.1,0.1,0.1,0.1,0.1])
        normal_noise = jax.random.normal(self.subkey)
        normal_noise = jnp.clip(normal_noise,-2.05,2.05)
        normal_noise = normal_noise * jnp.sqrt(noise)

        fobj = lambda x: self.odecallback(x, u,normal_noise)
        xa, xb, xc, xp, xe, xg = fsolve(func=fobj, x0=x0)
        g = jnp.array([0.08-xg])
        
        if not self.measure_disturbance:
            return g.item()
        if self.measure_disturbance:
            return g.item(), normal_noise