import random 
import jax
from jax import vmap
import jax.numpy as jnp
from scipy.optimize import fsolve, differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

class WilliamOttoReactor():

    def __init__(self,measure_disturbance=False):
        random_seed = random.randint(0, 2**32 - 1)
        self.key = jax.random.PRNGKey(random_seed)
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
            return g.item()
    
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
            return g.item()

    def reactor_drawing(self):
        n_sample = 50
        Fb = jnp.linspace(4,7,n_sample)
        Tr = jnp.linspace(70,100,n_sample)
        xx, yy = jnp.meshgrid(Fb,Tr)
        r1,r2 = xx.flatten(), yy.flatten()
        r1,r2 = jnp.reshape(r1,(len(r1),1)), jnp.reshape(r2,(len(r2),1))
        grid = jnp.hstack((r1,r2))
        cost = []
        for x in grid:
            self.solve_steady_state(x)
            cost.append(self.get_objective(x))
        cost = jnp.reshape(jnp.array(cost),jnp.shape(xx))
        plt.contour(xx,yy,cost)
        plt.colorbar()

    

if __name__ == "__main__":
    Reactor = WilliamOttoReactor()
    u = jnp.array([4.5,80.])
    sample_mean = []
    sample_con1 = []
    sample_con2 = []
    noise = 0.0002
    for i in range(10000):
        sample_mean.append(Reactor.get_objective(u,noise))
        sample_con1.append(Reactor.get_constraint1(u,noise))
        sample_con2.append(Reactor.get_constraint2(u,noise))
        Reactor.noise_generator()
    sample_mean = jnp.array(sample_mean)
    sample_con1 = jnp.array(sample_con1)
    sample_con2 = jnp.array(sample_con2)
    print('obj:',jnp.mean(sample_mean),jnp.std(sample_mean))
    print('con1:',jnp.mean(sample_con1),jnp.std(sample_con1))
    print('con2:',jnp.mean(sample_con2),jnp.std(sample_con2))

