import numpy as np
from scipy.spatial.distance import cdist
import jax
import jax.numpy as jnp
from jax import grad

def Cov_mat(kernel, X_norm, W, sf2):
    '''
    Calculates the covariance matrix of a dataset Xnorm
    --- decription ---
    '''

    if kernel == 'RBF':
        dist       = cdist(X_norm, X_norm, 'seuclidean', V=W)**2
        cov_matrix = sf2*np.exp(-0.5*dist)

        return cov_matrix
        # Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
    else:
        print('ERROR no kernel with name ', kernel)

kernel = 'RBF'
X_norm = np.array([[-1.5275252,-1.5275252],
                    [-0.21821788,-0.21821788],
                    [0.65465367,0.65465367],
                    [1.0910894,1.0910894]])

print("X norms \n",X_norm)
W = np.exp([1,1])
sf2 = np.exp([2*0])

print("Cov_mat from np: \n",Cov_mat(kernel,X_norm,W,sf2))

def seuclidean_jax(X,Y,V):
    V_sqrt = V**-0.5
    X_adjusted = X * V_sqrt
    Y_adjusted = Y * V_sqrt
    
    dist_mat = jnp.linalg.norm(X_adjusted[:,None,:]-Y_adjusted[None,:,:]+1e-8,axis=-1)

    # dist_mat = jnp.zeros((4,4))
    # for i in range(Y.shape[0]):
    #     for j in range(X.shape[0]):
    #         a = jnp.sqrt(jnp.sum((X_adjusted[j]-Y_adjusted[i])**2)+1e-8)
    #         jax.debug.print("a: \n{}", a)
    #         dist_mat = dist_mat.at[i,j].set(a)
    return jnp.linalg.norm(dist_mat)

print("Cov_mat from jnp: \n",np.exp(-0.5*(seuclidean_jax(X_norm,X_norm,W)**2)))

seuclidean_grad = grad(seuclidean_jax, argnums=2)
print(seuclidean_grad(X_norm,X_norm,W))



