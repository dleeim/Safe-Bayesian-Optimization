import jax 
import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize
import sobol_seq
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

class BayesianOpt():
    
    ###########################
    # --- initializing GP --- #
    ###########################    
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True):
        '''
        Arguments:
            X                       : training data input
            Y                       : training data output
            kernel                  : type of kernel (RBF is only available)
            multi_hyper             : number of multistart for hyperparameter optimization
            var_out                 :
        Returns:
            X_norm                  : normalized training data input
            Y_norm                  : normalized training data output
        '''
        # GP variable definitions
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim   = X.shape[0], X.shape[1]
        self.ny_dim                 = Y.shape[1]
        self.multi_hyper            = multi_hyper
        self.var_out                = var_out

        # normalize data
        self.X_mean, self.X_std     = jnp.mean(X, axis=0), jnp.std(X, axis=0)
        self.Y_mean, self.Y_std     = jnp.mean(Y, axis=0), jnp.std(Y, axis=0)
        self.X_norm, self.Y_norm    = (X-self.X_mean)/self.X_std, (Y-self.Y_mean)/self.Y_std
        
    def seuclidean_jax(self, X, Y, V):
        '''
        Description:
            Compute standardized euclidean distance between 
            each pair of the two collections of inputs.
        Arguments:
            X                       : input data in shape (nA,n)
            Y                       : input data in shape (nB,n)
            V                       : Variance vector
        Returns:
            dis_mat                 : matrix with elements as standardized euclidean distance 
                                      between two input data X and Y
        '''
        V_sqrt = V**-0.5
        X_adjusted = X*V_sqrt
        Y_adjusted = Y*V_sqrt
        dist_mat = jnp.linalg.norm(X_adjusted[:,None,:]-Y_adjusted[None,:,:],axis=-1)
        
        return dist_mat
    
    def Cov_mat(self, kernel, X_norm, W, sf2):
        '''
        Description:  
            Calculates the covariance matrix of a dataset Xnorm.
            Note: cdist => sqrt(sum(u_i-v_i)^2/V[x_i])
        Arguments:
            kernel                  : type of kernel (RBF is only available) 
            X_norm                  : normalized training data input
            W                       : weight(=1/length scale) matrix
            sf2                     : variance for signal 
        Returns:
            cov_matrix              : covariance matrix
        '''
        if W.shape[0] != X_norm.shape[1]:
            raise ValueError('W and X_norm dimension should be same')
        
        elif kernel != 'RBF':
            raise ValueError('ERROR no kernel with name ', kernel)
        
        else:
            dist       = self.seuclidean_jax(X_norm, X_norm, W)**2
            cov_matrix = sf2 * jnp.exp(-0.5*dist)
            
            return cov_matrix
        
    def negative_loglikelihood(self, hyper, X, Y):
        '''
        Description:
            negative log likelihood of hyperparameters
        Arguments:
            hyper: 
            X:
            Y:
        Returns:
            NLL: negative log likelihood of 
        '''
        W = jnp.exp(2*hyper[:self.nx_dim])    # W <=> 1/lambda
        sf2 = jnp.exp(2*hyper[self.nx_dim])  # variance of the signal
        sn2 = jnp.exp(2*hyper[self.nx_dim+1]) # variance of noise

        K = self.Cov_mat(self.kernel, X, W, sf2)    # (nxn) covariance matrix (noise free)
        K = K + (sn2+1e-8) * jnp.eye(self.n_point)  # (nxn) covariance matrix
        K = (K + K.T)*0.5                           # ensure K is symmetric
        L       = jnp.linalg.cholesky(K)           # do a Cholesky decomposition
        logdetK = 2 * jnp.sum(jnp.log(jnp.diag(L))) # calculate the log of the determinant of K
        invLY   = jax.scipy.linalg.solve_triangular(L, Y, lower=True) # obtain L^{-1}*Y
        alpha   = jax.scipy.linalg.solve_triangular(L.T, invLY, lower=False) # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL     = jnp.dot(Y.T, alpha)[0][0] + logdetK      # construct the NLL

        return logdetK
            

if __name__ == '__main__':

    #########_________Test for __init__:
    print("#########_________Test for __init__:")
    # --- define training data --- #
    Xtrain = jnp.array([-4, -1, 1, 2]).reshape(-1,1)
    ytrain    = jnp.sin(Xtrain)
    print(f"Train data: \n Xtrain: {Xtrain.reshape(1,-1)} \n ytrain: {ytrain.reshape(1,-1)}")

    # --- GP initialization --- #
    GP_m = BayesianOpt(Xtrain, ytrain, 'RBF', multi_hyper=2, var_out=True)
    print(f"X norm: \n{GP_m.X_norm.reshape(1,-1)} \n Y norm: \n{GP_m.Y_norm.reshape(1,-1)}")

    #########_________Test for Cov_mat:
    print("#########_________Test for Cov_mat:")
    W = jnp.exp(jnp.array([0.]))
    sf2 = jnp.exp(jnp.array([0.]))
    cov_matrix = GP_m.Cov_mat
    print(f"covariance matrix: \n{cov_matrix('RBF',GP_m.X_norm,W,sf2)}")

    cov_grad = grad(cov_matrix, argnums=2)
    print(f"covariance grad w.r.t W: \n{cov_grad('RBF',GP_m.X_norm,W,sf2)}")

    #########_________Test for negative log likelihood:
    print("#########_________Test for negative log likelihood:")
    hyper = jnp.array([0.,0.,-5.])
    NLL = GP_m.negative_loglikelihood
    NLL_grad = grad(NLL,argnums=0)
    print(NLL(hyper,GP_m.X_norm,GP_m.Y_norm))
    print(NLL_grad(hyper,GP_m.X_norm,GP_m.Y_norm))
    pass
