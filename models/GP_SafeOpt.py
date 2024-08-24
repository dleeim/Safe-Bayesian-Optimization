import jax 
import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize
import sobol_seq

class GP():
    def __init__(self,plant_system) -> None:
 
        self.plant_system               = plant_system
        self.n_fun                      = len(plant_system)
        self.key                        = jax.random.PRNGKey(42)
    
    ##################################
        # --- Data Sampling --- #
    ##################################
    
    def Ball_sampling(self,x_dim,n_sample,r_i,key):
        '''
        Description:
            This function samples randomly at (0,0) within a ball of radius r_i.
            By adding sampled point onto initial point (u1,u2) you will get 
            randomly sampled points around (u1,u2)
        Arguments:
            - x_dim                 : no of dimensions required for sampled point
            - n_sample              : number of sample required to create
            - r_i                   : radius from (0,0) of circle area for sampling
        Returns: 
            - d_init                : sampled distances from (0,0)
        '''
        x                           = jax.random.normal(key, (n_sample,x_dim))
        norm                        = jnp.linalg.norm(x,axis=-1).reshape(-1,1)
        r                           = (jax.random.uniform(key, (n_sample,1))) 
        d_init                      = r_i*r*x/norm
        return d_init
  
    def Data_sampling(self,n_sample,x_0,r):
        '''
        Description:
            sample input X from a circle with radius r and center at x_0
            obtain output Y from inserting input X into plant system
        Arguments:
            - n_sample              : number of data to be sampled
            - x_0                   : initial input / center of circle that input X will be sampled
            - r                     : radius of circle
            - plant_system          : [objective function, constraint 1, constraint 2, ...]
        Returns:
            - X, Y                  : sampled input X and output Y
        '''
        # === Collect Training Dataset (Input) === #
        self.key, subkey            = jax.random.split(self.key) 
        x_dim                       = jnp.shape(x_0)[0]                           
        X                           = self.Ball_sampling(x_dim,n_sample,r,subkey)
        X                           += x_0

        # === Collect Training Dataset === #
        n_fun                       = len(self.plant_system)
        Y                           = jnp.zeros((n_sample,n_fun))
        for i in range(n_fun):
            Y                       = Y.at[:,i].set(vmap(self.plant_system[i])(X))

        return X,Y

    #########################################
        # --- GP Initialization --- #
    #########################################

    def data_normalization(self):
        '''
        Description:
            normalize input X and output Y data in global variables
        Returns:
            X_norm                  : normalized input X
            Y_norm                  : normalized output Y
        '''
        self.X_mean, self.X_std     = jnp.mean(self.X, axis=0), jnp.std(self.X, axis=0)
        self.Y_mean, self.Y_std     = jnp.mean(self.Y, axis=0), jnp.std(self.Y, axis=0)
        X_norm, Y_norm              = (self.X-self.X_mean)/self.X_std, (self.Y-self.Y_mean)/self.Y_std

        return X_norm, Y_norm

    def squared_seuclidean_jax(self,X, Y, V):
        '''
        Description:
            Compute standardized Euclidean distance between 
            each pair of the two collections of inputs.
        Arguments:
            X                       : input data in shape (nA, n)
            Y                       : input data in shape (nB, n)
            V                       : Variance vector
        Returns:
            dis_mat                 : matrix with elements as standardized Euclidean distance 
                                      between two input data X and Y
        '''
        # Precompute the inverse square root of V
        V_sqrt_inv                  = V**-0.5

        # Apply the variance adjustment
        X_adjusted                  = X * V_sqrt_inv
        Y_adjusted                  = Y * V_sqrt_inv

        # Compute pairwise squared Euclidean distances efficiently
        dist_mat                    = -2 * jnp.dot(X_adjusted, Y_adjusted.T) + jnp.sum(X_adjusted**2, axis=1)[:, None] + jnp.sum(Y_adjusted**2, axis=1)
        
        return dist_mat

    def Cov_mat(self, kernel, X_norm, Y_norm, W, sf2):
        '''
        Description:  
            Calculates the covariance matrix of a dataset Xnorm.
        Arguments:
            kernel                  : type of kernel (currently RBF is available only) 
            X_norm                  : normalized training data input
            W                       : weight(=1/length scale) matrix
            sf2                     : variance for signal 
        Returns:
            cov_matrix              : covariance matrix
        '''
        if W.shape[0] != X_norm.shape[1]:
            raise ValueError('ERROR W and X_norm dimension should be same')
        elif kernel != 'RBF':
            raise ValueError('ERROR no kernel with name ', kernel)
        
        else:
            dist                    = self.squared_seuclidean_jax(X_norm,Y_norm, W)
            cov_matrix              = sf2 * jnp.exp(-0.5*dist) 
        
        return cov_matrix
    

    def calc_Cov_mat(self, kernel, X_norm, x_norm, ell, sf2):
        '''
        Description:  
            Calculates the covariance matrix of a dataset Xnorm and a sample data xnorm.
        Arguments:
            kernel                  : type of kernel (currently RBF is available only) 
            X_norm                  : normalized training data input
            ell                     : weight(=1/length scale) matrix
            sf2                     : variance for signal 
        Returns:
            cov_matrix              : covariance matrix
        '''
        x_norm = x_norm.reshape(1,self.nx_dim)
        if ell.shape[0] != X_norm.shape[1]:
            raise ValueError('ERROR W and X_norm dimension should be same')
        elif kernel != 'RBF':
            raise ValueError('ERROR no kernel with name ', kernel)
        
        else:
            dist                    = self.squared_seuclidean_jax(X_norm, x_norm, ell)
            cov_matrix              = sf2 * jnp.exp(-0.5*dist) 
            
            return cov_matrix

    def negative_loglikelihood(self, hyper, X, Y):
        '''
        Description:
            Negative log likelihood of hyperparameters.
        Arguments:
            hyper                   : hyperparameters (W,sf2,sn2)
            X                       : data input (usually normalized)
            Y                       : data output (usually normalized)
        Returns:
            NLL                     : negative log likelihood of hyperparameters (W,sf2,sn2)
        '''
        W                           = jnp.exp(2*hyper[:self.nx_dim]) # W <=> 1/lambda
        sf2                         = jnp.exp(2*hyper[self.nx_dim]) # variance of the signal
        sn2                         = jnp.exp(2*hyper[self.nx_dim+1]) # variance of noise
        K                           = self.Cov_mat(self.kernel, X, X, W, sf2) # (nxn) covariance matrix (noise free)
        K                           = K + (sn2+1e-8)*jnp.eye(self.n_point) # (nxn) covariance matrix
        K                           = (K + K.T)*0.5 # ensure K is symmetric
        L                           = jnp.linalg.cholesky(K) # do a Cholesky decomposition
        logdetK                     = 2 * jnp.sum(jnp.log(jnp.diag(L))) # calculate the log of the determinant of K
        invLY                       = jax.scipy.linalg.solve_triangular(L, Y, lower=True) # obtain L^{-1}*Y
        alpha                       = jax.scipy.linalg.solve_triangular(L.T, invLY, lower=False) # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL                         = jnp.dot(Y.T, alpha)[0][0] + logdetK # construct the NLL

        return NLL

    def determine_hyperparameters(self,X_norm,Y_norm,arbitrary:bool):
        '''
        Description:
            Determine optimal hyperparameter (W,sf2,sn2) given sample data input and output.
            Notice we construct one GP for each output dimension. 2 GP model for 2-d output.
        Arguments:
            None; uses global variables ie) self.X_norm, sample data input
        Result:
            - hypopt                : optimal hyperparameter (W,sf2,sn2)
            - invKopt               : inverse of covariance matrix with optimal hyperparameters 
        ''' 
        lb                          = jnp.array([-4.] * (self.nx_dim + 1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub                          = jnp.array([4.] * (self.nx_dim + 1) + [-2.])   # ub on parameters (this is inside the exponential)
        bounds                      = jnp.hstack((lb.reshape(self.nx_dim+2,1),
                                                  ub.reshape(self.nx_dim+2,1)))
        
        multi_start                 = self.multi_hyper # multistart on hyperparameter optimization
        multi_startvec              = sobol_seq.i4_sobol_generate(self.nx_dim + 2, multi_start)
        options                     = {'disp':False,'maxiter':10000} # solver options
        hypopt                      = jnp.zeros((self.nx_dim+2, self.ny_dim)) # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol                    = [0.]*multi_start # values for multistart
        localval                    = jnp.zeros((multi_start)) # variables for multistart
        invKopt = []

        if arbitrary:
            NLL_jit                     = jit(self.negative_loglikelihood_arb)
            NLL_grad                    = jit(grad(self.negative_loglikelihood_arb,argnums=0))
            n_point                     = self.n_point_arb
        else:
            NLL_jit                     = jit(self.negative_loglikelihood)
            NLL_grad                    = jit(grad(self.negative_loglikelihood,argnums=0))
            n_point                     = self.n_point

        
        for i in range(self.ny_dim):
            for j in range(multi_start):
                hyp_init            = jnp.array(lb + (ub - lb) * multi_startvec[j,:])

                res                 = minimize(NLL_jit, hyp_init, args=(X_norm, Y_norm[:,i:i+1]),
                                               method='SLSQP', options=options,bounds=bounds, jac=NLL_grad, tol=jnp.finfo(jnp.float32).eps)
                localsol[j]         = res.x
                localval            = localval.at[j].set(res.fun)

            # --- choosing best solution --- #
            minindex                = jnp.argmin(localval)
            hypopt                  = hypopt.at[:,i].set(localsol[minindex])
            ellopt                  = jnp.exp(2. * hypopt[:self.nx_dim,i])
            sf2opt                  = jnp.exp(2.*hypopt[self.nx_dim,i])
            sn2opt                  = jnp.exp(2.*hypopt[self.nx_dim+1,i]) + jnp.finfo(jnp.float32).eps

            Kopt                    = self.Cov_mat(self.kernel,X_norm,X_norm,ellopt,sf2opt) + sn2opt*jnp.eye(n_point)
            invKopt                 += [jnp.linalg.inv(Kopt)]
  
        return hypopt, invKopt

    def GP_initialization(self, X, Y, kernel, multi_hyper, var_out=True):
        '''
        Description:
            Initialize GP by using input data X, output data Y
        Arguments:
            - X                     : training data input
            - Y                     : training data output
            - kernel                : type of kernel (RBF is only available)
            - multi_hyper           : number of multistart for hyperparameter optimization
            - var_out               :
        Returns:
            - X_norm                : normalized training data input
            - Y_norm                : normalized training data output
        '''
        # GP variable definitions
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim   = self.X.shape[0], self.X.shape[1]
        self.ny_dim                 = self.Y.shape[1]
        self.multi_hyper            = multi_hyper
        self.var_out                = var_out

        # Normalize data
        self.X_norm,self.Y_norm     = self.data_normalization()

        # Find optimal hyperparameter and inverse of covariance matrix
        self.hypopt, self.invKopt   = self.determine_hyperparameters(self.X_norm,self.Y_norm,arbitrary=False)
    
    ###################################################
                # --- GP inference --- #
    ###################################################
    
    def GP_inference(self,x):
        '''
        Description:
            GP inference for a new data point x
        Argument:
            - x                     : new data point
        Results:
            - mean sample           : mean of GP inference of x
            - var_sample            : variance of GP inference of x
        '''
        xnorm                       = (x-self.X_mean)/self.X_std
        mean                        = jnp.zeros((self.ny_dim))
        var                         = jnp.zeros((self.ny_dim))
        
        # --- Set mean of constraints to be below 0 --- #
        mean_set                    = -1.
        mean_prior                  = (mean_set-self.Y_mean)/self.Y_std # Arbitrarily set prior mean = -1 Or you can change length hyperparameters
        mean_prior                  = mean_prior.at[0].set(0.0)
        
        # --- Loop over each output (GP) --- #
        for i in range(self.ny_dim):
            invK                    = self.invKopt[i]
            hyper                   = self.hypopt[:,i]
            ellopt, sf2opt          = jnp.exp(2*hyper[:self.nx_dim]), jnp.exp(2*hyper[self.nx_dim])

            # --- determine covariance of each output --- #
            k                       = self.calc_Cov_mat(self.kernel,self.X_norm,xnorm,ellopt,sf2opt)
            mean                    = mean.at[i].set(mean_prior[i]+jnp.matmul(jnp.matmul(k.T,invK),(self.Y_norm[:,i]-mean_prior[i]))[0])
            var                     = var.at[i].set(jnp.maximum(0, (sf2opt - jnp.matmul(jnp.matmul(k.T,invK),k))[0,0]))

        # --- compute un-normalized mean --- # 
        mean_sample                 = mean*self.Y_std + self.Y_mean
        var_sample                  = var*self.Y_std**2
        
        if self.var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]
        
    def add_sample(self,x_new,y_new):
        '''
        Description:
            Add new observation x_new and y_new into dataset
            and find new optimal hyperparameters and invers of cov mat
        Arguments:
            - x_new                 : new input data
            - y_new                 : new output data
        '''
        # Add the new sample to the data set
        self.X                      = jnp.vstack([self.X,x_new])
        self.Y                      = jnp.vstack([self.Y,y_new])
        self.n_point                = self.X.shape[0]

        # normalize data
        self.X_mean, self.X_std     = jnp.mean(self.X, axis=0), jnp.std(self.X, axis=0)
        self.Y_mean, self.Y_std     = jnp.mean(self.Y, axis=0), jnp.std(self.Y, axis=0)
        self.X_norm, self.Y_norm    = (self.X-self.X_mean)/self.X_std, (self.Y-self.Y_mean)/self.Y_std

        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters(self.X_norm,self.Y_norm,arbitrary=False)

    def negative_loglikelihood_arb(self, hyper, X, Y):
        '''
        Description:
            Negative log likelihood of hyperparameters for arbitrary GP.
        Arguments:
            hyper                   : hyperparameters (W,sf2,sn2)
            X                       : data input (usually normalized)
            Y                       : data output (usually normalized)
        Returns:
            NLL                     : negative log likelihood of hyperparameters (W,sf2,sn2)
        '''
        W                           = jnp.exp(2*hyper[:self.nx_dim]) # W <=> 1/lambda
        sf2                         = jnp.exp(2*hyper[self.nx_dim]) # variance of the signal
        sn2                         = jnp.exp(2*hyper[self.nx_dim+1]) # variance of noise
        K                           = self.Cov_mat(self.kernel, X, X, W, sf2) # (nxn) covariance matrix (noise free)
        K                           = K + (sn2+1e-8)*jnp.eye(self.n_point_arb) # (nxn) covariance matrix
        K                           = (K + K.T)*0.5 # ensure K is symmetric
        L                           = jnp.linalg.cholesky(K) # do a Cholesky decomposition
        logdetK                     = 2 * jnp.sum(jnp.log(jnp.diag(L))) # calculate the log of the determinant of K
        invLY                       = jax.scipy.linalg.solve_triangular(L, Y, lower=True) # obtain L^{-1}*Y
        alpha                       = jax.scipy.linalg.solve_triangular(L.T, invLY, lower=False) # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL                         = jnp.dot(Y.T, alpha)[0][0] + logdetK # construct the NLL

        return NLL

    def create_GP_arb(self,x_new,y_new):
        '''
        Description:
            Create arbitrary GP parameters by adding new observation x_new and y_new into arbitrary dataset
            and find arbitrary optimal hyperparameters and inverse of covmat. This is used in expander.
        Arguments:
            - x_new                 : new input data
            - y_new                 : new output data
        '''
        # Add new sample to arbitrary data set
        self.X_arb                  = jnp.vstack([self.X,x_new])
        self.Y_arb                  = jnp.vstack([self.Y,y_new])
        self.n_point_arb            = self.X_arb.shape[0]

        # normalize data
        self.X_mean_arb, self.X_std_arb     = jnp.mean(self.X_arb, axis=0), jnp.std(self.X_arb, axis=0)
        self.Y_mean_arb, self.Y_std_arb     = jnp.mean(self.Y_arb, axis=0), jnp.std(self.Y_arb, axis=0)
        self.X_norm_arb                     = (self.X_arb-self.X_mean_arb)/self.X_std_arb
        self.Y_norm_arb                     = (self.Y_arb-self.Y_mean_arb)/self.Y_std_arb

        # determine hyperparameters
        self.hypopt_arb, self.invKopt_arb   = self.determine_hyperparameters(self.X_norm_arb,self.Y_norm_arb,arbitrary=True) 

    def GP_inference_arb(self,x):
        '''
        Description:
            GP inference using arbitrary GP parameters for a new data point x
        Argument:
            - x                     : new data point
        Results:
            - mean sample           : mean of GP inference of x
            - var_sample            : variance of GP inference of x
        '''
        xnorm                       = (x-self.X_mean_arb)/self.X_std_arb
        mean                        = jnp.zeros((self.ny_dim))
        var                         = jnp.zeros((self.ny_dim))
        
        # --- Set mean of constraints to be below 0 --- #
        mean_set                    = -1.
        mean_prior                  = (mean_set-self.Y_mean_arb)/self.Y_std_arb # Arbitrarily set prior mean = -1 Or you can change length hyperparameters
        mean_prior                  = mean_prior.at[0].set(0.0)
        
        # --- Loop over each output (GP) --- #
        for i in range(self.ny_dim):
            invK                    = self.invKopt_arb[i]
            hyper                   = self.hypopt_arb[:,i]
            ellopt, sf2opt          = jnp.exp(2*hyper[:self.nx_dim]), jnp.exp(2*hyper[self.nx_dim])

            # --- determine covariance of each output --- #
            k                       = self.calc_Cov_mat(self.kernel,self.X_norm_arb,xnorm,ellopt,sf2opt)
            mean                    = mean.at[i].set(mean_prior[i]+jnp.matmul(jnp.matmul(k.T,invK),(self.Y_norm_arb[:,i]-mean_prior[i]))[0])
            var                     = var.at[i].set(jnp.maximum(0, (sf2opt - jnp.matmul(jnp.matmul(k.T,invK),k))[0,0]))

        # --- compute un-normalized mean --- # 
        mean_sample                 = mean*self.Y_std_arb + self.Y_mean_arb
        var_sample                  = var*self.Y_std_arb**2
        
        if self.var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]



        
