import time
import jax 
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from jax.scipy.optimize import minimize as jminimize
import sobol_seq
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import Benoit_Problem

class BRTO():
    
    def __init__(self,plant_system) -> None:
        '''
        Global Variables:
        '''
        self.plant_system                   = plant_system
        self.n_fun                          = len(plant_system)
        self.key                            = jax.random.PRNGKey(42)
        self.squared_seuclidean_jax_jit     = jit(self.squared_seuclidean_jax)
 
    ###################################
        # --- Data Sampling --- #
    ###################################
    
    def Data_sampling(self,n_sample,x_0,r,plant_system):
        '''
        Description:
            sample input X from a circle with radius r and center at x_0
            obtain output Y from inserting input X into plant system
        Arguments:
            - n_sample: number of data to be sampled
            - x_0: initial input / center of circle that input X will be sampled
            - r: radius of circle
            - plant_system: [objective function, constraint 1, constraint 2, ...]
        Returns:
            - X, Y: sampled input X and output Y
        '''
        # === Collect Training Dataset (Input) === #
        self.key, subkey                    = jax.random.split(self.key) 
        x_dim                               = jnp.shape(x_0)[0]                           
        X                                   = self.Ball_sampling(x_dim,n_sample,r,subkey)
        X                                   += x_0

        # === Collect Training Dataset === #
        n_fun                               = len(plant_system)
        Y                                   = jnp.zeros((n_sample,n_fun))
        for i in range(n_fun):
            Y                               = Y.at[:,i].set(vmap(plant_system[i])(X))

        return X,Y
    
    def Ball_sampling(self,x_dim,n_sample,r_i,key):
        '''
        Description:
            This function samples randomly at (0,0) within a ball of radius r_i.
            By adding sampled point onto initial point (u1,u2) you will get 
            randomly sampled points around (u1,u2)
        Arguments:
            - x_dim                         : no of dimensions required for sampled point
            - n_sample                      : number of sample required to create
            - r_i                           : radius from (0,0) of circle area for sampling
        Returns: 
            - d_init                        : sampled distances from (0,0)
        '''
        x                                   = jax.random.normal(key, (n_sample,x_dim))
        norm                                = jnp.linalg.norm(x,axis=-1).reshape(-1,1)
        r                                   = (jax.random.uniform(key, (n_sample,1))) 
        d_init                              = r_i*r*x/norm
        return d_init
  

    #########################################
        # --- GP Initialization --- #
    #########################################

    def GP_initialization(self, X, Y, kernel, multi_hyper, var_out=True):
        '''
        Description:
            Initialize GP by using input data X, output data Y
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
        self.n_point, self.nx_dim   = self.X.shape[0], self.X.shape[1]
        self.ny_dim                 = self.Y.shape[1]
        self.multi_hyper            = multi_hyper
        self.var_out                = var_out

        # Normalize data
        self.X_norm,self.Y_norm     = self.data_normalization()

        # Find optimal hyperparameter and inverse of covariance matrix
        self.hypopt, self.invKopt   = self.determine_hyperparameters()


    def data_normalization(self):
        '''
        Description:
            normalize input X and output Y data in global variables
        Returns:
            X_norm: normalized input X
            Y_norm: normalized output Y
        '''
        self.X_mean, self.X_std     = jnp.mean(self.X, axis=0), jnp.std(self.X, axis=0)
        self.Y_mean, self.Y_std     = jnp.mean(self.Y, axis=0), jnp.std(self.Y, axis=0)
        X_norm, Y_norm    = (self.X-self.X_mean)/self.X_std, (self.Y-self.Y_mean)/self.Y_std

        return X_norm, Y_norm

    # def squared_seuclidean_jax(self,X, Y, V):
    #     '''
    #     Description:
    #         Compute standardized euclidean distance between 
    #         each pair of the two collections of inputs.
    #     Arguments:
    #         X                       : input data in shape (nA, n)
    #         Y                       : input data in shape (nB, n)
    #         V                       : Variance vector
    #     Returns:
    #         dis_mat                 : matrix with elements as standardized euclidean distance 
    #                                     between two input data X and Y
    #     '''
    #     V_sqrt = V**-0.5
    #     X_adjusted = X * V_sqrt
    #     Y_adjusted = Y * V_sqrt

    #     # Compute pairwise squared SEuclidean distances
    #     dist_mat = jnp.sum((X_adjusted[:, None, :] - Y_adjusted[None, :, :])**2, axis=-1)
        
    #     return dist_mat

    def squared_seuclidean_jax(self,X, Y, V):
        '''
        Description:
            Compute standardized Euclidean distance between 
            each pair of the two collections of inputs.
        Arguments:
            X       : input data in shape (nA, n)
            Y       : input data in shape (nB, n)
            V       : Variance vector
        Returns:
            dis_mat : matrix with elements as standardized Euclidean distance 
                      between two input data X and Y
        '''
        # Precompute the inverse square root of V
        V_sqrt_inv = V**-0.5

        # Apply the variance adjustment
        X_adjusted = X * V_sqrt_inv
        Y_adjusted = Y * V_sqrt_inv

        # Compute pairwise squared Euclidean distances efficiently
        dist_mat = -2 * jnp.dot(X_adjusted, Y_adjusted.T) + jnp.sum(X_adjusted**2, axis=1)[:, None] + jnp.sum(Y_adjusted**2, axis=1)
        
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
            dist                    = self.squared_seuclidean_jax_jit(X_norm, Y_norm, W)
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
            dist                    = self.squared_seuclidean_jax_jit(X_norm, x_norm, ell)
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


    def determine_hyperparameters(self):
        '''
        Description:
            Determine optimal hyperparameter (W,sf2,sn2) given sample data input and output.
            Notice we construct one GP for each output dimension. 2 GP model for 2-d output.
        Arguments:
            None; uses global variables ie) self.X_norm, sample data input
        Result:
            hypopt                  : optimal hyperparameter (W,sf2,sn2)
            invKopt                 : inverse of covariance matrix with optimal hyperparameters 
        ''' 
        lb                          = jnp.array([-4.] * (self.nx_dim + 1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub                          = jnp.array([4.] * (self.nx_dim + 1) + [-2.])   # ub on parameters (this is inside the exponential)
        bounds                      = jnp.hstack((lb.reshape(self.nx_dim+2,1),
                                                  ub.reshape(self.nx_dim+2,1)))
        
        # multi_start                 = self.multi_hyper                          # multistart on hyperparameter optimization
        multi_start= 1
        multi_startvec              = sobol_seq.i4_sobol_generate(self.nx_dim + 2, multi_start)
        options                     = {'disp':False,'maxiter':10000}          # solver options
        hypopt                      = jnp.zeros((self.nx_dim+2, self.ny_dim))            # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol                    = [0.]*multi_start                        # values for multistart
        localval                    = jnp.zeros((multi_start))                 # variables for multistart
        
        invKopt = []
        NLL                         = self.negative_loglikelihood
        NLL_grad                    = grad(NLL,argnums=0)
        
        for i in range(self.ny_dim):
            for j in range(multi_start):
                hyp_init            = jnp.array(lb + (ub - lb) * multi_startvec[j,:])

                res                 = minimize(NLL, hyp_init, args=(self.X_norm, self.Y_norm[:,i:i+1]),
                                               method='SLSQP', options=options,bounds=bounds, jac='3-point', tol=jnp.finfo(jnp.float32).eps)
                localsol[j]         = res.x
                localval            = localval.at[j].set(res.fun)

            # --- choosing best solution --- #
            minindex                = jnp.argmin(localval)
            hypopt                  = hypopt.at[:,i].set(localsol[minindex])
            ellopt                  = jnp.exp(2. * hypopt[:self.nx_dim,i])
            sf2opt                  = jnp.exp(2.*hypopt[self.nx_dim,i])
            sn2opt                  = jnp.exp(2.*hypopt[self.nx_dim+1,i]) + jnp.finfo(jnp.float32).eps

            Kopt                    = self.Cov_mat(self.kernel,self.X_norm,self.X_norm,ellopt,sf2opt) + sn2opt*jnp.eye(self.n_point)
            invKopt                 += [jnp.linalg.inv(Kopt)]
  
        return hypopt, invKopt
    
    ###################################################
                # --- GP inference --- #
    ###################################################

    def GP_inference_np(self,x):
        '''
        Description:
            GP inference for a new data point x
        Argument:
            x                       : new data point
        Results:
            mean sample             : mean of GP inference of x
            var_sample              : variance of GP inference of x
        '''
        xnorm = (x-self.X_mean)/self.X_std
        mean = jnp.zeros((self.ny_dim))
        var = jnp.zeros((self.ny_dim))
        
        # --- Loop over each output (GP) --- #
        for i in range(self.ny_dim):
            invK = self.invKopt[i]
            hyper = self.hypopt[:,i]
            ellopt, sf2opt = jnp.exp(2*hyper[:self.nx_dim]), jnp.exp(2*hyper[self.nx_dim])

            # --- determine covariance of each output --- #

            k = self.calc_Cov_mat(self.kernel,self.X_norm,xnorm,ellopt,sf2opt)
            mean = mean.at[i].set(jnp.matmul(jnp.matmul(k.T,invK),self.Y_norm[:,i])[0])
            var = var.at[i].set(max(0, (sf2opt - jnp.matmul(jnp.matmul(k.T,invK),k))[0,0]))

        # --- compute un-normalized mean --- # 
        mean_sample = mean*self.Y_std + self.Y_mean
        var_sample = var*self.Y_std**2
        
        if self.var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]

    #######################################################
                # --- Optimize Acquisition --- #          
    #######################################################

    def optimize_acquisition(self,r,x_0,b=0,multi_start=1):
        '''
        Description:
            Find minimizer x* in a Gaussian Process Lower Confidence Bound
        Argument:
            - r                             : radius of trust region area
            - x_0                       : previous input observed
            - GP_m                          : Gaussian Process Model
            - b                             : parameter for exploration
        Results:
            - result.x                      : a distance from input input_0 to observe the 
                                              minimal output of function
            - result.fun                    : minimal output of function
        '''
        options                             = {'disp':False, 'maxiter':10000} 
        cons                                = []
        localsol                            = [0.]*multi_start
        localval                            = jnp.zeros((multi_start))

        # Collect All objective function and constraints(model constraint + trust region)
        
        for i in range(self.n_fun):
            if i == 0:
                obj_fun                     = lambda d: self.obj_fun(x_0+d, b)
                obj_grad                    = lambda d: self.obj_fun_grad(x_0+d, b)
            else: 
                cons.append({'type'         : 'ineq',
                             'fun'          : lambda d: self.constraint(x_0+d,b,i),
                             'jac'          : lambda d: self.constraint_grad(x_0+d,b,i)
                             })
                
        cons.append({'type'                 : 'ineq',
                     'fun'                  : lambda d: self.TR_constraint(d,r),
                     'jac'                  : lambda d: self.TR_constraint_grad(d,r)
                     })
        
        # Multistart Optimization
        d0                                  = self.Ball_sampling(self.nx_dim,multi_start,r,self.key)
        for j in range(multi_start):
            d0_j                            = d0[j,:]
            print(f"iteration: {j}")
            print(f"initial x0: {x_0+d0_j}")

            res                             = minimize(obj_fun, d0_j, constraints=cons, method='SLSQP', 
                                                       jac=obj_grad,options=options)
            localsol[j] = res.x
            localval = localval.at[j].set(res.fun)

            print(f"final x0: {x_0+res.x}")
            print(f"final fun: {res.fun}")

        minindex = jnp.argmin(localval)
        xopt = localsol[minindex]
        funopt = localval[minindex]
        
        return xopt, funopt
    

    def obj_fun(self, x, b):
        GP_inference                        = self.GP_inference_np(x)
        mean                                = GP_inference[0][0]
        std                                 = jnp.sqrt(GP_inference[1][0])
        value                               = mean - b*std

        return value
    
    def add_grad(self,func):
        return grad(func,argnums=0)
    

    def obj_fun_grad(self, x, b):
        value                               = self.jitgrad(self.obj_fun,argnums=0)(x, b)
        # jax.debug.print("x: {}", x)
        # jax.debug.print("objfungrad: {}", value)
        return value 

    def constraint(self, x, b, index):
        GP_inference                        = self.GP_inference_np(x)
        mean                                = GP_inference[0][index]
        std                                 = jnp.sqrt(GP_inference[1][index])
        value                               = mean - b*std
        # jax.debug.print("x: {}", x)
        # jax.debug.print("const: {}", value)

        return value
    
    def constraint_grad(self, x, b, index):
        value                               = self.jitgrad(self.constraint,argnums=0)(x, b, index)
        # jax.debug.print("x: {}", x)
        # jax.debug.print("constgrad: {}", value)

        return value
    
    def TR_constraint(self,d,r):
        value                               = r - jnp.linalg.norm(d)
        # jax.debug.print("TR: {}", value)

        return value

    def TR_constraint_grad(self,d,r):
        value                               = self.jitgrad(self.TR_constraint,argnums=0)(d,r)
        # jax.debug.print("Trgrad: {}", value)

        return value

    #############################################
                # --- Add sample --- #          
    #############################################
    
    def add_sample(self,x_new,y_new):
        '''
        Description:
            Add new observation x_new and y_new into dataset
            and find new optimal hyperparameters and invers of cov mat
        Arguments:
            - x_new: new input data
            - y_new: new output data
        '''
        # Add the new sample to the data set
        self.X = jnp.vstack([self.X,x_new])
        self.Y = jnp.vstack([self.Y,y_new])
        self.n_point = self.X.shape[0]

        # normalize data
        self.X_mean, self.X_std     = jnp.mean(self.X, axis=0), jnp.std(self.X, axis=0)
        self.Y_mean, self.Y_std     = jnp.mean(self.Y, axis=0), jnp.std(self.Y, axis=0)
        self.X_norm, self.Y_norm    = (self.X-self.X_mean)/self.X_std, (self.Y-self.Y_mean)/self.Y_std

        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters()




if __name__ == '__main__':
    
    ##########################################
    ##### Test Case 1: GP_Initialization #####
    ##########################################

    # --- Preparation --- #
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system]
    GP_m = BRTO(plant_system)

    def func_mean(x,index):
        return GP_m.GP_inference_np(x)[0][index]
    
    def func_var(x,index):
        return GP_m.GP_inference_np(x)[1][index]
    
    def variances(x):
        return GP_m.GP_inference_np(x)[1]
    
    def change_jaxgrad(x,delta,func,index):
        func_grad = grad(func,argnums=0)(x,index)
        predicted_change = func_grad * delta
        return predicted_change
    
    def check_jaxgrad(x,delta,func,index):
        xdelta_0 = x + jnp.array([delta,0.])
        xdelta_1 = x + jnp.array([0.,delta])

        func_new_0 = func(xdelta_0,index)
        func_new_1 = func(xdelta_1,index)

        func_grad = grad(func,argnums=0)(x,index)
        predicted_change = func_grad * delta    

        func_newgrad_0 = func(x,index) + predicted_change[0]
        func_newgrad_1 = func(x,index) + predicted_change[1]

        print(f"\n check for accuracy for jax grad in {func}")
        print(f"input test: {x}")
        print(f"jaxgrad: {func_grad}")
        print(f"Actual new function after increase in 1st dim: {func_new_0}")
        print(f"Actual new function after increase in 2st dim: {func_new_1}")
        print(f"Predicted(jaxgrad) new function after increase in 1st dim: {func_newgrad_0}")
        print(f"Predicted(jaxgrad) new function after increase in 2st dim: {func_newgrad_1}")
        print(f"Difference between actual and predicted in 1st dim: {func_new_0 - func_newgrad_0}")
        print(f"Difference between actual and predicted in 1st dim: {func_new_1 - func_newgrad_1}")
    
    x_0 = jnp.array([1.4,-0.8])
    print(f"initial x: {x_0}")

    # --- Data Sampling --- #
    n_sample = 4
    r = 0.5
    X,Y = GP_m.Data_sampling(n_sample,x_0,r,plant_system)

    print("# --- Data Sampling --- #")
    print(f'X: \n{X}')
    print(f"Y: \n{Y}")

    # --- GP initialization --- #
    GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=2, var_out=True)
    print(f"GP mean: {GP_m.Y_mean}")

    # --- negative log likelihood --- #
    hyper = jnp.array([[ 0.,  0.,  0., -5.],
                       [ 2.,  -2.,   2.,  -6.5],])

    for i in range(hyper.shape[0]):
        NLL = GP_m.negative_loglikelihood(hyper[i],GP_m.X_norm,GP_m.Y_norm[:,i:i+1])
    for i in range(hyper.shape[0]):
        NLL = GP_m.negative_loglikelihood(hyper[i],GP_m.X_norm,GP_m.Y_norm[:,i:i+1])

    # --- Test Gaussian Process Model inference --- #
    x_1 = jnp.array([1.4,-0.8])
    GP_inference = GP_m.GP_inference_np(x_1)

    ## Check if plant and model provides similar output using sampled data as input
    print("\n#___Check if plant and model provides similar output using sampled data as input___#")
    print(f"test input: {x_1}")
    print(f"plant obj: {plant_system[0](x_1)}")
    print(f"model obj: {func_mean(x_1,index=0)}")
    print(f"plant con: {plant_system[1](x_1)}")
    print(f"model con: {func_mean(x_1,index=1)}")

    ## Check if variance is approx 0 at sampled input
    print("\n#___Check variance at sampled input___#")
    print(f"variance: {variances(x_1)}")

    # --- check gradient of GP_inference --- #

    # check objective function
    x_2 = jnp.array([1.4,-0.8])
    delta = 0.0001
    check_jaxgrad(x_2,delta,func_mean,index=0)

    # check constraint
    check_jaxgrad(x_2,delta,func_mean,index=1)

    # #############################################################
    # #### Test Case 2: Optimization of Lower Confidence Bound ####
    # #############################################################

    # --- Optimize Acquisition --- #
    r_i = 0.5
    d_new, obj = GP_m.optimize_acquisition(r_i,x_0,multi_start=5)
    print(f"optimal new input(model): {x_0+d_new}")
    print(f"corresponding new output(model): {obj}")
    print(f"Euclidean norm of d_new(model): {jnp.linalg.norm(d_new)}")
