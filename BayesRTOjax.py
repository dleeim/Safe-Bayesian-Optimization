import time
import jax 
import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize
import sobol_seq

class BayesianOpt():
    
    def __init__(self,plant_system) -> None:
 
        self.plant_system               = plant_system
        self.n_fun                      = len(plant_system)
        self.key                        = jax.random.PRNGKey(42)
    
    ##################################
        # --- Data Sampling --- #
    ##################################
    
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
  

    #########################################
        # --- GP Initialization --- #
    #########################################

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
        self.hypopt, self.invKopt   = self.determine_hyperparameters()


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


    def determine_hyperparameters(self):
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
        multi_start= 1
        multi_startvec              = sobol_seq.i4_sobol_generate(self.nx_dim + 2, multi_start)
        options                     = {'disp':False,'maxiter':10000} # solver options
        hypopt                      = jnp.zeros((self.nx_dim+2, self.ny_dim)) # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol                    = [0.]*multi_start # values for multistart
        localval                    = jnp.zeros((multi_start)) # variables for multistart
        
        invKopt = []
        self.NLL_jit                = jit(self.negative_loglikelihood)
        NLL_grad                    = grad(self.negative_loglikelihood,argnums=0)
        
        for i in range(self.ny_dim):
            for j in range(multi_start):
                hyp_init            = jnp.array(lb + (ub - lb) * multi_startvec[j,:])

                res                 = minimize(self.NLL_jit, hyp_init, args=(self.X_norm, self.Y_norm[:,i:i+1]),
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
            - x                     : new data point
        Results:
            - mean sample           : mean of GP inference of x
            - var_sample            : variance of GP inference of x
        '''
        xnorm                       = (x-self.X_mean)/self.X_std
        mean                        = jnp.zeros((self.ny_dim))
        var                         = jnp.zeros((self.ny_dim))
        
        # --- Loop over each output (GP) --- #
        for i in range(self.ny_dim):
            invK                    = self.invKopt[i]
            hyper                   = self.hypopt[:,i]
            ellopt, sf2opt          = jnp.exp(2*hyper[:self.nx_dim]), jnp.exp(2*hyper[self.nx_dim])

            # --- determine covariance of each output --- #
            k                       = self.calc_Cov_mat(self.kernel,self.X_norm,xnorm,ellopt,sf2opt)
            mean                    = mean.at[i].set(jnp.matmul(jnp.matmul(k.T,invK),self.Y_norm[:,i])[0])
            var                     = var.at[i].set(jnp.maximum(0, (sf2opt - jnp.matmul(jnp.matmul(k.T,invK),k))[0,0]))

        # --- compute un-normalized mean --- # 
        mean_sample                 = mean*self.Y_std + self.Y_mean
        var_sample                  = var*self.Y_std**2
        
        if self.var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]

    #######################################################
                # --- Optimize Acquisition --- #          
    #######################################################

    def minimize_acquisition(self,r,x_0,data_storage,b=0,multi_start=1):
        '''
        Description:
            Find minimizer x* in a Gaussian Process Lower Confidence Bound
        Argument:
            - r                     : radius of trust region area
            - x_0                   : previous input observed
            - GP_m                  : Gaussian Process Model
            - b                     : parameter for exploration
        Results:
            - result.x              : a distance from input input_0 to observe the 
                                      minimal output of function
            - result.fun            : minimal output of function
        '''
        # Initialization
        options                     = {'disp':False, 'maxiter':10000,'ftol': 1e-12} 
        cons                        = []
        localsol                    = [x_0.tolist()]
        localval                    = [data_storage.data['plant_temporary'][0][0]]

        # Jit relavent class methods and JAX grad
        self.GP_inference_np_jit    = jit(self.GP_inference_np)
        obj_fun_jitgrad             = jit(grad(self.obj_fun))
        constraint_jitgrad          = jit(grad(self.constraint,argnums=0))
        TR_constraint_jitgrad       = jit(grad(self.TR_constraint,argnums=0))
        
        # Collect All objective function and constraints(model constraint + trust region)
        for i in range(self.n_fun):
            if i == 0:
                obj_fun             = lambda d: self.obj_fun(x_0+d, b)
                obj_grad            = lambda d: obj_fun_jitgrad(x_0+d, b)
            
            else: 
                cons.append({'type' : 'ineq',
                             'fun'  : lambda d: self.constraint(x_0+d,b,i),
                             'jac'  : lambda d: constraint_jitgrad(x_0+d,b,i)
                             })        
        
        cons.append({'type'         : 'ineq',
                     'fun'          : lambda d: self.TR_constraint(d,r),
                     'jac'          : lambda d: TR_constraint_jitgrad(d,r)
                     })
        
        # Perform Multistart Optimization
        self.key, subkey            = jax.random.split(self.key) 
        d0                          = self.Ball_sampling(self.nx_dim,multi_start,r,subkey)

        for j in range(multi_start):
            d0_j                    = d0[j,:]
            res                     = minimize(obj_fun, d0_j, constraints=cons, method='SLSQP', 
                                               jac=obj_grad,options=options,tol=1e-8)
            
            for con in cons:
                if con['fun'](res.x) < -0.01:
                    passed = False
                    break # Barrier when minimize significantly fails 
                else:
                    passed = True
            if passed:
                    localsol.append(res.x)
                    localval.append(res.fun)

        localsol                    = jnp.array(localsol)
        localval                    = jnp.array(localval)
        minindex                    = jnp.argmin(localval)
        xopt                        = localsol[minindex]
        funopt                      = localval[minindex]
        return xopt, funopt
    
    def obj_fun(self, x, b):
        GP_inference                = self.GP_inference_np_jit(x)
        mean                        = GP_inference[0][0]
        std                         = jnp.sqrt(GP_inference[1][0])
        value                       = mean - b*std

        return value

    def constraint(self, x, b, index):
        GP_inference                = self.GP_inference_np_jit(x)
        mean                        = GP_inference[0][index]
        std                         = jnp.sqrt(GP_inference[1][index])
        # value                       = mean - b*std
        value                       = mean

        return value
    
    def TR_constraint(self,d,r):
        value                       = r - jnp.linalg.norm(d+1e-8)
        # jax.debug.print("value: {}",value)
        return value
    
    ########################################################
                # --- Real Time Optimization --- #          
    ########################################################

    def RTOminimize(self,n_iter,x_initial,TR_parameters,multi_start,b):
        '''
        Description:
            Real-Time Optimization algorithm in (n_iter) iterations to conduct steps as follows:
                1. Minimizes Gaussian Process acquisition function in Trust Region to find new observations x_new
                2. Retrieve outputs from plant systems and re-train the Gaussian Process
        Arguments:
            - n_iter                : number of iterations
            - x_initial             : initial x value
            - radius                : radius for trust region
            - multi_start           : number of multi start optimization for each step
            - b                     : parameter for exploration in acquisition function
        Returns:
            - data                  : data stored during each step in Real-Time Optimization
        '''
        # Initialize Data Storage
        keys                        = ['i','x_initial','x_new','plant_output','TR_radius','plant_temporary']
        data_storage                = DataStorage(keys)

        # Retrieve radius from Trust Region Parameters
        radius = TR_parameters['radius']

        # Collect data at x_initial to DataStorage
        plant_output                = self.calculate_plant_outputs(x_initial)
        data_dict                   = self.create_data_points(0,x_initial,x_initial,plant_output,radius)
        data_storage.add_data_points(data_dict)
        
        # create temporary plant output for comparison in Trust Region Update
        data_storage.data['plant_temporary'].append(plant_output.tolist())

        # Real-Time Optimization
        for i in range(n_iter):

            # Bayesian Optimization
            d_new, obj              = self.minimize_acquisition(radius,x_initial,data_storage,multi_start=multi_start,b=b)

            # Retrieve Data from plant system
            plant_output            = self.calculate_plant_outputs(x_initial+d_new)

            # Collect Data to DataStorage
            data_dict               = self.create_data_points(i+1,x_initial,x_initial+d_new,plant_output,radius)
            data_storage.add_data_points(data_dict)

            # Trust Region Update:
            x_new, radius_new       = self.update_TR(x_initial,x_initial+d_new,radius,
                                                     TR_parameters,data_storage)
            
            # Add sample to Gaussian Process
            self.add_sample(x_initial+d_new,plant_output)

            # Preparation for next iter:
            x_initial               = x_new
            radius                  = radius_new
        
        data                        = data_storage.get_data()

        return data
    
    def create_data_points(self,iter,x_initial,x_new,plant_output,radius):
        data_dict = {
            'i'                     : iter,
            'x_initial'             : x_initial.tolist(),
            'x_new'                 : x_new.tolist(),
            'plant_output'          : plant_output.tolist(),
            'TR_radius'             : radius
        }

        return data_dict
    
    def calculate_plant_outputs(self,x):

        plant_output            = []
        for plant in self.plant_system:
            plant_output.append(plant(x)) 

        return jnp.array(plant_output)
    
    def update_TR(self,x_initial,x_new,radius,TR_parameters,data_storage):
        '''
        Description:
        Arguments:
        Returns:
            - x_new:
            - r: 
        '''
        # TR Parameters:
        r = radius
        r_max = TR_parameters['radius_max']
        r_red = TR_parameters['radius_red']
        r_inc = TR_parameters['radius_inc']
        rho_lb = TR_parameters['rho_lb']
        rho_ub = TR_parameters['rho_ub']
        # Check plant constraints to update trust region
        for i in range(self.n_fun-1):
            plant_const_now = data_storage.data['plant_output'][-1][i+1]
            
            if plant_const_now < 0:
                return x_initial, r*r_red
            else:
                pass

        # Calculate rho and use to update trust region
        plant_previous  = data_storage.data['plant_temporary'][0][0]
        plant_now       = data_storage.data['plant_output'][-1][0]
        GP_previous     = self.GP_inference_np_jit(x_initial)[0][0]
        GP_now          = self.GP_inference_np_jit(x_new)[0][0]

        rho             = (plant_now-plant_previous)/(GP_now-GP_previous+1e-8)

        
        if plant_previous < plant_now:
            return x_initial, r*r_red
        else:
            pass
        
        if rho < rho_lb:
            return x_initial, r*r_red
        
        elif rho >= rho_lb and rho < rho_ub: 
            data_storage.data['plant_temporary'][0][0] = plant_now
            return x_new, r
        
        else: # rho >= rho_ub
            data_storage.data['plant_temporary'][0][0] = plant_now
            return x_new, min(r*r_inc,r_max)
        
    
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
        self.hypopt, self.invKopt   = self.determine_hyperparameters()


class DataStorage:

    def __init__(self,keys):
        self.data = {}
        for key in keys:
            if type(key) == str:
                self.data[key] = []
            else:
                raise TypeError(f"Key '{key}' is not a string type")
    
    def add_data_points(self, data_dict):
        for key, new_data_point in data_dict.items():
            if key in self.data:
                self.data[key].append(new_data_point)
            else:
                raise KeyError(f"Key '{key}' not found in data sets")
        
    def get_data(self):
        for i in self.data.keys():
            self.data[i] = np.array(self.data[i])
            
        return self.data
    


