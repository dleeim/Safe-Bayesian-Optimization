import jax 
import jax.numpy as jnp
from jax import grad, value_and_grad
from scipy.optimize import minimize
from jax.scipy.optimize import minimize as jminimize
import sobol_seq
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

class BayesianOpt():
    
    ######################################################
                # --- initializing GP --- #
    ######################################################   
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

        self.hypopt, self.invKopt   = self.determine_hyperparameters()

    ######################################################
        # --- Standardized Euclidean Distance --- #
    ######################################################

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
        V_sqrt                      = V**-0.5
        X_adjusted                  = X*V_sqrt
        Y_adjusted                  = Y*V_sqrt

        # Need to add epsilon to prevent numerical instability caused when jax grad is used on 
        # negative log likelihood method
        epsilon                     = jnp.finfo(jnp.float64).eps
        dist_mat                    = jnp.linalg.norm(X_adjusted[:,None,:]-Y_adjusted[None,:,:]+epsilon,axis=-1)
        return dist_mat
    
    ######################################################
            # --- Covariance Matrix --- #
    ######################################################

    def Cov_mat(self, kernel, X_norm, Y_norm, W, sf2):
        '''
        Description:  
            Calculates the covariance matrix of a dataset Xnorm.
            Note: cdist => sqrt(sum(u_i-v_i)^2/V[x_i]).
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
            dist                    = self.seuclidean_jax(X_norm, Y_norm, W)**2
            cov_matrix              = sf2 * jnp.exp(-0.5*dist) 
            
            return cov_matrix
    
    ######################################################
          # --- Covariance Matrix for X sample --- #
    ######################################################
    
    def calc_Cov_mat(self, kernel, x_norm, X_norm, ell, sf2):
        '''
        Description:  
            Calculates the covariance matrix of a dataset Xnorm.
            Note: cdist => sqrt(sum(u_i-v_i)^2/V[x_i]).
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
            dist                    = self.seuclidean_jax(X_norm, x_norm, ell)**2
            cov_matrix              = sf2 * jnp.exp(-0.5*dist) 

            return cov_matrix

    ######################################################
            # --- negative log likelihood ---- #
    ######################################################

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
        sn2                         = jnp.exp(2*hyper[self.nx_dim+1]) + jnp.finfo(jnp.float32).eps # variance of noise

        K                           = self.Cov_mat(self.kernel, X, X, W, sf2) # (nxn) covariance matrix (noise free)
        K                           = K + sn2*jnp.eye(self.n_point) # (nxn) covariance matrix
        K                           = (K + K.T)*0.5 # ensure K is symmetric
        L                           = jnp.linalg.cholesky(K) # do a Cholesky decomposition
        logdetK                     = 2 * jnp.sum(jnp.log(jnp.diag(L))) # calculate the log of the determinant of K
        invLY                       = jax.scipy.linalg.solve_triangular(L, Y, lower=True) # obtain L^{-1}*Y
        alpha                       = jax.scipy.linalg.solve_triangular(L.T, invLY, lower=False) # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL                         = jnp.dot(Y.T, alpha)[0][0] + logdetK # construct the NLL

        return NLL
    
    ########################################################
    # --- Minimizing NLL (hyperparameter optimization) --- #
    ########################################################

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
        
        multi_start                 = self.multi_hyper                          # multistart on hyperparameter optimization
        multi_startvec              = sobol_seq.i4_sobol_generate(self.nx_dim + 2, multi_start)
        
        options                     = {'disp':False,'maxiter':10000}          # solver options
        hypopt                      = jnp.zeros((self.nx_dim+2, self.ny_dim))            # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol                    = [0.]*multi_start                        # values for multistart
        localval                    = jnp.zeros((multi_start))                 # variables for multistart
        
        invKopt = []
        NLL                         = self.negative_loglikelihood
        NLL_value_and_grad          = value_and_grad(NLL)
        for i in range(self.ny_dim):
            for j in range(multi_start):
                hyp_init            = jnp.array(lb + (ub - lb) * multi_startvec[j,:])
                res                 = minimize(NLL, hyp_init, args=(self.X_norm, self.Y_norm[:,i:i+1]),
                                               method='SLSQP', options=options,bounds=bounds, jac="3-point", tol=1e-12)
                localsol[j]         = res.x
                localval            = localval.at[j].set(res.fun)
            print("localsol: ",localsol)
            print("localval: ",localval)

            # --- choosing best solution --- #
            minindex                = jnp.argmin(localval)
            hypopt                  = hypopt.at[:,i].set(localsol[minindex])
            ellopt                  = jnp.exp(2. * hypopt[:self.nx_dim,i])
            sf2opt                  = jnp.exp(2.*hypopt[self.nx_dim,i])
            sn2opt                  = jnp.exp(2.*hypopt[self.nx_dim+1,i]) + jnp.finfo(jnp.float32).eps

            Kopt                    = self.Cov_mat(self.kernel,self.X_norm,self.X_norm,ellopt,sf2opt) + sn2opt*jnp.eye(self.n_point)
            invKopt                 += [jnp.linalg.inv(Kopt)]

            print("X norm: ", self.X_norm)
            print("Kopt: ", Kopt)

        return hypopt, invKopt
    
    ######################################################
                # --- GP inference --- #
    ######################################################

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
            hyper = self.hypopt[:,i:i+1]
            ellopt, sf2opt = jnp.exp(2*hyper[:self.nx_dim]), jnp.exp(2*hyper[self.nx_dim])

            # --- determine covariance of each output --- #
            k = self.calc_Cov_mat(self.kernel,xnorm,self.X_norm,ellopt,sf2opt)
            mean = mean.at[i].set(jnp.matmul(jnp.matmul(k.T,invK),self.Y_norm[:,i])[0])
            var = var.at[i].set(max(0, (sf2opt - jnp.matmul(jnp.matmul(k.T,invK),k))[0,0]))

        # --- compute un-normalized mean --- # 
        mean_sample = mean*self.Y_std + self.Y_mean
        var_sample = var*self.Y_std**2

        if self.var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]

    ######################################################
                # --- Acquisition Function --- #
    ######################################################

    def aquisition_func(self, x, b):
        if not self.var_out:
            raise ValueError('ERROR no variance provided from GP initialization (varout=False)')
        
        mean, var = self.GP_inference_np(x)
        mean_obj, var_obj = mean[0], var[0]
        return mean_obj - b*jnp.sqrt(var_obj)

    ######################################################
            # --- Optimize Acquisition Function --- #
    ######################################################

    def optimize_acquisition(self, x0, b):     
        acquisition_func_value_and_grad = value_and_grad(self.aquisition_func, argnums=0)
        result = minimize(acquisition_func_value_and_grad,x0,args=(b),
                          method='SLSQP',options={'ftol': 1e-9},jac=True)

        return result.x
    
    def add_sample(self,x_new,y_new):
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

    # #########_________Test for __init__:
    # print("#########_________Test for __init__:")
    # # --- define training data --- #
    # Xtrain = jnp.array([0.6977621,0.68817824,0.7073464,0.7169304,-1.2094676,-1.6007491]).reshape(-1,1)
    # ytrain    = jnp.sin(Xtrain)
    # nx_dim = Xtrain.shape[1]
    # print(f"Train data: \n Xtrain: {Xtrain.reshape(1,-1)} \n ytrain: {ytrain.reshape(1,-1)}")

    # # --- GP initialization --- #
    # GP_m = BayesianOpt(Xtrain, ytrain, 'RBF', multi_hyper=2, var_out=True)
    # print(f"X norm: \n{GP_m.X_norm.reshape(1,-1)} \n Y norm: \n{GP_m.Y_norm.reshape(1,-1)}")

    # #########_________Test for Cov_mat:
    # print("\n#########_________Test for Cov_mat:")
    # W = jnp.exp(2*jnp.array([0.29708328]))
    # sf2 = jnp.exp(2*jnp.array([0.87414811]))
    # sn2 = jnp.exp(2*jnp.array([-6.90056908]))
    # cov_matrix = GP_m.Cov_mat
    # print(f"covariance matrix: \n{cov_matrix('RBF',GP_m.X_norm,GP_m.X_norm,W,sf2)+sn2*jnp.eye(6)}")

    # #########_________Test for negative log likelihood:
    # print("\n#########_________Test for negative log likelihood:")
    # hyper = jnp.array([0.,0.,-5.])
    # NLL = GP_m.negative_loglikelihood
    # NLL_grad = grad(NLL,argnums=0)
    # NLL_value_and_grad = value_and_grad(NLL,argnums=0)
    # print(f"NLL: {NLL(hyper,GP_m.X_norm,GP_m.Y_norm)}")
    # print(f"NLL_grad: {NLL_grad(hyper,GP_m.X_norm,GP_m.Y_norm)}")
    # lb               = jnp.array([-4.] * (nx_dim + 1) + [-8.])  # lb on parameters (this is inside the exponential)
    # ub               = jnp.array([4.] * (nx_dim + 1) + [-2.])   # ub on parameters (this is inside the exponential)
    # bounds           = jnp.hstack((lb.reshape(-1,1),ub.reshape(-1,1)))
    # options  = {'disp':False, 'maxiter':10000} 

    # # optimal hyperparameters without bounds 
    # res = jminimize(GP_m.negative_loglikelihood,hyper,args=(GP_m.X_norm,GP_m.Y_norm),method='BFGS',tol=1e-12)
    # print(f"optimal hyperparameters without bounds(jax minimize BFGS): \n {res.x}")

    # # optimal hyperparameters with bounds but using jax = 'True'
    # res = minimize(NLL_value_and_grad, hyper, args=(GP_m.X_norm, GP_m.Y_norm),
    #                            method='SLSQP', options=options, bounds=bounds, jac=True, tol = 1e-12)
    # print(f"optimal hyperparameters with bounds(scipy minimize SLSQP, jax = True): \n {res.x}")

    # #########_________Test for determining optimal hyperparameter:
    # print("\n#########_________Test for determining optimal hyperparameter:")
    # print(f"optimal hyperparameter: \n{GP_m.hypopt} \ninverse of covariance matrix: \n{GP_m.invKopt}")
    
    # #########_________Test for GP_inference:
    # print("\n#########_________Test for GP_inference:")
    # x_new = jnp.array([-6.])
    # print(GP_m.GP_inference_np(x_new))
    
    # #########_________Test for acquisition_func:
    # print("\n#########_________Test for acquisition_func:")
    # b = 2
    # print(GP_m.aquisition_func(x_new,b))

    # #########_________Test for optimize acquisition_func:
    # print("\n#########_________Test for optimize_acquisition_func:")
    # x0 = jnp.array([-1.])
    # print(GP_m.optimize_acquisition(x0,b))

    # #########_________Test for add_sample:
    # print("\n#########_________Test for add_sample:")
    # x_new = jnp.array([-1.])
    # y_new = jnp.sin(x_new)
    # GP_m.add_sample(x_new,y_new)
    # print(f"new Xnorm: {GP_m.X_norm}, \nnew Y norm: {GP_m.Y_norm}")
    # print(f"new hypopt: {GP_m.hypopt}, \nnew invKopt: {GP_m.invKopt}")
    
    #########_________Test for Bayesian Optimization:
    print("\n#########_________Test for Bayesian Optimization:")

    # --- (can ignore this function) function for creating file for a frame --- #
    def create_frame(t,filename):
        n_test      = 200
        Xtest       = jnp.linspace(-20.,20.,n_test)
        fx_test     = jnp.sin(Xtest)
        Ytest_mean  = jnp.zeros(n_test)
        Ytest_std   = jnp.zeros(n_test)
        b           = 1.
        
        plt.figure()

        # plot observed points
        plt.plot(GP_m.X, GP_m.Y, 'kx', mew=2)

        # plot the samples of posteriors
        plt.plot(Xtest, fx_test, 'black', linewidth=1)

        # --- use GP to predict test data --- #
        for ii in range(n_test):
            m_ii, std_ii   = GP_m.GP_inference_np(Xtest[ii])
            Ytest_mean = Ytest_mean.at[ii].set(m_ii[0]) 
            Ytest_std = Ytest_std.at[ii].set(std_ii[0])

        # plot GP confidence intervals (+- b * standard deviation)
        plt.gca().fill_between(Xtest, 
                            Ytest_mean - b*jnp.sqrt(Ytest_std), 
                            Ytest_mean + b*jnp.sqrt(Ytest_std), 
                            color='C0', alpha=0.2)

        # plot GP mean
        plt.plot(Xtest, Ytest_mean, 'C0', lw=2)

        plt.axis([-20, 20, -2, 3])
        plt.title(f'Gaussian Process Regression at iteration: {int(t*10)}')
        plt.legend(('training', 'true function', 'GP mean', 'GP conf interval'),
                loc='lower right')
        
        plt.savefig(filename)
        plt.close()

    # --- build Bayesian Optimization --- #
    n_iter = 4
    key = jax.random.PRNGKey(42)
    # x0 = jax.random.choice(key, Xtrain, replace=True) # random choice from the train data
    x0 = jnp.array([-6.])
    b = 1.   # exploration factor

    # --- GP initialization --- #
    Xtrain = jnp.array([-4.01, -4.02, -4., -3.99]).reshape(-1,1)
    ytrain = jnp.sin(Xtrain)

    GP_m = BayesianOpt(Xtrain, ytrain, 'RBF', multi_hyper=2, var_out=True)

    # --- Do Bayesian Optmization --- #
    filenames = []
    for i in range(n_iter):
        
        # create a frame
        t = i * 0.1
        filename = f'frame_{i:02d}.png'
        create_frame(t,filename)
        filenames.append(filename)

        # New Observation
        x_new = GP_m.optimize_acquisition(x0,b)
        y_new = jnp.sin(x_new)
        GP_m.add_sample(x_new,y_new)

        # For next iteration
        x0 = x_new

        if i == n_iter-1:
            # create a last frame
            t = n_iter * 0.1
            filename = f'frame_{n_iter:02d}.png'
            create_frame(t,filename)
            filenames.append(filename)

    # create a GIF from saved frames
    frame_duration = 1000
    with imageio.get_writer('BayesOptforsine.gif', mode='I', duration=frame_duration) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # remove individual frame files
    for filename in filenames:
        os.remove(filename)