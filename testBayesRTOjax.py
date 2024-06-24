import jax 
import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize
# from scipy.optimize import minimize
import sobol_seq
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

class BayesianOpt():
    
    ###########################
    # --- initializing GP --- #
    ###########################    
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True):
        
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
        print(f"initial norms: {self.X_norm, self.Y_norm}")

        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters()        
    
    #############################
    # --- Covariance Matrix --- #
    #############################    
    
    def Cov_mat(self, kernel, X_norm, W, sf2):
        '''
        Calculates the covariance matrix of a dataset Xnorm
        '''
    
        if kernel == 'RBF':
            dist       = self.calc_cdist(X_norm, X_norm, W)**2 
            cov_matrix = sf2 * jnp.exp(-0.5 * dist)
            return cov_matrix
        else:
            raise ValueError('ERROR no kernel with name ', kernel)

    ################################
    # --- Covariance of sample --- #
    ################################    
        
    def calc_cov_sample(self, xnorm, Xnorm, ell, sf2):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        '''    
        dist = self.calc_cdist(Xnorm, xnorm.reshape(1, self.nx_dim), ell)**2
        cov_matrix = sf2 * jnp.exp(-0.5 * dist)
        return cov_matrix
    
    def calc_cdist(self, A, B, V):
        """
        Custom implementation of cdist using JAX.
        """
        A_scaled = A / jnp.sqrt(V)
        B_scaled = B / jnp.sqrt(V)
        return jnp.sqrt(jnp.sum((A_scaled[:, None, :] - B_scaled[None, :, :]) ** 2, axis=-1))
        
    ###################################
    # --- negative log likelihood --- #
    ###################################   
    
    def negative_loglikelihood(self, hyper, X, Y):
        '''
        Computes the negative log likelihood
        ''' 
        # internal parameters
        jax.debug.print('hyper: {}', hyper)
        n_point, nx_dim = self.n_point, self.nx_dim
        kernel          = self.kernel
        W               = jnp.exp(2 * hyper[:nx_dim])   # W <=> 1/lambda
        sf2             = jnp.exp(2 * hyper[nx_dim])    # variance of the signal 
        sn2             = jnp.exp(2 * hyper[nx_dim+1])  # variance of noise
    
        
        K       = self.Cov_mat(kernel, X, W, sf2)  # (nxn) covariance matrix (noise free)
        K       = K + (sn2 + 1e-8) * jnp.eye(n_point) # (nxn) covariance matrix
        K       = (K + K.T) * 0.5                  # ensure K is symmetric
        L       = jnp.linalg.cholesky(K)           # do a Cholesky decomposition
        logdetK = 2 * jnp.sum(jnp.log(jnp.diag(L))) # calculate the log of the determinant of K
        invLY   = jax.scipy.linalg.solve_triangular(L, Y, lower=True) # obtain L^{-1}*Y
        alpha   = jax.scipy.linalg.solve_triangular(L.T, invLY, lower=False) # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL     = jnp.dot(Y.T, alpha) + logdetK      # construct the NLL

        return NLL
    
    # Gradient of the negative log likelihood
    def negative_loglikelihood_with_grad(self, hyper, X, Y):
        return self.negative_loglikelihood(hyper, X, Y), grad(self.negative_loglikelihood, argnums=0)(hyper, X, Y)

    ############################################################
    # --- Minimizing the NLL (hyperparameter optimization) --- #
    ############################################################   
    
    def determine_hyperparameters(self):
        '''
        Determines the optimal hyperparameters by minimizing the negative log likelihood.
        '''   
        X_norm, Y_norm  = self.X_norm, self.Y_norm
        nx_dim, n_point = self.nx_dim, self.n_point
        kernel, ny_dim  = self.kernel, self.ny_dim
        
        lb               = jnp.array([-4.] * (nx_dim + 1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub               = jnp.array([4.] * (nx_dim + 1) + [-2.])   # ub on parameters (this is inside the exponential)
        bounds           = jnp.hstack((lb.reshape(nx_dim+2,1),
                                       ub.reshape(nx_dim+2,1)))
        multi_start      = self.multi_hyper                          # multistart on hyperparameter optimization
        multi_startvec   = sobol_seq.i4_sobol_generate(nx_dim + 2, multi_start)
        
        options  = {'disp':False, 'maxiter':10000}                   # Solver option
        hypopt   = jnp.zeros((nx_dim + 2, ny_dim))                   # hyperparams w's + sf2 + sn2 (one for each GP i.e. output var)
        localsol = [0.] * multi_start                                # values for multistart
        localval = jnp.zeros((multi_start))                          # variables for multistart

        invKopt = []
        for i in range(ny_dim):
            for j in range(multi_start):
                hyp_init    = lb + (ub - lb) * multi_startvec[j,:]
                res = minimize(self.negative_loglikelihood, hyp_init, args=(X_norm, Y_norm[:, i]),
                               method='SLSQP', options=options, bounds=bounds, jac='3-point', tol = 1e-12)
                # res = minimize(self.negative_loglikelihood,hyp_init,args=(X_norm, Y_norm[:, i]),method='BFGS',tol=1e-12)
                localsol[j] = res.x
                localval = localval.at[j].set(res.fun)

            # --- choosing best solution --- #
            minindex    = jnp.argmin(localval)
            hypopt      = hypopt.at[:,i].set(localsol[minindex])
            ellopt      = jnp.exp(2. * hypopt[:nx_dim,i])
            sf2opt      = jnp.exp(2. * hypopt[nx_dim,i])
            sn2opt      = jnp.exp(2. * hypopt[nx_dim+1,i])

            Kopt        = self.Cov_mat(kernel, X_norm, ellopt, sf2opt) + sn2opt * jnp.eye(n_point)
            invKopt     += [jnp.linalg.solve(Kopt, jnp.eye(n_point))]

        
        return hypopt, invKopt

    ########################
    # --- GP inference --- #
    ########################     
    
    def GP_inference_np(self, x):
        '''
        GP inference for new data point x
        '''
        nx_dim                   = self.nx_dim
        kernel, ny_dim           = self.kernel, self.ny_dim
        hypopt, Cov_mat          = self.hypopt, self.Cov_mat
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        calc_cov_sample          = self.calc_cov_sample
        invKsample               = self.invKopt
        Xsample, Ysample         = self.X_norm, self.Y_norm
        var_out                  = self.var_out

        xnorm = (x - meanX) / stdX
        mean  = jnp.zeros(ny_dim)
        var   = jnp.zeros(ny_dim)
        # --- Loop over each output (GP) --- #
        for i in range(ny_dim):
            invK           = invKsample[i]
            hyper          = hypopt[:,i]
            ellopt, sf2opt = jnp.exp(2 * hyper[:nx_dim]), jnp.exp(2 * hyper[nx_dim])

            # --- determine covariance of each output --- #
            k       = calc_cov_sample(xnorm, Xsample, ellopt, sf2opt)
            mean    = mean.at[i].set(jnp.dot(jnp.dot(k.T, invK), Ysample[:,i]))
            var     = var.at[i].set(jnp.maximum(0, sf2opt - jnp.dot(jnp.dot(k.T, invK), k))) # numerical error
            var[i]  = jnp.maximum(0, sf2opt - jnp.dot(jnp.dot(k.T, invK), k)) # numerical error

        # --- compute un-normalized mean --- #    
        mean_sample = mean * stdY + meanY
        var_sample  = var * stdY**2
        
        if var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]
        
    #########################################
    # --- Optimize Acquisition Function --- #
    #########################################    
        
    def aquisition_func(self, x, b):
        mean, var = self.GP_inference_np(x)
        return mean - b * var

    def optimize_acquisition(self, x0, b):
        #### ALERT! NEED TO CHANGE THE MINIMIZE USING JAX GRAD
        result = minimize(self.aquisition_func,x0,args=(b),method='SLSQP',options={'ftol': 1e-9})
        return result.params
    
    ##########################################
    # --- Add Sample and reinitialize GP --- #
    ##########################################

    def add_sample(self, x_new, y_new):
        # Add the new sample to the data set
        self.X         = jnp.vstack([self.X, x_new])
        self.Y         = jnp.vstack([self.Y, y_new])
        self.n_point   = self.X.shape[0]
        
        # normalize data
        self.X_mean, self.X_std     = jnp.mean(self.X, axis=0), jnp.std(self.X, axis=0)
        self.Y_mean, self.Y_std     = jnp.mean(self.Y, axis=0), jnp.std(self.Y, axis=0)
        self.X_norm, self.Y_norm    = (self.X - self.X_mean) / self.X_std, (self.Y - self.Y_mean) / self.Y_std

        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters()

# Test Cases
if __name__ == '__main__':

    ##### --- Test for Gaussian Process __init__ ---#####
    print("##### --- Test for Gaussian Process __init__ ---#####")
    # --- define training data --- #
    Xtrain = jnp.array([-4, -1, 1, 2])
    ndata  = Xtrain.shape[0]
    Xtrain = Xtrain.reshape(ndata,1)
    fx     = jnp.sin(Xtrain)
    ytrain = fx
    print(f"Train data: \n Xtrain: {Xtrain.reshape(1,-1)} \n ytrain: {ytrain.reshape(1,-1)}")

    # --- build a GP model --- #
    GP_m = BayesianOpt(Xtrain, ytrain, 'RBF', multi_hyper=2, var_out=True)

    # print(f"Sample X mean: {GP_m.X_mean}, Sample Y mean: {GP_m.Y_mean}")
    # print(f"Sample X norm: {GP_m.X_norm.T}, Sample Y norm: {GP_m.Y_norm.T}")

    # ##### --- Test for Gaussian Process __init__ ---#####
    # print("##### --- Test for Gaussian Process determine_hyperparameters ---#####")
    

    # ##### --- Test for Bayesian Optimization ---#####
    # print(f"\n ##### --- Test for Bayesian Optimization ---#####")
    
    # # --- (can ignore this function) function for creating file for a frame --- #
    # def create_frame(t,filename):
    #     n_test      = 100
    #     Xtest       = jnp.linspace(-10,10,n_test)
    #     fx_test     = jnp.sin(Xtest)
    #     Ytest_mean  = jnp.zeros(n_test)
    #     Ytest_std   = jnp.zeros(n_test)
    #     b           = 2
        
    #     plt.figure()

    #     # plot observed points
    #     plt.plot(GP_m.X, GP_m.Y, 'kx', mew=2)

    #     # plot the samples of posteriors
    #     plt.plot(Xtest, fx_test, 'black', linewidth=1)

    #     # --- use GP to predict test data --- #
    #     for ii in range(n_test):
    #         m_ii, std_ii   = GP_m.GP_inference_np(Xtest[ii])
    #         Ytest_mean[ii] = m_ii 
    #         Ytest_std[ii]  = std_ii

    #     # plot GP confidence intervals (+- b * standard deviation)
    #     plt.gca().fill_between(Xtest.flat, 
    #                         Ytest_mean - b*jnp.sqrt(Ytest_std), 
    #                         Ytest_mean + b*jnp.sqrt(Ytest_std), 
    #                         color='C0', alpha=0.2)

    #     # plot GP mean
    #     plt.plot(Xtest, Ytest_mean, 'C0', lw=2)

    #     plt.axis([-20, 20, -2, 3])
    #     plt.title(f'Gaussian Process Regression at iteration: {int(t*10)}')
    #     plt.legend(('training', 'true function', 'GP mean', 'GP conf interval'),
    #             loc='lower right')
        
    #     plt.savefig(filename)
    #     plt.close()

    # # --- check for Gaussian Process Model at initial state --- #
    # print(f"# --- check for Gaussian Process Model after initialization --- # \n")
    # print(f"Mean and Variance at x = -4: {GP_m.GP_inference_np(-4)})")
    # print(f"Mean and Variance at x = -7: {GP_m.GP_inference_np(-7)})")

    # # --- build Bayesian Optimization --- #
    # n_iter = 10
    # key = jax.random.PRNGKey(42)
    # x0 = jax.random.choice(key, Xtrain) # random choice from the train data
    # b = 2   # exploration factor

    # # --- Do Bayesian Optmization --- #
    # filenames = []
    # for i in range(n_iter):
        
    #     # create a frame
    #     t = i * 0.1
    #     filename = f'frame_{i:02d}.png'
    #     create_frame(t,filename)
    #     filenames.append(filename)

    #     # New Observation
    #     x_new = GP_m.optimize_acquisition(x0,b)
    #     y_new = np.sin(x_new)
    #     GP_m.add_sample(x_new,y_new)

    #     # For next iteration
    #     x0 = x_new


    #     if i == n_iter-1:
    #         # create a last frame
    #         t = n_iter * 0.1
    #         filename = f'frame_{n_iter:02d}.png'
    #         create_frame(t,filename)
    #         filenames.append(filename)

    # # create a GIF from saved frames
    # frame_duration = 1000
    # with imageio.get_writer('BayesOptforsine.gif', mode='I', duration=frame_duration) as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    
    # # remove individual frame files
    # for filename in filenames:
    #     os.remove(filename)


    # print(f"# --- check result on bayesian optimization --- # \n")
    # print(f"no of iteration: {n_iter}")
    # print(f"observation x: {GP_m.X.reshape(1,-1)}")
    # print(f"observation y: {GP_m.Y.reshape(1,-1)}")


    # # Test for Irregular Case   
    # print(f"# --- Test for Irregular Case with Error --- #") 
    # x = np.array([[ 8.14987947],
    #               [ 8.14987947],
    #               [ 8.14987947],
    #               [ 8.14987947],
    #               [-8.92389378],
    #               [-8.92389378]])
    
    # y = np.array([[ 0.95654072],
    #               [ 0.95654072],
    #               [ 0.95654072],
    #               [ 0.95654072],
    #               [-0.48020129],
    #               [-0.48020129]])

    # GP_m = BayesianOpt(x, y, 'RBF', multi_hyper=2, var_out=True)

    # t = 0
    # filename = 'TestforIrregularCase'
    # create_frame(t,filename)

    # x1 = -8.92389378
    # print(GP_m.GP_inference_np(x1))

