# GP model from https://github.com/OptiMaL-PSE-Lab/Gaussian-Process-from-scratch

import numpy as np
import time
import numpy.random as rnd
from scipy.spatial.distance import cdist
import sobol_seq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
        self.X_mean, self.X_std     = np.mean(X, axis=0), np.std(X, axis=0)
        self.Y_mean, self.Y_std     = np.mean(Y, axis=0), np.std(Y, axis=0)
        self.X_norm, self.Y_norm    = (X-self.X_mean)/self.X_std, (Y-self.Y_mean)/self.Y_std

        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters()        
    
    #############################
    # --- Covariance Matrix --- #
    #############################    
    
    def Cov_mat(self, kernel, X_norm, W, sf2):
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

    ################################
    # --- Covariance of sample --- #
    ################################    
        
    def calc_cov_sample(self,xnorm,Xnorm,ell,sf2):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        --- decription ---
        '''    
        # internal parameters
        nx_dim = self.nx_dim

        dist = cdist(Xnorm, xnorm.reshape(1,nx_dim), 'seuclidean', V=ell)**2
        cov_matrix = sf2 * np.exp(-.5*dist)

        return cov_matrix                
        
    ###################################
    # --- negative log likelihood --- #
    ###################################   
    
    def negative_loglikelihood(self, hyper, X, Y):
        '''
        --- decription ---
        ''' 
        # internal parameters
        n_point, nx_dim = self.n_point, self.nx_dim
        kernel          = self.kernel
        
        W               = np.exp(2*hyper[:nx_dim])   # W <=> 1/lambda
        sf2             = np.exp(2*hyper[nx_dim])    # variance of the signal 
        sn2             = np.exp(2*hyper[nx_dim+1])  # variance of noise

        K       = self.Cov_mat(kernel, X, W, sf2)  # (nxn) covariance matrix (noise free)
        K       = K + (sn2 + 1e-8)*np.eye(n_point) # (nxn) covariance matrix
        K       = (K + K.T)*0.5                    # ensure K is simetric
        L       = np.linalg.cholesky(K)            # do a cholesky decomposition
        logdetK = 2 * np.sum(np.log(np.diag(L)))   # calculate the log of the determinant of K the 2* is due to the fact that L^2 = K
        invLY   = np.linalg.solve(L,Y)             # obtain L^{-1}*Y
        alpha   = np.linalg.solve(L.T,invLY)       # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL     = np.dot(Y.T,alpha) + logdetK      # construct the NLL

        return NLL
    
    ############################################################
    # --- Minimizing the NLL (hyperparameter optimization) --- #
    ############################################################   
    
    def determine_hyperparameters(self):
        '''
        --- decription ---
        Notice we construct one GP for each output
        '''   
        # internal parameters
        X_norm, Y_norm  = self.X_norm, self.Y_norm
        nx_dim, n_point = self.nx_dim, self.n_point
        kernel, ny_dim  = self.kernel, self.ny_dim
        Cov_mat         = self.Cov_mat
        
        
        lb               = np.array([-4.]*(nx_dim+1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub               = np.array([4.]*(nx_dim+1) + [ -2.])  # ub on parameters (this is inside the exponential)
        bounds           = np.hstack((lb.reshape(nx_dim+2,1),
                                      ub.reshape(nx_dim+2,1)))
        multi_start      = self.multi_hyper                   # multistart on hyperparameter optimization
        multi_startvec   = sobol_seq.i4_sobol_generate(nx_dim + 2,multi_start)

        options  = {'disp':False,'maxiter':10000}          # solver options
        hypopt   = np.zeros((nx_dim+2, ny_dim))            # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol = [0.]*multi_start                        # values for multistart
        localval = np.zeros((multi_start))                 # variables for multistart

        invKopt = []
        # --- loop over outputs (GPs) --- #
        for i in range(ny_dim):

            # --- multistart loop --- # 
            for j in range(multi_start):
                print('multi_start hyper parameter optimization iteration = ',j,'  input dimension = ',i)
                hyp_init    = lb + (ub-lb)*multi_startvec[j,:]
                # --- hyper-parameter optimization --- #
                res = minimize(self.negative_loglikelihood,hyp_init,args=(X_norm,Y_norm[:,i])\
                               ,method='SLSQP',options=options,bounds=bounds,tol=1e-12)
                localsol[j] = res.x
                localval[j] = res.fun

            # --- choosing best solution --- #
            minindex    = np.argmin(localval)
            hypopt[:,i] = localsol[minindex]
            ellopt      = np.exp(2.*hypopt[:nx_dim,i])
            sf2opt      = np.exp(2.*hypopt[nx_dim,i])
            sn2opt      = np.exp(2.*hypopt[nx_dim+1,i]) + 1e-8

            # --- constructing optimal K --- #
            Kopt        = Cov_mat(kernel, X_norm, ellopt, sf2opt) + sn2opt*np.eye(n_point)
            # --- inverting K --- #
            invKopt     += [np.linalg.solve(Kopt,np.eye(n_point))]

        return hypopt, invKopt

    ########################
    # --- GP inference --- #
    ########################     
    
    def GP_inference_np(self, x):
        '''
        --- decription ---
        '''
        nx_dim                   = self.nx_dim
        kernel, ny_dim           = self.kernel, self.ny_dim
        hypopt, Cov_mat          = self.hypopt, self.Cov_mat
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        calc_cov_sample          = self.calc_cov_sample
        invKsample               = self.invKopt
        Xsample, Ysample         = self.X_norm, self.Y_norm
        var_out                  = self.var_out
        # Sigma_w                  = self.Sigma_w  # (if input noise)

        xnorm = (x - meanX)/stdX
        mean  = np.zeros(ny_dim)
        var   = np.zeros(ny_dim)
        # --- Loop over each output (GP) --- #
        for i in range(ny_dim):
            invK           = invKsample[i]
            hyper          = hypopt[:,i]
            ellopt, sf2opt = np.exp(2*hyper[:nx_dim]), np.exp(2*hyper[nx_dim])

            # --- determine covariance of each output --- #
            k       = calc_cov_sample(xnorm,Xsample,ellopt,sf2opt)
            mean[i] = np.matmul(np.matmul(k.T,invK),Ysample[:,i])
            var[i]  = max(0, sf2opt - np.matmul(np.matmul(k.T,invK),k)) # numerical error
            # var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k)  # (if input noise)

        # --- compute un-normalized mean --- #    
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2
        
        if var_out:
            return mean_sample, var_sample
        else:
            return mean_sample.flatten()[0]
        
    #########################################
    # --- Optimize Acquisition Function --- #
    #########################################    
        
    def aquisition_func(self,x,b):
        mean, var = self.GP_inference_np(x)

        return mean - b*var

    def optimize_acquisition(self,x0,b):
        result = minimize(self.aquisition_func,x0,args=(b),method='SLsQP',options={'ftol': 1e-9})

        return result.x
    
    ##########################################
    # --- Add Sample and reinitialize GP --- #
    ##########################################

    def add_sample(self,x_new,y_new):
        # Add the new sample to the data set
        self.X         = np.vstack([self.X,x_new])
        self.Y         = np.vstack([self.Y,y_new])
        self.n_point   = self.X.shape[0]
        
        # normalize data
        self.X_mean, self.X_std     = np.mean(self.X, axis=0), np.std(self.X, axis=0)
        self.Y_mean, self.Y_std     = np.mean(self.Y, axis=0), np.std(self.Y, axis=0)
        self.X_norm, self.Y_norm    = (self.X-self.X_mean)/self.X_std, (self.Y-self.Y_mean)/self.Y_std

        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters() 

        pass

##### --- Data ---#####



# ##### --- Data ---#####
# # --- define training data --- #
# Xtrain = np.array([-4, -1, 1, 2])
# ndata  = Xtrain.shape[0]
# Xtrain = Xtrain.reshape(ndata,1)
# fx     = np.sin(Xtrain)
# ytrain = fx
# # eps    = np.random.normal(0, 1e-3, ndata)
# # ytrain = fx + eps.reshape(ndata,1)

# # --- define test data --- #
# n_test      = 30
# Xtest       = np.linspace(-7.0, 5.0, num=n_test)
# fx_test     = np.sin(Xtest)
# Ytest_mean  = np.zeros(n_test)
# Ytest_std   = np.zeros(n_test)


# ##### --- Test for Bayesian Optimization ---#####
# # --- build a GP model --- #
# GP_m = BayesianOpt(Xtrain, ytrain, 'RBF', multi_hyper=2, var_out=True)

# # --- build Bayesian Optimization --- #
# rng = np.random.default_rng()
# x0 = rng.choice(Xtrain) # random choice from the train data
# for i in range(2):
#     x_new = GP_m.optimize_acquisition(x0,b=3)
#     y_new = np.sin(x_new)
#     GP_m.add_sample(x_new,y_new)

#     # For next iteration
#     x0 = x_new
#     print(f"optimal input: {x_new}")

# # --- use GP to predict test data --- #
# for ii in range(n_test):
#     m_ii, std_ii   = GP_m.GP_inference_np(Xtest[ii])
#     Ytest_mean[ii] = m_ii 
#     Ytest_std[ii]  = std_ii

# ####### --- Plotting --- #######
# # plot observed points
# plt.plot(GP_m.X, GP_m.Y, 'kx', mew=2)

# # plot the samples of posteriors
# plt.plot(Xtest, fx_test, 'black', linewidth=1)

# # plot GP confidence intervals (+- 3 * standard deviation)
# plt.gca().fill_between(Xtest.flat, 
#                        Ytest_mean - 3*np.sqrt(Ytest_std), 
#                        Ytest_mean + 3*np.sqrt(Ytest_std), 
#                        color='C0', alpha=0.2)

# # plot GP mean
# plt.plot(Xtest, Ytest_mean, 'C0', lw=2)

# plt.axis([-7, 5, -2, 2])
# plt.title('Gaussian Process Regression')
# plt.legend(('training', 'true function', 'GP mean', 'GP conf interval'),
#            loc='lower right')
# plt.show()