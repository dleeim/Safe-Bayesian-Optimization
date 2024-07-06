import scr.BayesOpt as BayesOpt
import numpy as np

if __name__ == '__main__':
    ##_________Test for comparing with BayesRTOjax.py:
    print("#########_________Test for comparing with BayesRTOjax.py:")
    # --- define training data --- #
    Xtrain = np.array([[ 1.305728,   -0.69302666],
                       [ 1.3228958,  -1.198658  ],
                       [ 1.4750773,  -0.75442755],
                       [ 0.9418043,  -0.87447894]])
    ytrain = np.array([[1.280307,  3.2114954],
                       [1.6011345, 3.2834308],
                       [1.632175,  3.4147716],
                       [0.8281207, 2.9260488]])
    GP_m = BayesOpt.BayesianOpt(Xtrain, ytrain, 'RBF', multi_hyper=2, var_out=True)

    # --- NLL --- #
    hyper = np.array([[ 0.,  0.,  0., -5.],
                       [ 2.,  -2.,   2.,  -6.5],])
    for i in range(hyper.shape[0]):
        NLL = GP_m.negative_loglikelihood(hyper[i],GP_m.X_norm,GP_m.Y_norm[:,i:i+1])

    # --- GP initialization --- #
    def mean(x):
        return GP_m.GP_inference_np(x)[0][0]

    print(f"GP mean: {GP_m.Y_mean}")

    x_1 = np.array([1.4,-0.8])
    print(f"GP inference: {mean(x_1)}")
    
