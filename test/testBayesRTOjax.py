import jax.numpy as jnp
from jax import grad
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../source')))
import BayesRTOjax
import Benoit_Problem

if __name__ == '__main__':
    
    ##########################################
    ##### Test Case 1: GP_Initialization #####
    ##########################################

    # --- Preparation --- #
    plant_system = [Benoit_Problem.Benoit_System_1,
                    Benoit_Problem.con1_system]
    GP_m = BayesRTOjax.BRTO(plant_system)

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
    X,Y = GP_m.Data_sampling(n_sample,x_0,r)

    print("# --- Data Sampling --- #")
    print(f'X: \n{X}')
    print(f"Y: \n{Y}")

    # --- GP initialization --- #
    GP_m.GP_initialization(X, Y, 'RBF', multi_hyper=2, var_out=True)
    print(f"GP mean: {GP_m.Y_mean}")

    # # --- negative log likelihood --- #
    # hyper = jnp.array([[ 0.,  0.,  0., -5.],
    #                    [ 2.,  -2.,   2.,  -6.5],])

    # for i in range(hyper.shape[0]):
    #     NLL = GP_m.negative_loglikelihood(hyper[i],GP_m.X_norm,GP_m.Y_norm[:,i:i+1])
    # for i in range(hyper.shape[0]):
    #     NLL = GP_m.negative_loglikelihood(hyper[i],GP_m.X_norm,GP_m.Y_norm[:,i:i+1])

    # # --- Test Gaussian Process Model inference --- #
    # x_1 = jnp.array([1.4,-0.8])
    # GP_inference = GP_m.GP_inference_np(x_1)

    # ## Check if plant and model provides similar output using sampled data as input
    # print("\n#___Check if plant and model provides similar output using sampled data as input___#")
    # print(f"test input: {x_1}")
    # print(f"plant obj: {plant_system[0](x_1)}")
    # print(f"model obj: {func_mean(x_1,index=0)}")
    # print(f"plant con: {plant_system[1](x_1)}")
    # print(f"model con: {func_mean(x_1,index=1)}")

    # ## Check if variance is approx 0 at sampled input
    # print("\n#___Check variance at sampled input___#")
    # print(f"variance: {variances(x_1)}")

    # # --- check gradient of GP_inference --- #

    # # check objective function
    # x_2 = jnp.array([1.4,-0.8])
    # delta = 0.0001
    # check_jaxgrad(x_2,delta,func_mean,index=0)

    # # check constraint
    # check_jaxgrad(x_2,delta,func_mean,index=1)

    # #############################################################
    # #### Test Case 2: Optimization of Lower Confidence Bound ####
    # #############################################################

    # --- Optimize Acquisition --- #
    r_i = 0.5
    d_new, obj = GP_m.optimize_acquisition(r_i,x_0,multi_start=1)
    print(f"optimal new input(model): {x_0+d_new}")
    print(f"corresponding new output(model): {obj}")
    print(f"Euclidean norm of d_new(model): {jnp.linalg.norm(d_new)}")




    # # --- Add sample --- #
    # print(f"\n#___Check Add Sample___#")
    # output_new = []
    # for plant in plant_system:
    #     output_new.append(plant(x_0+d_new))
    # print(output_new)
    # output_new = jnp.array(output_new)
    # print(f"add sample: {x_0+d_new, output_new}")
    # GP_m.add_sample(x_0+d_new,output_new)

    # print(f"output after add sample(model): {func_mean(x_0+d_new,0)}")
    # print(f"constraint after add sample(model): {func_mean(x_0+d_new,1)}")
    # print(f"new variances(model): {variances(x_0+d_new)}")
    # print(f"plant output: {plant_system[0](x_0+d_new)}")
    # print(f"plant constraint: {plant_system[1](x_0+d_new)}")

    # x_0 = x_0 + d_new
    # d_new, obj = GP_m.optimize_acquisition(r_i,x_0,multi_start=10)
    # print(f"optimal new input(model): {x_0+d_new}")
    # print(f"corresponding new output(model): {obj}")
    # print(f"Euclidean norm of d_new(model): {jnp.linalg.norm(d_new)}")

    