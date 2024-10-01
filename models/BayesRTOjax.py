import jax 
import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.optimize import minimize
from models.GP_Classic import GP

class BayesianOpt(GP):
    def __init__(self,plant_system):
        GP.__init__(self,plant_system)

    #######################################################
                # --- Optimize Acquisition --- #          
    #######################################################

    def minimize_acquisition(self,r,x_0,data_storage,b=0,multi_start=5):
        '''
        Description:
            Find minimizer x* in a Gaussian Process Lower Confidence Bound
        Argument:
            - r                     : radius of trust region area
            - x_0                   : previous input observed
            - data_storage          : class for storing data
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
        self.GP_inference_jit       = jit(self.GP_inference)
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
                if con['fun'](res.x) < -0.01: # Barrier when minimize significantly fails to satisfy any constraints
                    passed = False
                    break 
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
        GP_inference                = self.GP_inference_jit(x)
        mean                        = GP_inference[0][0]
        std                         = jnp.sqrt(GP_inference[1][0])
        value                       = mean - b*std

        return value

    def constraint(self, x, b, index):
        GP_inference                = self.GP_inference_jit(x)
        mean                        = GP_inference[0][index]
        std                         = jnp.sqrt(GP_inference[1][index])
        value                       = mean - b*std
        # value                       = mean

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
        keys                        = ['i','x_initial','x_new','plant_output',"GP_cons","GP_cons_safe",
                                       'TR_radius','plant_temporary']
        data_storage                = DataStorage(keys)

        # Retrieve radius from Trust Region Parameters
        radius = TR_parameters['radius']

        # Collect data at x_initial to DataStorage
        plant_output                = self.calculate_plant_outputs(x_initial)
        GP_cons,GP_cons_safe        = self.calculate_GP_cons(x_initial,b)
        data_dict                   = self.create_data_points(0,x_initial,x_initial,plant_output,GP_cons,GP_cons_safe,radius)
        data_storage.add_data_points(data_dict)
        
        # create temporary plant output for comparison in Trust Region Update
        data_storage.data['plant_temporary'].append(plant_output.tolist())

        # Real-Time Optimization
        for i in range(n_iter):

            # Bayesian Optimization
            d_new, obj              = self.minimize_acquisition(radius,x_initial,data_storage,multi_start=multi_start,b=b)

            # Retrieve Data from plant system
            plant_output            = self.calculate_plant_outputs(x_initial+d_new)
            GP_cons,GP_cons_safe        = self.calculate_GP_cons(x_initial,b)
            
            # Collect Data to DataStorage
            data_dict               = self.create_data_points(i+1,x_initial,x_initial+d_new,plant_output,GP_cons,GP_cons_safe,radius)
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
    
    def create_data_points(self,iter,x_initial,x_new,plant_output,GP_cons,GP_cons_safe,radius):
        data_dict = {
            'i'                     : iter,
            'x_initial'             : x_initial.tolist(),
            'x_new'                 : x_new.tolist(),
            'plant_output'          : plant_output.tolist(),
            'GP_cons'               : GP_cons.tolist(),
            'GP_cons_safe'          : GP_cons_safe.tolist(),
            'TR_radius'             : radius
        }

        return data_dict
    
    def calculate_plant_outputs(self,x):

        plant_output            = []
        for plant in self.plant_system:
            plant_output.append(plant(x)) 

        return jnp.array(plant_output)
    
    def calculate_GP_cons(self,x,b):
        cons = []
        cons_safe = []
        for i in range(self.n_fun):

            if i != 0:
                GP_inference = self.GP_inference(x)
                mean                = GP_inference[0][i]
                std                 = jnp.sqrt(GP_inference[1][i])

                cons.append(mean)
                cons_safe.append(mean-b*std)
            else:
                pass

        return jnp.array(cons), jnp.array(cons_safe)
    
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
        GP_previous     = self.GP_inference_jit(x_initial)[0][0]
        GP_now          = self.GP_inference_jit(x_new)[0][0]

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
    


