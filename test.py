#######################
# --- TR function --- #
#######################

def Adjust_TR(self, Delta0, xk, xnew, GP_obj):
    '''
    Adjusts the TR depending on the rho ratio between xk and xnew
    '''
    Delta_max, eta0, eta1          = self.Delta_max, self.eta0, self.eta1
    gamma_red, gamma_incr          = self.gamma_red, self.gamma_incr
    obj_system                     = self.obj_system

    # --- compute rho --- #
    plant_i     = obj_system(np.array(xk).flatten())
    plant_iplus = obj_system(np.array(xnew).flatten())
    rho         = (plant_i - plant_iplus)/(GP_obj.GP_inference_np(np.array(xk).flatten())[0]-
                                            GP_obj.GP_inference_np(np.array(xnew).flatten())[0] )

    # --- Update TR --- #
    if plant_iplus<plant_i:
        if rho>=eta0:
            if rho>=eta1:
                Delta0 = min(Delta0*gamma_incr, Delta_max)
            elif rho<eta1:
                Delta0 = Delta0*gamma_red
            # Note: xk = xnew this is done later in the code
        if rho<eta0:
            #print('rho<eta0 -- backtracking')
            self.suboptimal_list.append(copy.deepcopy(xk))
            xnew   = xk
            Delta0 = Delta0*gamma_red
    else:
        self.suboptimal_list.append(copy.deepcopy(xk))
        xnew   = xk
        Delta0 = Delta0*gamma_red
        #print('plant_iplus<plant_i -- backtracking')

    return Delta0, xnew, xk

