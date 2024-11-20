import casadi as ca
import numpy as np

class WilliamOttoReactorSteadyState:
    def __init__(self):
        pass

    def ode_callback(self, w, x):
        # Unpack the state variables
        xa = w[0]
        xb = w[1]
        xc = w[2]
        xp = w[3]
        xe = w[4]
        xg = w[5]

        Fb = x[0]
        Tr = x[1]
        Fa = 1.8275  # Mass flow rate Reactant A
        Fr = Fa + Fb
        Vr = 2105.2  # Reactor volume

        # Reaction constants
        k1_base, k2_base, k3_base = 1.6599e6, 7.2177e8, 2.6745e12
        eta1, eta2, eta3 = 6666.7, 8333.3, 11111

        # Temperature-dependent rate constants
        k1 = k1_base * ca.exp(-eta1 / (Tr + 273))
        k2 = k2_base * ca.exp(-eta2 / (Tr + 273))
        k3 = k3_base * ca.exp(-eta3 / (Tr + 273))

        # Differential equations (treated as algebraic here)
        dxa = (Fa - Fr * xa - Vr * xa * xb * k1) / Vr
        dxb = (Fb - Fr * xb - Vr * xa * xb * k1 - Vr * xb * xc * k2) / Vr
        dxc = -(Fr * xc) / Vr + 2 * xa * xb * k1 - 2 * xb * xc * k2 - xc * xp * k3
        dxp = -(Fr * xp) / Vr + xb * xc * k2 - 0.5 * xp * xc * k3
        dxe = -(Fr * xe) / Vr + 2 * xb * xc * k2
        dxg = -(Fr * xg) / Vr + 1.5 * xp * xc * k3

        return ca.vertcat(dxa, dxb, dxc, dxp, dxe, dxg)

    def solve_steady_state(self, Fb, Tr, initial_guess):
        # Define state variables and inputs
        w = ca.MX.sym("w", 6)  # State variables [xa, xb, xc, xp, xe, xg]
        x = ca.MX.sym("x", 2)  # Inputs [Fb, Tr]

        # Define ODE system
        ode = self.ode_callback(w, x)

        # Create the rootfinder to solve steady state
        rootfinder = ca.rootfinder("steady_state_solver", "newton", {"x": w, "p": x, "g": ode})

        # Solve the system
        steady_state = rootfinder(initial_guess, [Fb, Tr])
        return steady_state.full().flatten()

# Example usage
if __name__ == "__main__":
    reactor = WilliamOttoReactorSteadyState()
    
    # Inputs
    Fb = 5.0  # Mass flow rate of reactant B
    Tr = 85.0  # Reactor temperature in Kelvin
    initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Initial guess for states

    # Solve for steady state
    steady_state = reactor.solve_steady_state(Fb, Tr, initial_guess)

    # Print the results
    print("Steady-state solution:")
    print(f"Xa: {steady_state[0]:.4f}, Xb: {steady_state[1]:.4f}, Xc: {steady_state[2]:.4f}, "
          f"Xp: {steady_state[3]:.4f}, Xe: {steady_state[4]:.4f}, Xg: {steady_state[5]:.4f}")
