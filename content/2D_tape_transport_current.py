#%% 1) Geometry and mesh generation ##########################################

from utils.geometry import mesh_tape_comsol

mesh = mesh_tape_comsol()
print(f"Imported 2D ribbon mesh with {mesh.ne} triangles, {mesh.nedge} edges and {mesh.nv} nodes.")

## Save the mesh as .vtu, to be viewed by Paraview.
from ngsolve import VTKOutput

'''
vtk = VTKOutput(
    ma = mesh,
    coefs = [],                 
    names = [],
    filename = "mesh2D_only",
    subdivision = 0
)
vtk.Do()
'''

#%% 2) Physical parameters ######################################

from ngsolve import pi

## Default physical parameters
mu0 = 4e-7 * pi     # permeability of air (H/m)
sigmaair = 1        # conductivity of air approximation (S/m)
epsJ = 2.2e-16      # tolerence to avoid the evaluation of zero at a negative power, used in the norm of current density ||J|| = sqrt(Jx^2+Jy^2+Jz^2+epsJ)

## Definition of the source field
I0 = 22.4           # Amplitude of the transport current (A)
freq = 50           # frequency (Hz)
theta = pi/2        # angle of the source field (rad)

## HTS physical parameters of the power law
n = 101             # n material parameter
Ec = 1e-4           # local critical current criterion (V/m)
Jc = 2.8e10         # critical current density (A/m²)

## HTS resistivity model
def rho_hts(normcurlH, Ec=Ec, Jc=Jc, n=n):
    """
    Resistivity of the superconductor defined by the power law : 
    rho(||curlH||) = (Ec/Jc)*(||curlH||/Jc)^(n-1)
    """
    return (Ec/Jc)*(normcurlH/Jc)**(n-1)

def drho_hts_dnormcurlH(normcurlH):
    """
    Returns d rho(||curlH||) / d ||curlH||
    """
    return (Ec/(Jc)**2)*(n-1)*(normcurlH/Jc)**(n-2)

def drho_hts_dnormcurlH_over_normcurlH(normcurlH):
    """
    Returns (1/||curlH||) * [d rho(||curlH||) / d ||curlH||]
    """
    return (Ec/(Jc)**3)*(n-1)*(normcurlH/Jc)**(n-3)

## HTS cross-section (for normalization)
from ngsolve import Integrate, dx
from ngsolve import CoefficientFunction as CF
cross_section_hts = Integrate(CF(1) * dx("hts"), mesh)

#%% 3) Time loop setting  ######################################

T_final = 1/freq    # final time = one period 
T_half = T_final/2  # half of the final time

dt_basic = 0.2*T_final / 100
dt =  dt_basic / 2              # first time step
dt_min = dt_basic / 10000       # minimum time step
dt_max = dt_basic               # maximum time step

#%% 4) H-formulation ##########################################

from ngsolve import curl, sqrt

def residual_mixed(sol, test_funcs):
    """
    Residual of H-formulation with imposed transport current
    
    :param sol: Trial function
    :param test_funcs: Test function
    """
    v, mu = test_funcs[0], test_funcs[1]
            
    if hasattr(sol, 'components'):
        H, lam = sol.components[0], sol.components[1]
    else:
        H, lam = sol[0], sol[1]
            
    curls_H = curl(H)
    curls_v = curl(v)
    normCurlH = sqrt(curls_H*curls_H + epsJ)
            
    # 1. Physical terms
    res =  mu0 * (1/dt_try) * H * v * dx
    res += -mu0 * (1/dt_try) * sol_prev.components[0] * v * dx
    res += rho_hts(normCurlH) * curls_H * curls_v * dx("hts")
    res += (1/sigmaair) * curls_H * curls_v * dx("air")
            
    # 2. & 3. Lagrange terms FOR THE GLOBAL CONSTRAINT
    # Standard form for a constraint ∫ curl(H) dx = I_target
            
    # Bilinear term : λ * ∫ curl(v) dx
    res += lam * curls_v * dx("hts")
            
    # Bilinear term : μ * ∫ curl(H) dx
    res += mu * curls_H * dx("hts")
            
    # Linear term : - μ * I_target
    # But μ is a scalar, so we write : ∫ μ * (I_target/cross_section_hts) dx
    res += - (I_target/cross_section_hts) * mu * dx("hts")
            
    return res

## Derivative of the residual function   
def d_residual_mixed(sol_k, trial_funcs, test_funcs):
    """
    Directional derivative of the residual of constrained H-formulation w.r.t. Htilde
    
    :param sol_k: Evaluation point
    :param trial_funcs: Trial function
    :param test_funcs: Test function
    """
    dH, dlam = trial_funcs[0], trial_funcs[1]
    v, mu = test_funcs[0], test_funcs[1]
            
    if hasattr(sol_k, 'components'):
        H_k, _ = sol_k.components[0], sol_k.components[1]
    else:
        H_k, _ = sol_k[0], sol_k[1]
            
    curls_H_k = curl(H_k)
    curls_dH = curl(dH)
    curls_v = curl(v)
    normCurlH_k = sqrt(curls_H_k*curls_H_k + epsJ)
            
    # Dérivée des termes physiques
    dres = mu0 * (1/dt_try) * dH * v * dx
    dres += (1/sigmaair) * curls_dH * curls_v * dx("air")
    dres += rho_hts(normCurlH_k) * curls_dH * curls_v * dx("hts")
    dres += drho_hts_dnormcurlH_over_normcurlH(normCurlH_k) * (curls_H_k*curls_dH)*(curls_H_k*curls_v) * dx("hts")
            
    # Dérivée des termes de Lagrange
    dres += dlam * curls_v * dx("hts")
    dres += mu * curls_dH * dx("hts")
            
    return dres

#%% 5) FEM resolution ##########################################

### a) Initialisation FEM and time

## Function space
from ngsolve import HCurl, NumberSpace
fesH = HCurl(mesh, order=1, dirichlet="out")  # Functional space for H, order and boundary conditions
fesLambda = NumberSpace(mesh)                 # NumberSpace : Scalar lagrange multiplier λ space 
fes_mix = fesH * fesLambda                    # mixed-use space

## Initial fields at t=0
from ngsolve import GridFunction
sol_prev = GridFunction(fes_mix)
sol_prev.components[0].vec[:] = 0  # H initial = 0
sol_prev.components[1].vec[:] = 0  # λ initial = 0

### b) Time loop

time_list = [0.0]        # list to store time instances
ac_losses_list = [0.0]   # list to store instantanuous AC losses 
steplist = []            # list to store time steps
Ed = 0.0                 # initial mean AC losses 

t_current = 0.0          # initial time
step = 0                 # time step counter 

import time 
from utils.solver_mixed import newton_mixed
from ngsolve import Integrate
from ngsolve import sin

start = time.perf_counter()

while t_current < T_final:

    converged = False
    dt_try = dt
    
    ## If we want to force to pass at the moments T_final/2 and T_final
    if t_current < T_half < t_current + dt_try:
        dt_try = T_half - t_current
        print(f" Adjustment to reach T_final/2 : dt_try = {dt_try:.6e}")
    elif t_current + dt_try > T_final:
        dt_try = T_final - t_current
        if dt_try < 1e-8:  # If the remaining step is too small
            print(f" Last time step is negligible ({dt_try:.2e}s), simulation completed at t={t_current:.6e}s")
            break  # We stop here, without taking that last microscopic step.
        print(f" Last step to reach T_final: dt_try = {dt_try:.6e}")

    while not converged:
        t_next = t_current + dt_try
        print(f"\n=== Step {step} ===")
        print(f"Try: t={t_next:.6e}, dt={dt_try:.6e}")
        
        ### Imposed transport current at the current moment
        I_target = I0 * sin(2*pi*freq*t_next)
        
        ### Initialisation solution for Newton-Raphson
        sol_init = GridFunction(fes_mix)
        sol_init.components[0].Set(sol_prev.components[0])
        sol_init.components[1].Set(sol_prev.components[1])
        
        ### Nonlinear resolution with Newton-Raphson
        result = newton_mixed(fes_mix, residual_mixed, d_residual_mixed, sol_init, verbosity=1, use_multithreading = False)
        
        ### Convergence check
        if result["status"] == 0:
            converged = True
        else:
            dt_try = max(dt_try * 0.7, dt_min) # time step reduction
            print(f"  🔻 Newton failed, réduction dt → {dt_try:.6e}")
            if dt_try <= dt_min:
                raise RuntimeError("Impossible to converge even with minimal dt !")
            continue
    
    ## Post-processing for this time step
    H_sol = result["solution"].components[0]      # Magnetic field solution
    lambda_sol = result["solution"].components[1] # Lagrange multiplier solution
    J_hts = curl(H_sol)                           # Induced currents

    ### Instantaneous losses in the superconductor
    ac_losses = Integrate(J_hts * rho_hts(sqrt(J_hts*J_hts + epsJ)) * J_hts * dx("hts"), mesh)

    ### Integration of instantaneous losses per half cycle [T_half, T_final] : explicit Euler
    if t_next > T_half:
        if t_current < T_half:
            Ed += 2 * freq * ac_losses * (t_next - T_half)
        else:
            Ed += 2 * freq * ac_losses * (t_next - t_current)
        
    ### Store the results
    ac_losses_list.append(ac_losses)
    time_list.append(t_next)
    steplist.append(dt_try)
    
    ### Export VTK if necessary (example for t=0.015)
    '''
    if abs(t_next - 0.015) < 1e-3:
        vtk = VTKOutput(ma=mesh,
                        coefs=[J_hts / Jc],
                        names=["J_hts_0015"],
                        filename="J_hts_0015")
        vtk.Do()

        vtk = VTKOutput(ma=mesh,
                        coefs=[H_sol],
                        names=["H_sol_0015"],
                        filename="H_sol_0015")
        vtk.Do()
    '''
    print(f"  ✔ Converged | Instant: {t_next:.6f} | Losses: {ac_losses:.6e}")

    ### Update for the next step
    sol_prev.components[0].vec.data = result["solution"].components[0].vec
    sol_prev.components[1].vec.data = result["solution"].components[1].vec
    t_current = t_next
    step += 1
    
    ### Adaptive adjustment: increase dt if everything is going well
    dt = min(dt_try * 2, dt_max)

end = time.perf_counter()

#%% 6) Final results ##########################################

print(f"\n{'='*50}")
print("Final results")
print(f"{'='*50}")
print(f"Total simulation time : {end-start:.3f} s")
print(f"Average losses Ed (half cycle) : {Ed:.6e} W/m")

## Data plot

from utils.trace import plot_result
plot_result(time_list, ac_losses_list, filename_comsol = "results_COMSOL/2D/transport_current/AC_Losses_"+str(I0)+"A.txt")

## Data save

import numpy as np
np.save("Losses_2D_tape_transport_current_"+str(I0)+"A.npy", np.array(ac_losses_list))  # Instantanuous AC losses 
np.save("Time_tape_transport_current_"+str(I0)+"A.npy", np.array(time_list))         # Time instances
np.save("steplist_tape_transport_current_"+str(I0)+"A.npy", np.array(steplist))      # The time steps used



# %%
