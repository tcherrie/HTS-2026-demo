#%% 1) Geometry and mesh generation ##########################################

from utils.geometry import mesh_bulk_comsol

mesh = mesh_bulk_comsol()
print(f"Imported 3D bulk mesh with {mesh.ne} triangles, {mesh.nedge} edges and {mesh.nv} nodes.")

## uncomment to save the mesh as .vtu, to be viewed by Paraview.
from ngsolve import VTKOutput
'''
vtk = VTKOutput(
    ma = mesh,
    coefs = [],                 
    names = [],
    filename = "mesh_only",
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
Bmax = 5*1e-3       # amplitude (T)
freq = 50           # frequency (Hz)
theta = pi/2        # angle of the source field (rad)

from ngsolve import CoefficientFunction as CF
from ngsolve import cos, sin
H_s_dir = Bmax / mu0 * CF( (0, 0, 1))

## HTS physical parameters of the power law
n = 25             # n material parameter
Ec = 1e-4          # local critical current criterion (V/m)
Jc = 2.5*1e6       # critical current density (A/m²)

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

#%% 3) Time loop setting  ######################################

T_final = 1/freq    # final time = one period 
T_half = T_final/2  # half of the final time

dt_basic = (T_final / 100) / 3
dt =  dt_basic / 2              # first time step
dt_min = dt_basic / 10000       # minimum time step
dt_max = dt_basic               # maximum time step

#%% 4) H-formulation ##########################################

from ngsolve import curl, HCurl, sqrt, dx

def residual(H_tilde, v) :
    """
    Residual of H-formulation
    :param H_tilde: Trial function
    :param v: Test function
    """
    curls_H_tilde = curl(H_tilde)
    normcurlH = sqrt(curls_H_tilde*curls_H_tilde + epsJ)
    curls_v = curl(v)

    res = mu0 * (1/dt_try) * H_tilde* v * dx
    res += -mu0 * (1/dt_try) * H_tilde_prev * v * dx
    res += rho_hts(normcurlH) * (curls_H_tilde) * (curls_v) * dx("hts")
    res += (1/sigmaair) * (curls_H_tilde) * (curls_v) * dx("air")
    res += mu0 * derHs * v * dx
    return res

## Derivative of the residual function   
def d_residual(H_tilde_k, dH_tilde, v):
    """
    Directional derivative of the residual of H-formulation w.r.t. Htilde
    
    :param H_tilde_k: Evaluation point
    :param dH_tilde: Trial function
    :param v: Test function
    """
    curls_tilde_k = curl(H_tilde_k)
    curls_dH_tilde = curl(dH_tilde)
    curls_v = curl(v)

    normcurlH_tilde_k = sqrt(curls_tilde_k*curls_tilde_k + epsJ)

    dres = mu0 * (1/dt_try) * dH_tilde * v * dx
    dres += (1/sigmaair) * (curls_dH_tilde) * (curls_v) * dx("air")
    dres += rho_hts(normcurlH_tilde_k) * (curls_dH_tilde) * (curls_v) * dx("hts")    
    dres += drho_hts_dnormcurlH_over_normcurlH(normcurlH_tilde_k) * ( curls_tilde_k * curls_dH_tilde ) * ( curls_tilde_k * curls_v ) * dx("hts")    
    
    return dres

#%% 5) FEM resolution ##########################################

### a) Initialisation FEM and time

## Function space
fes = HCurl(mesh, order=0, dirichlet="out")  # Functional space, order and boundary conditions
dH_tilde = fes.TrialFunction()               # Unknown function (dofs)
v_test = fes.TestFunction()                  # Test function

## Initial field at t=0
from ngsolve import GridFunction
H_tilde_prev = GridFunction(fes)
H_tilde_prev.vec[:] = 0  

### b) Time loop

time_list = [0.0]           # list to store time instances
ac_losses_list = [0.0]      # list to store instantanuous AC losses 
steplist = []               # list to store time steps
Ed = 0.0                    # initial mean AC losses (W/m)

t_current = 0.0             # initial time
step = 0                    # time step counter

import time 
from utils.solver import newton
from ngsolve import Integrate

start = time.perf_counter()

while t_current < T_final:

    converged = False
    dt_try = dt  

    ## If we want to force to pass exactly at the moments T_final/2 and T_final
    if t_current < T_half < t_current + dt_try:
        dt_try = T_half - t_current
        print(f" Adjustment to reach T_final/2 : dt_try = {dt_try:.6e}")
    elif t_current + dt_try > T_final:
        dt_try = T_final - t_current
        print(f" Adjustment of the last step to reach T_final : dt_try = {dt_try:.6e}")

    while not converged:

        t_next = t_current + dt_try

        print(f"\n=== Step {step} ===")
        print(f"Try: t={t_next:.6e}, dt={dt_try:.6e}")

        ### External field and its derivative at the current moment
        Hs = sin(2*pi*freq*t_next) * H_s_dir
        derHs =  2 * pi * freq * cos(2*pi*freq*t_next) * H_s_dir 

        ### Initialisation solution for Newton-Raphson
        sol = GridFunction(fes)
        sol.Set(H_tilde_prev)

        ### Nonlinear resolution with Newton-Raphson
        result = newton(fes, residual, d_residual, sol, verbosity=1, use_multithreading = True)

        ### Analyse convergence 
        if result["status"] == 0:
            converged = True
        else:
            new_dt_try = max(dt_try * 0.7, dt_min)  # time step reduction
            print(f"🔻 Newton failed, gradual reduction dt → {new_dt_try:.6e}")
            if new_dt_try == dt_min:
                raise RuntimeError("⛔ Impossible to converge even with minimal dt !")
            dt_try = new_dt_try
            continue

    ## Post-processing for this time step
    H_sol = result["solution"]
    H_t = H_sol + Hs            # Global magnetic field
    J_hts = curl(H_sol)         # induced currents

    ### Instantaneous losses in the superconductor
    ac_losses = Integrate( J_hts * rho_hts(sqrt(J_hts*J_hts + epsJ)) * J_hts *mesh.MaterialCF({"hts":1}), mesh)

    ### Integration of instantaneous losses per half cycle [T_half, T_final] : explicit Euler
    if t_next > T_half:
        if t_current < T_half:
            Ed += 2 * freq * ac_losses * (t_next - T_half)
        else:
            delta_t_eff = t_next - t_current     
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
                        coefs=[H_t],
                        names=["H_t_0015"],
                        filename="H_t_0015")
        vtk.Do()
    '''
    print(f"  ✔ Converged | Instant: {t_next:.6f} | Losses: {ac_losses:.6e}")

    ### Update for the next step
    H_tilde_prev.Set(H_sol)
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
plot_result(time_list, ac_losses_list, filename_comsol = "results_COMSOL/3D/AC_Losses_"+str(int(Bmax*1000))+"mT.txt")

## Data save

import numpy as np
np.save("Losses_3D_bulk_"+str(int(Bmax*1000))+".npy", np.array(ac_losses_list))    # Instantanuous AC losses 
np.save("Time_3D_bulk_"+str(int(Bmax*1000))+".npy", np.array(time_list))           # Time instances  
np.save("steplist_3D_bulk_"+str(int(Bmax*1000))+".npy", np.array(steplist))        # The time steps used

