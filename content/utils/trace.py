#%%
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def plot_result(t, v, filename_comsol = None, T_final = 0.02, dt_interp = 2e-4, filesave_name = None):
        

    time_uniform = np.arange(0, T_final + dt_interp, dt_interp)

    ### interpolation of v on the prescribed time steps
    loss_O0_interp = interp1d(t, v, kind='cubic', fill_value="extrapolate")
    v_ordre0_interp = loss_O0_interp(time_uniform)

    # Simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_uniform, v_ordre0_interp, 'b-', label="NGSolve")

    
    if filename_comsol is not None:
        temps_comsol, pertes_comsol = np.loadtxt(filename_comsol, unpack=True, skiprows=6)
        plt.plot(temps_comsol, pertes_comsol, 'r--', label="COMSOL")

    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("AC Losses")
    plt.legend()
    if filesave_name is not None :
        plt.savefig(filesave_name, dpi=150, bbox_inches="tight")
    plt.show()

    # %%
