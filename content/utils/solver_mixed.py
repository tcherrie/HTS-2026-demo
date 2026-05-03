import ngsolve as ngs
from numpy.linalg import norm
from numpy import isnan
import numpy as np
from ngsolve.webgui import Draw
from time import time
import multiprocessing

def newton_mixed(fes : ngs.FESpace,                                                 # finite element space
          residual : callable,                                                      # residual(state, test)
          residual_derivative : callable = None,                                    # residual_derivative(state, trial, test) (optional)
          initial_guess :  ngs.GridFunction |  ngs.CoefficientFunction = ngs.CF(0), # initial guess
          # Inspection parameters
          verbosity : int = 1,                                                      # verbosity level (0 - silent to 3 - detailed)
          draw : bool = False,                                                      # draw intermediate solutions
          # Newton parameters
          maxit_newton : int = 50,              # maximum number of Newton outer iterations
          tol_dec : float = 1e-6,               # (absolute) tolerance on Newton decrement : sqrt( < residual(uOld), du > )
          tol_res : float = 1e-3,               # (absolute) tolerance on residual 
          rtol_res : float = 1e-6,              # relative tolerance on the residual between 2 iterations (to save 1 useless iteration in case of linear problem)
          # Line search parameters
          linesearch : bool = True,             # flag to enable line search (recommended)
          maxit_linesearch : int = 50,          # maximum iteration number within the line search
          minstep_linesearch : float = 1e-12,   # minimum step size allowed in the line search 
          armijo_linesearch : float = 0.1,      # Armijo coefficient in [0, 1) such that |residual(u-step*du)|² < residual²(u) - armijo_linesearch*step*(|residual(u)|²)'(du)
          step_factor_linesearch : float = 0.5, # step size reduction factor in (0, 1) to reduce the step if too big
          use_multithreading : bool = False     # flag to enable parallelization with TaskManager during assembly
          ) -> dict:

    """
    Solve a nonlinear PDE using Newton method 

    Parameters
    ----------
    fes : ngs.FESpace
        The mixed finite element space.

    residual : callable
        Function taking (state, test function) and returning the residual form.

    residual_derivative : callable, optional
        Function taking (state, trial function, test function) and returning
        the bilinear form of the derivative. If None, symbolic differentiation is used.

    initial_guess : ngs.GridFunction or ngs.CoefficientFunction, optional
        Initial solution guess. Default is 0 everywhere.

    verbosity : int, optional
        Verbosity level (0 = silent, 3 = very detailed). Default is 1.

    draw : bool, optional
        Whether to visualize intermediate results. Default is False.

    maxit_newton : int, optional
        Maximum number of Newton iterations. Default is 50.

    tol : float, optional
        Absolute convergence tolerance on the Newton decrement. Default is 1e-8.

    rtol_res : float, optional
        relative tolerance on the residual between 2 iterations (to save 1 useless 
        iteration in case of linear problem). Default is 1e-10.

    linesearch : bool, optional
        Enable or disable line search. Default is True.

    maxit_linesearch : int, optional
        Maximum number of line search iterations. Default is 20.

    minstep_linesearch : float, optional
        Minimum allowable step size during line search. Default is 1e-12.

    armijo_linesearch : float, optional
        Armijo condition coefficient for line search. Default is 0.1.

    step_factor_linesearch : float, optional
        Multiplicative factor to reduce step size in line search. Default is 0.3.

    use_multithreading : bool, optional
        Enable or disable parallelization with TaskManager during assembly. Default is Flase.

    Returns
    -------
    results : dict
        A dictionary containing:
        - "solution" : final solution (ngs.GridFunction)
        - "status" : integer code indicating termination reason (see below)
        - "linear_detected" : True if linear problem detected early
        - "iteration" : number of Newton iterations performed
        - "last_inverse" : last tangent matrix decomposition (for reuse or debugging)
        - "residual" : list of residual norms per iteration
        - "decrement" : list of Newton decrement values per iteration
        - "wall_time" : total computation time in seconds

    Status codes:
    -------------
    0 : ✅ SUCCESS — Newton converged successfully.
    1 : ❌ FAILURE — Maximum number of Newton iterations reached.
    2 : ❌ FAILURE — Line search failed: minimum step size reached.
    3 : ❌ FAILURE — Line search failed: max number of iterations reached.
    4 : ❌ FAILURE — NaN encountered in the residual (after line search if enabled).
    """

    # I) Initialization

    tStart = time()
    if verbosity >= 2 : print(f"-------------------- START NEWTON ---------------------")
    if verbosity >= 3 : print(f"Initializing  ..... ", end = "")
    
    # Configuration of the multithreading
    if use_multithreading:
        try:
            ngs.SetNumThreads(multiprocessing.cpu_count())
        except:
            pass

    # IMPORTANT: For mixed spaces, TnT() returns Tuples, we must unpack them properly.
    trial_funcs, test_funcs = fes.TnT()
    
    def compute_residual_norm(sol):
        """Calculate |r(sol)| with the use of multithreading"""
        if use_multithreading:
            with ngs.TaskManager():
                res_vec = ngs.LinearForm(residual(sol, test_funcs)).Assemble().vec
        else:
            res_vec = ngs.LinearForm(residual(sol, test_funcs)).Assemble().vec
        
        # Extract the free degrees of freedom
        freedofs = fes.FreeDofs()
        vec_np = res_vec.FV().NumPy()
        free_indices = np.where(freedofs)[0]
        if len(free_indices) > 0:
            return norm(vec_np[free_indices])
        return 0.0
    
    # Function to compute res²
    res2 = lambda sol : compute_residual_norm(sol)**2
    
    # GridFunctions
    state = ngs.GridFunction(fes)
    state_linesearch = ngs.GridFunction(fes)
    descent = ngs.GridFunction(fes)
    
    # Set initial guess
    if type(initial_guess) is ngs.GridFunction and initial_guess.space == fes:
        state.vec.data = initial_guess.vec.data
    else:
        state.Set(initial_guess)
    
    counter_newton = 0
    decrement_list = []
    res2_state = res2(state)
    residual_list = [np.sqrt(res2_state) if res2_state > 0 else 0.0]
    status = 0
    linear = False

    if draw : 
        try:
            scene = Draw(state)
        except:
            if verbosity >= 1 : print("Warning: Could not draw solution")
    if verbosity >= 3 : print(f"done ({(time()-tStart) * 1000 :.2f} ms).")
    if verbosity >= 2 : print(f"Initial residual : {residual_list[-1] :.5e}")
    if verbosity >= 3 : print(f"Start loop  ....... ")

    # Variable to store Kinv
    Kinv = None
    
    # II) Newton loop
    while True:
        counter_newton += 1
        if verbosity >= 2 : print(f" It {counter_newton} -------------------------------------------------")

        # a) Assembly
        tStartAssembly = time()
        if verbosity >= 3 : print(f" - Assembly ....... ", end = "")
        
        # Assembling the residue with TaskManager if requested
        if use_multithreading:
            with ngs.TaskManager():
                # Residue assembly
                res = ngs.LinearForm(fes)
                # Use test_funcs, which is a tuple for mixed spaces.
                res += residual(state, test_funcs)
                res.Assemble()
                
                # Derivative assembly
                if residual_derivative is None:
                    # Différentiation symbolique
                    dres = ngs.BilinearForm(fes)
                    # trial_funcs is a tuple for mixed spaces
                    dres += residual(trial_funcs, test_funcs)
                    dres.AssembleLinearization(state.vec)
                else:
                    # User-supplied derivative
                    dres = ngs.BilinearForm(fes)
                    dres += residual_derivative(state, trial_funcs, test_funcs)
                    dres.Assemble()
        else:
            res = ngs.LinearForm(fes)
            res += residual(state, test_funcs)
            res.Assemble()
            
            if residual_derivative is None:
                dres = ngs.BilinearForm(fes)
                dres += residual(trial_funcs, test_funcs)
                dres.AssembleLinearization(state.vec)
            else:
                dres = ngs.BilinearForm(fes)
                dres += residual_derivative(state, trial_funcs, test_funcs)
                dres.Assemble()


        if verbosity >= 3 : print(f"done ({(time()-tStartAssembly) * 1000 :.2f} ms).")
        
        # b) Solve
        tStartSolve = time()
        if verbosity >= 3 : print(f" - Solve .......... ", end = "")
        
        freedofs = fes.FreeDofs()
        Kinv = dres.mat.Inverse(freedofs=freedofs, inverse="pardiso")
        descent.vec.data = Kinv * res.vec
        
        if verbosity >= 3 : print(f"done ({(time()-tStartSolve) * 1000 :.2f} ms).")

        # c) Calculation of Newton's decree
        decrement = np.sqrt(abs(ngs.InnerProduct(res.vec, descent.vec)))
        decrement_list.append(decrement)
        
        # d) Line search
        if linesearch:
            tStartLineSearch = time()
            if verbosity >= 2 : print(f" - Line search .... ")
            step = 1.0
            counter_linesearch = 0
            
            state_linesearch.vec.data = state.vec - step * descent.vec
            res2_ls = res2(state_linesearch)
            if verbosity >= 2 : print(f"   it {counter_linesearch} : ||residual|| = {np.sqrt(res2_ls) :.5e} | step = {step :.2e}")

            while not (res2_ls < (1 - 2 * armijo_linesearch * step) * res2_state):
                step *= step_factor_linesearch
                state_linesearch.vec.data = state.vec - step * descent.vec
                res2_ls = res2(state_linesearch)
                counter_linesearch += 1
                
                if verbosity >= 2 : print(f"   it {counter_linesearch} : ||residual|| = {np.sqrt(res2_ls) :.5e} | step = {step :.2e}")
                
                if counter_linesearch >= maxit_linesearch:
                    if verbosity >= 1 : print(f"❌ FAILURE: maximal number of line search iterations reached !!")
                    status = 3
                    break
                
                if step < minstep_linesearch:
                    if verbosity >= 1 : print(f"❌ FAILURE: minimal line search step reached !!")
                    status = 2
                    break
            
            if verbosity >= 3 : print(f" - Line search done ({(time()-tStartLineSearch) * 1000 :.2f} ms).")

            if not status: 
                state.vec.data = state_linesearch.vec
        else:
            state.vec.data = state.vec - descent.vec

        # e) NaN verification
        if isnan(res2_state):
            status = 4
            if verbosity >= 1 : 
                print(f"❌ FAILURE: NaN detected ", end = "")
                if linesearch : print("after line search ", end = "")
            print("!!")
            break

        if status: 
            break

        # f) Calculation of new residue
        res2_state = res2(state)
        residual_norm = np.sqrt(res2_state) if res2_state > 0 else 0.0
        residual_list.append(residual_norm)
        
        if verbosity >= 2 : print(f" - Conv : ||residual|| = {residual_list[-1]:.5e} | decr = {decrement_list[-1] :.5e}")
        
        if draw and 'scene' in locals():
            try:
                scene.Redraw()
            except:
                pass
        
        if verbosity >= 3 : print(f" - Newton iteration done ({(time()-tStartAssembly) * 1000 :.2f} ms).")

        # g) Critères d'arrêt
        if len(residual_list) > 1 and residual_list[-2] > 0:
            if residual_list[-1] / residual_list[-2] < rtol_res:
                if verbosity >= 2 : print(f"Stop because linear problem detected.")
                linear = True
                break
        
        if decrement_list[-1] < tol_dec : 
            if verbosity >= 2 : print(f"Stop because decrement is lower than tol_dec.")
            break

        if residual_list[-1] < tol_res : 
            if verbosity >= 2 : print(f"Stop because residual is lower than tol_res.")
            break

        if counter_newton >= maxit_newton: 
            if verbosity >= 1 : print(f"❌ FAILURE: maximum number of Newton iterations reached !!")
            status = 1
            break
    
    # III) Export results
    if verbosity >=2 and not status : 
        print(f"-------------------------------------------------------")  
        print(f" ✅ SUCCESS: Newton has converged in {counter_newton} iteration", end = "")
        if  counter_newton > 1 : print("s.")
        else : print(".") 
    if verbosity >=2 :  print(f" Total wall time: {(time() - tStart) :.2f} s.")
    
    results = {"solution" : state, 
               "status" : status, 
               "linear_detected" : linear,
               "iteration": counter_newton, 
               "last_inverse" : Kinv, 
               "residual" : residual_list,
               "decrement": decrement_list,
               "wall_time" : time() - tStart}
    
    if verbosity >=2 : print(f" --------------------- END NEWTON --------------------- ")  
    return results