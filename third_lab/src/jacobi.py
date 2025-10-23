import numpy as np
import warnings
from evaluations import *

warnings.filterwarnings('ignore', category=RuntimeWarning)

def solve_jacobi(A_orig, b_orig, eps=1e-10, max_iter_est=100_000, eval_options=None):
    if eval_options is None:
        eval_options = {
            'cond': True,
            'spectral_radius': True,
            'norms': True,
            'residual': True,
            'benchmark': True,
            'iterations': True,
            'a_priori': True,
            'stability_error': True
        }


    A = A_orig.copy()
    b = b_orig.copy()
    evals = {}
    n = A.shape[0]
    col_perm = list(range(n))
    k_est = np.inf
    scales = np.ones(n)

    print(f"\n======== Starting Jacobi Solver for {n}x{n} System ========")
    print(f"Target Epsilon (eps): {eps}")

    diag_dom = is_strictly_diagonally_dominant(A)
    evals['diagonal_dominance'] = bool(diag_dom)
    print("Initial diagonal dominance:", diag_dom)

    if not diag_dom:
        A_new, b_new, success, col_perm, scales = try_make_diagonally_dominant(A.copy(), b.copy())
        evals['reordered_for_dominance'] = bool(success)
        if success:
            A, b = A_new, b_new
            diag_dom = is_strictly_diagonally_dominant(A)
            evals['diagonal_dominance'] = diag_dom
            print("Using reordered system for Jacobi.")
        else:
            print("WARNING: Matrix is not diagonally dominant and reordering failed. Convergence not guaranteed.")

    diag = np.diag(A).astype(float)
    if np.any(np.isclose(diag, 0.0)):
        raise ValueError("Zero (or near-zero) diagonal element detected; cannot apply Jacobi.")

    C, d = compute_iteration_matrix(A, b)
    print("\nCalculated Iteration Matrix C and Vector d.")

    if eval_options.get('spectral_radius', True):
        rho = get_spectral_radius(C)
        evals['spectral_radius'] = rho

    if eval_options.get('norms', True):
        norms = get_norms(C)
        evals['norms'] = norms

    if eval_options.get('cond', True):
        try:
            cond = float(np.linalg.cond(A_orig))
            evals['cond'] = cond
        except np.linalg.LinAlgError:
            evals['cond'] = None

    if eval_options.get('a_priori', True):
        a_priori_iters = get_a_priori(C, d, eps)
        evals['a_priori_iterations_estimate'] = a_priori_iters
        k_est = a_priori_iters if (a_priori_iters is not None) else max_iter_est

    k_max = int(min(k_est if np.isfinite(k_est) else max_iter_est, max_iter_est))

    x = d.copy()
    converged = False
    k = 0

    print("\n--- Starting Iteration Process ---")
    try:
        for k in range(1, k_max + 1):
            x_new = C @ x + d
            diff_norm = np.linalg.norm(x_new - x, ord=np.inf)

            if diff_norm < eps:
                x = x_new
                converged = True
                break
            x = x_new
        else:
             print(f"Did not converge after max iterations ({k_max}).")

    except RuntimeWarning:
        print("WARNING: Runtime error occurred during iteration.")

    evals['iterations'] = int(k)
    evals['converged'] = bool(converged)
    print("--- Iteration Process Finished ---")

    x_scaled = x * scales
    x_final = np.zeros_like(x)
    x_final[col_perm] = x_scaled

    if eval_options.get('residual', True):
        evals['residual_norm'], evals['relative_residual_norm'] = get_residual(A_orig, x_final, b_orig)

    evals['epsilon'] = float(eps)
    evals['matrix_size'] = int(n)

    return x_final, evals