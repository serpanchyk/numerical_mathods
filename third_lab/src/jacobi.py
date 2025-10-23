import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

def try_make_diagonally_dominant(A, b):
    print("--- Attempting to make matrix A diagonally dominant ---")
    n = A.shape[0]
    permutation = np.zeros(n, dtype=int)
    available_rows = list(range(n))

    for i in range(n):
        pivot_row = -1
        max_val = -1.0
        for row_idx in available_rows:
            if abs(A[row_idx, i]) > max_val:
                max_val = abs(A[row_idx, i])
                pivot_row = row_idx

        if pivot_row != -1:
            permutation[i] = pivot_row
            available_rows.remove(pivot_row)
        else:
            print(f"Could not find a dominant pivot for column {i}. Stop.")
            return A, b, False

    A_permuted = A[permutation, :]
    b_permuted = b[permutation]
    print("Permutation applied:", permutation)

    diag_vals = np.abs(np.diag(A_permuted))
    row_sums = np.sum(np.abs(A_permuted), axis=1) - diag_vals
    is_dominant = np.all(diag_vals > row_sums)
    print("New matrix is diagonally dominant:", is_dominant)
    print("--- End dominance attempt ---")

    return A_permuted, b_permuted, is_dominant


def solve_jacobi(A, b, eps=1e-10, max_iter_est=100_000, eval_options=None):
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

    evals = {}
    n = A.shape[0]

    print(f"\n======== Starting Jacobi Solver for {n}x{n} System ========")
    print(f"Target Epsilon (eps): {eps}")

    initial_diag = np.abs(np.diag(A))
    initial_row_sums = np.sum(np.abs(A), axis=1) - initial_diag
    diag_dom = np.all(initial_diag > initial_row_sums)
    evals['diagonal_dominance'] = bool(diag_dom)
    print("Initial diagonal dominance:", diag_dom)

    if not diag_dom:
        A_new, b_new, success = try_make_diagonally_dominant(A.copy(), b.copy())
        evals['reordered_for_dominance'] = bool(success)
        if success:
            A, b = A_new, b_new
            diag_vals = np.abs(np.diag(A))
            row_sums = np.sum(np.abs(A), axis=1) - diag_vals
            evals['diagonal_dominance'] = bool(np.all(diag_vals > row_sums))
            print("Using reordered system for Jacobi.")
        else:
            print("WARNING: Matrix is not diagonally dominant and reordering failed. Convergence not guaranteed.")

    diag = np.diag(A).astype(float)
    if np.any(np.isclose(diag, 0.0)):
        raise ValueError("Zero (or near-zero) diagonal element detected; cannot apply Jacobi.")

    inv_diag = 1.0 / diag
    L = np.tril(A, k=-1)
    R = np.triu(A, k=1)

    C = -(L + R) * inv_diag[:, np.newaxis]
    d = b * inv_diag
    print("\nCalculated Iteration Matrix C and Vector d.")

    rho = None
    try:
        if eval_options.get('spectral_radius', True):
            eigvals = np.linalg.eigvals(C)
            rho = float(np.max(np.abs(eigvals)))
            evals['spectral_radius'] = rho
            print(f"Spectral Radius (rho(C)): {rho:.6f}")
            if rho >= 1.0:
                print("WARNING: Spectral Radius >= 1.0. The method will likely not converge.")

    except np.linalg.LinAlgError:
        evals['spectral_radius'] = None
        rho = np.inf
        print("Could not compute spectral radius.")

    if eval_options.get('norms', True):
        norms = {
            "1-norm (col-sum)": float(np.linalg.norm(C, ord=1)),
            "2-norm (spectral)": float(np.linalg.norm(C, ord=2)),
            "inf-norm (row-sum)": float(np.linalg.norm(C, ord=np.inf))
        }
        evals['norms'] = norms
        print("\nMatrix C Norms:")
        for name, val in norms.items():
             print(f"- {name}: {val:.6f}")

    if eval_options.get('cond', True):
        try:
            cond = float(np.linalg.cond(A))
            evals['cond'] = cond
            print(f"Condition Number of A (cond(A)): {cond:.2f}")
        except np.linalg.LinAlgError:
            evals['cond'] = None
            print("Could not compute condition number.")

    norm_C_inf = evals.get('norms', {}).get('inf-norm (row-sum)', np.linalg.norm(C, ord=np.inf))
    a_priori_iters = None
    if eval_options.get('a_priori', True) and norm_C_inf < 1.0:
        x0 = d.copy()
        x1 = C @ x0 + d
        delta = np.linalg.norm(x1 - x0, ord=np.inf)
        if delta > 0:
            numerator_arg = eps * (1 - norm_C_inf) / delta
            if numerator_arg > 0:
                try:
                    a_priori_iters = int(np.ceil(np.log(numerator_arg) / np.log(norm_C_inf)))
                    a_priori_iters = max(a_priori_iters, 1)
                except Exception:
                    a_priori_iters = None
        evals['a_priori_iterations_estimate'] = None if a_priori_iters is None else int(a_priori_iters)
        print(f"A-Priori Iteration Estimate: {a_priori_iters}")


    k_est = a_priori_iters if (a_priori_iters is not None) else max_iter_est
    k_max = int(min(k_est if np.isfinite(k_est) else max_iter_est, max_iter_est))
    print(f"Maximum iterations set to: {k_max}")

    x = d.copy()
    converged = False
    k = 0

    print("\n--- Starting Iteration Process ---")
    try:
        for k in range(1, k_max + 1):
            x_new = C @ x + d
            diff_norm = np.linalg.norm(x_new - x, ord=np.inf)
            if k % (k_max // 10 if k_max >= 10 else 1) == 0 or k == 1:
                 print(f"Iter {k}: ||x^(k) - x^(k-1)||_inf = {diff_norm:.10e}")

            if diff_norm < eps:
                x = x_new
                converged = True
                print(f"Converged at iteration k={k}.")
                print(f"Final ||x^(k) - x^(k-1)||_inf = {diff_norm:.10e} < {eps:.10e}")
                break
            x = x_new
        else:
             print(f"Did not converge after max iterations ({k_max}).")

    except RuntimeWarning:
        print("WARNING: Runtime error occurred during iteration.")

    evals['iterations'] = int(k)
    evals['converged'] = bool(converged)
    print("--- Iteration Process Finished ---")

    if eval_options.get('residual', True):
        try:
            residual_vec = A @ x - b
            residual_norm = float(np.linalg.norm(residual_vec))
            denom = (np.linalg.norm(A) * np.linalg.norm(x))
            relative_residual = float(residual_norm / denom) if denom != 0 else None
            evals['residual_norm'] = residual_norm
            evals['relative_residual_norm'] = relative_residual
            print(f"Residual Norm (||Ax-b||): {residual_norm:.6e}")
            print(f"Relative Residual Norm: {relative_residual:.6e}")
        except Exception:
            evals['residual_norm'] = None
            evals['relative_residual_norm'] = None
            print("Could not compute residual norms.")

    evals['epsilon'] = float(eps)
    evals['matrix_size'] = int(n)

    return x, evals