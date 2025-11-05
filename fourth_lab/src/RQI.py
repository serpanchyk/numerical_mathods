import numpy as np
import time
import warnings
from support_functions import *

warnings.filterwarnings('ignore', category=RuntimeWarning)

def solve_inverse_iteration(A_orig, eval_options=None, tol=1e-12, k_max=200):

    if eval_options is None:
        eval_options = {
            'cond': True,
            'benchmark': True,
            'iterations': True,
        }

    A = A_orig.astype(float).copy()
    n = A.shape[0]

    evals = {
        'convergence': True,
        'matrix_size': n,
        'tolerance': tol
    }

    if not is_symmetric(A):
        print('A is not symmetric. Method works best for symmetric matrices.')

    rng = np.random.default_rng(0)
    x = rng.random(n)
    x /= np.linalg.norm(x)

    start_time = time.time()
    residual_norm = np.inf
    k = 0
    rho = float(x.T @ A @ x)

    # Головний цикл зворотних ітерацій
    while residual_norm > tol and k < k_max:
        # Розв’язуємо A y = x  (еквівалентно y = A⁻¹ x, але без обернення)
        y = np.linalg.solve(A, x)
        x = y / np.linalg.norm(y)

        rho = float(x.T @ A @ x)
        residual_norm = np.linalg.norm(A @ x - rho * x)
        k += 1

    end_time = time.time()

    if k >= k_max:
        print(f"Inverse iteration ended without convergence after {k_max} iterations.")
        evals['convergence'] = False

    eigen_pairs = [[rho, x]]

    evals['execution_time_sec'] = end_time - start_time
    evals['iterations'] = k
    evals['residual_norm'] = residual_norm

    if eval_options.get('cond'):
        evals['cond'] = np.linalg.cond(A)

    if eval_options.get('benchmark'):
        try:
            true_eigs = np.linalg.eigvals(A)
            smallest_true = true_eigs[np.argmin(np.abs(true_eigs))]
            evals['benchmark value'] = smallest_true
            evals['absolute error'] = abs(rho - smallest_true)
            evals['relative error'] = evals['absolute error'] / abs(smallest_true)
        except Exception:
            print("Benchmark evaluation failed.")

    return eigen_pairs, evals
