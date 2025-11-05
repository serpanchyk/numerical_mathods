import warnings
import numpy as np
import time
from support_functions import *

warnings.filterwarnings('ignore', category=RuntimeWarning)

def solve_jacobi(A_orig, eval_options=None, tol=1e-8, k_max=100000):
    if eval_options is None:
        eval_options = {'cond': True, 'benchmark': True, 'iterations': True}

    A = A_orig.copy().astype(float)
    n = A.shape[0]
    V = np.eye(n)

    evals = {'convergence': True, 'matrix_size': n, 'tolerance': tol}
    k = 0

    if not is_symmetric(A):
        print('A is not symmetric. Jacobi method is designed for symmetric matrices.')
        return None, evals

    start_time = time.time()

    # current off-diagonal Frobenius norm (sqrt of sum squares of off-diagonal)
    def off_diag_norm(mat):
        return np.sqrt(np.sum(mat**2) - np.sum(np.diag(mat)**2))

    od_norm = off_diag_norm(A)

    while od_norm > tol and k < k_max:
        for p in range(n-1):
            for q in range(p+1, n):
                a_pp = A[p, p]
                a_qq = A[q, q]
                a_pq = A[p, q]

                if abs(a_pq) <= 0.0:
                    continue

                phi = 0.5 * np.arctan2(2.0 * a_pq, a_qq - a_pp)
                c = np.cos(phi)
                s = np.sin(phi)

                G = np.eye(n)
                G[p, p] = c
                G[q, q] = c
                G[p, q] = s
                G[q, p] = -s

                A = G.T @ A @ G

                A = (A + A.T) / 2.0

                V = V @ G

                k += 1
                if k >= k_max:
                    break
            if k >= k_max:
                break

        od_norm = off_diag_norm(A)

    end_time = time.time()

    if od_norm > tol:
        print(f"Jacobi algorithm ended without convergence after {k} rotations (k_max={k_max}).")
        evals['convergence'] = False

    eigen_vals = np.diag(A)
    eigen_vecs = V

    idx = np.argsort(eigen_vals)
    eigen_pairs = [[eigen_vals[i], eigen_vecs[:, i]] for i in idx]

    evals['execution_time_sec'] = end_time - start_time
    if eval_options.get('cond', False):
        evals['cond'] = np.linalg.cond(A_orig)
    if eval_options.get('iterations', False):
        evals['iterations'] = k

    if eval_options.get('benchmark', False):
        try:
            true_vals = np.linalg.eigvals(A_orig)
            sorted_true = np.sort(true_vals)
            sorted_jacobi = np.sort(eigen_vals)
            abs_error = np.linalg.norm(sorted_jacobi - sorted_true)
            evals['absolute_error'] = abs_error
            evals['relative_error'] = abs_error / np.linalg.norm(sorted_true)
        except Exception:
            print("Benchmark evaluation failed.")

    return eigen_pairs, evals

