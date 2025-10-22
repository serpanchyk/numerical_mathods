import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

def try_make_diagonally_dominant(A, b):
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
            return A, b, False

    A_permuted = A[permutation, :]
    b_permuted = b[permutation]

    diag_vals = np.abs(np.diag(A_permuted))
    row_sums = np.sum(np.abs(A_permuted), axis=1) - diag_vals
    is_dominant = np.all(diag_vals > row_sums)

    return A_permuted, b_permuted, is_dominant


def solve_jacobi(A, b, eps=1e-10, max_iter_est=100_000, eval_options=None):
    """
    Jacobi solver that returns (x, iterations, evals_dict).
    eval_options: dict to enable/disable specific evaluations. Supported keys:
      'cond', 'spectral_radius', 'spectral_vector', 'norms', 'residual',
      'benchmark', 'a_priori'
    By default all are enabled.
    """
    if eval_options is None:
        eval_options = {
            'cond': True,
            'spectral_radius': True,
            'spectral_vector': False,
            'norms': True,
            'residual': True,
            'benchmark': False,
            'a_priori': True
        }

    evals = {}
    n = A.shape[0]

    # Diagonal dominance check and optional row reordering
    initial_diag = np.abs(np.diag(A))
    initial_row_sums = np.sum(np.abs(A), axis=1) - initial_diag
    diag_dom = np.all(initial_diag > initial_row_sums)
    evals['diagonal_dominance'] = bool(diag_dom)

    if not diag_dom:
        A_new, b_new, success = try_make_diagonally_dominant(A.copy(), b.copy())
        evals['reordered_for_dominance'] = bool(success)
        if success:
            A, b = A_new, b_new
            # Update diag dominance flag after reorder
            diag_vals = np.abs(np.diag(A))
            row_sums = np.sum(np.abs(A), axis=1) - diag_vals
            evals['diagonal_dominance'] = bool(np.all(diag_vals > row_sums))

    diag = np.diag(A).astype(float)
    if np.any(np.isclose(diag, 0.0)):
        raise ValueError("Zero (or near-zero) diagonal element detected; cannot apply Jacobi.")

    inv_diag = 1.0 / diag
    L = np.tril(A, k=-1)
    R = np.triu(A, k=1)

    C = -(L + R) * inv_diag[:, np.newaxis]
    d = b * inv_diag

    # spectral radius and spectral vector (if requested)
    rho = None
    spectral_vector = None
    try:
        if eval_options.get('spectral_radius', True) or eval_options.get('spectral_vector', False):
            eigvals = np.linalg.eigvals(C)
            rho = float(np.max(np.abs(eigvals)))
            evals['spectral_radius'] = rho
            if eval_options.get('spectral_vector', False):
                # compute eigenvector for eigenvalue with max modulus
                vals, vecs = np.linalg.eig(C)
                idx = int(np.argmax(np.abs(vals)))
                spectral_vector = vecs[:, idx]
                evals['spectral_vector'] = spectral_vector.tolist()
    except np.linalg.LinAlgError:
        evals['spectral_radius'] = None
        evals['spectral_vector'] = None
        rho = np.inf

    # Norms
    if eval_options.get('norms', True):
        evals['norms'] = {
            "1-norm (col-sum)": float(np.linalg.norm(C, ord=1)),
            "2-norm (spectral)": float(np.linalg.norm(C, ord=2)),
            "inf-norm (row-sum)": float(np.linalg.norm(C, ord=np.inf))
        }

    # Condition number (optional)
    if eval_options.get('cond', True):
        try:
            evals['cond'] = float(np.linalg.cond(A))
        except np.linalg.LinAlgError:
            evals['cond'] = None

    # a priori estimate if possible
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

    # iteration bounds
    k_est = a_priori_iters if (a_priori_iters is not None) else max_iter_est
    k_max = int(min(k_est if np.isfinite(k_est) else max_iter_est, max_iter_est))

    # Iteration loop
    x = d.copy()
    converged = False
    k = 0
    for k in range(1, k_max + 1):
        x_new = C @ x + d
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            x = x_new
            converged = True
            break
        x = x_new

    evals['iterations'] = int(k)
    evals['converged'] = bool(converged)

    # Residuals (optional)
    if eval_options.get('residual', True):
        try:
            residual_norm = float(np.linalg.norm(A @ x - b))
            denom = (np.linalg.norm(A) * np.linalg.norm(x))
            relative_residual = float(residual_norm / denom) if denom != 0 else None
            evals['residual_norm'] = residual_norm
            evals['relative_residual_norm'] = relative_residual
        except Exception:
            evals['residual_norm'] = None
            evals['relative_residual_norm'] = None

    # Put spectral radius in evals if not already set
    if 'spectral_radius' not in evals:
        try:
            evals['spectral_radius'] = float(np.max(np.abs(np.linalg.eigvals(C))))
        except Exception:
            evals['spectral_radius'] = None

    # final metadata
    evals['epsilon'] = float(eps)
    evals['matrix_size'] = int(n)

    return x, int(k), evals
