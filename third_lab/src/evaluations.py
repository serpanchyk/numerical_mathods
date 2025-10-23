import numpy as np
from itertools import permutations, product
from typing import Tuple, List


def apply_col_scaling(A: np.ndarray, scales: Tuple[float, ...]) -> np.ndarray:
    S = np.diag(scales)
    return A.dot(S)


def is_strictly_diagonally_dominant(M: np.ndarray) -> bool:
    diag = np.abs(np.diag(M))
    off = np.sum(np.abs(M), axis=1) - diag
    return np.all(diag > off)


def try_make_diagonally_dominant(A: np.ndarray, b: np.ndarray, max_scale: int = 12) -> Tuple[
    np.ndarray, np.ndarray, bool, List[int], np.ndarray]:
    print("--- Attempting to make matrix A diagonally dominant ---")
    n = A.shape[0]
    default_scales = np.ones(n, dtype=float)

    if is_strictly_diagonally_dominant(A):
        print("Matrix already strictly diagonally dominant ✅")
        return A.copy(), b.copy(), True, list(range(n)), default_scales

    scale_values = list(range(1, max_scale + 1))

    if n == 3:
        iterator = product(scale_values, repeat=3)
    elif n <= 4:
        print(f"WARNING: n={n}. Scale search will be very slow ({len(scale_values) ** n} combinations).")
        iterator = product(scale_values, repeat=n)
    else:
        print(f"WARNING: n={n}. Skipping scale search, trying permutations only.")
        iterator = [tuple(np.ones(n, dtype=int))]

    for row_perm in permutations(range(n)):
        A_row = A[list(row_perm), :]
        b_row = b[list(row_perm)]

        for col_perm in permutations(range(n)):
            A_perm = A_row[:, list(col_perm)]

            if is_strictly_diagonally_dominant(A_perm):
                print("✅ Found diagonally dominant form (permutations only)")
                return A_perm, b_row, True, list(col_perm), default_scales

            if n <= 4:
                scale_iterator = product(scale_values, repeat=n) if n > 3 else iterator
                for scales_int in scale_iterator:
                    scales = np.array(scales_int, dtype=float)
                    A_scaled = apply_col_scaling(A_perm, scales)
                    if is_strictly_diagonally_dominant(A_scaled):
                        print(f"✅ Found dominant form with scaling! Scales: {scales}")
                        return A_scaled, b_row, True, list(col_perm), scales

            if n == 3:
                iterator = product(scale_values, repeat=3)

    print("❌ Could not make the matrix diagonally dominant with given search limits.")
    print("--- End dominance attempt ---")
    return A, b, False, list(range(n)), default_scales


def compute_iteration_matrix(A, b):
    diag = np.diag(A)
    inv_diag = 1.0 / diag
    L = np.tril(A, k=-1)
    R = np.triu(A, k=1)

    C = -(L + R) * inv_diag[:, np.newaxis]
    d = b * inv_diag

    return C, d

def get_spectral_radius(A):
    try:
        eigvals = np.linalg.eigvals(A)
        rho = float(np.max(np.abs(eigvals)))
    except np.linalg.LinAlgError:
        rho = np.inf
        print("Could not compute spectral radius.")

    return rho

def get_norms(A):
    norms = {
        "1-norm (col-sum)": float(np.linalg.norm(A, ord=1)),
        "2-norm (spectral)": float(np.linalg.norm(A, ord=2)),
        "inf-norm (row-sum)": float(np.linalg.norm(A, ord=np.inf))
    }

    return norms

def get_a_priori(C, d, eps):
    norm_C_inf = np.linalg.norm(C, ord=np.inf)
    a_priori_iters = None
    if norm_C_inf < 1.0:
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

    return a_priori_iters

def get_residual(A, x, b):
    try:
        residual_vec = A @ x - b
        residual_norm = float(np.linalg.norm(residual_vec))
        denom = (np.linalg.norm(A) * np.linalg.norm(x))
        relative_residual = float(residual_norm / denom) if denom != 0 else None
    except Exception:
        residual_norm = None
        relative_residual = None

    return residual_norm, relative_residual