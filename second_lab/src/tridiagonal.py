import numpy as np
from decimal import Decimal, getcontext

def solve_tridiagonal(D_matrix, f_vector):
    n = D_matrix.shape[0]
    tol = 1e-12

    print("--- Starting Shuttle Method for Tridiagonal Systems ---")

    print("1. Checking: Tridiagonal matrix structure.")
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and not abs(D_matrix[i, j]) < tol:
                print("ERROR: Matrix is not tridiagonal!")
                return None, 0
    print("   -> OK. Matrix is tridiagonal.")

    if D_matrix.shape[0] != D_matrix.shape[1] or D_matrix.shape[0] != len(f_vector):
        print("Error: Matrix and vector dimensions do not match.")
        return None, 0

    N = D_matrix.shape[0] - 1
    if N < 2:
        print("Error: Matrix is too small for the shuttle method.")
        return None, 0

    print("2. Checking: Sufficient conditions for correctness and stability.")
    H1 = -D_matrix[0, 1]
    H2 = -D_matrix[N, N - 1]

    c_diag = np.array([D_matrix[i, i] for i in range(1, N)])
    a_diag = np.array([D_matrix[i, i - 1] for i in range(1, N)])
    b_diag = np.array([D_matrix[i, i + 1] for i in range(1, N)])

    is_diagonally_dominant = True
    strict_inequality_found = False

    for i in range(N - 1):
        if abs(c_diag[i]) < abs(a_diag[i]) + abs(b_diag[i]):
            is_diagonally_dominant = False
            break
        if abs(c_diag[i]) > abs(a_diag[i]) + abs(b_diag[i]):
            strict_inequality_found = True

    h1_cond_met = abs(H1) <= 1
    h2_cond_met = abs(H2) <= 1

    if abs(H1) < 1 or abs(H2) < 1:
        strict_inequality_found = True

    all_conditions_met = is_diagonally_dominant and h1_cond_met and h2_cond_met and strict_inequality_found

    if not all_conditions_met:
        print(
            "   -> WARNING! Sufficient conditions for stability are NOT met. Calculation proceeds, but may be unstable.")
    else:
        print("   -> OK. Sufficient conditions for stability are met.")

    print("3. Extracting coefficients from matrix and vector...")
    mu1 = f_vector[0]
    mu2 = f_vector[N]

    A_coeffs = np.array([D_matrix[i, i - 1] for i in range(1, N)])
    B_coeffs = np.array([D_matrix[i, i + 1] for i in range(1, N)])
    C_coeffs = np.array([-D_matrix[i, i] for i in range(1, N)])
    f_coeffs = np.array([-f_vector[i] for i in range(1, N)])
    print("   -> Coefficients extracted.")

    m = N // 2

    alpha = np.zeros(N + 2)
    beta = np.zeros(N + 2)
    gamma = np.zeros(N + 2)
    delta = np.zeros(N + 2)
    y = np.zeros(N + 1)

    getcontext().prec = 80
    determinant = Decimal(1)

    try:
        print(f"4. Rightward pass (from i=0 to i={m})...")
        alpha[1] = H1
        beta[1] = mu1

        for i in range(1, m + 1):
            idx = i - 1
            denominator = C_coeffs[idx] - A_coeffs[idx] * alpha[i]
            determinant *= Decimal(denominator)
            if abs(denominator) < tol:
                raise ZeroDivisionError(f"Denominator is near zero in the rightward pass at i={i}.")
            alpha[i + 1] = B_coeffs[idx] / denominator
            beta[i + 1] = (f_coeffs[idx] + A_coeffs[idx] * beta[i]) / denominator

        print(f"5. Leftward pass (from i={N} to i={m + 1})...")
        gamma[N] = H2
        delta[N] = mu2

        for i in range(N - 1, m, -1):
            idx = i - 1
            denominator = C_coeffs[idx] - B_coeffs[idx] * gamma[i + 1]
            determinant *= Decimal(denominator)
            if abs(denominator) < tol:
                raise ZeroDivisionError(f"Denominator is near zero in the leftward pass at i={i}.")
            gamma[i] = A_coeffs[idx] / denominator
            delta[i] = (f_coeffs[idx] + B_coeffs[idx] * delta[i + 1]) / denominator

        print(f"6. Meeting at point m={m} and solving for y[m]...")
        denominator_ym = 1 - alpha[m + 1] * gamma[m + 1]
        determinant *= Decimal(denominator_ym)

        if abs(denominator_ym) < tol:
            raise ZeroDivisionError("Denominator is near zero when calculating y_m.")

        y[m] = (alpha[m + 1] * delta[m + 1] + beta[m + 1]) / denominator_ym

        print("7. Backward substitution from center to boundaries...")
        for i in range(m - 1, -1, -1):
            y[i] = alpha[i + 1] * y[i + 1] + beta[i + 1]

        for i in range(m + 1, N + 1):
            y[i] = gamma[i] * y[i - 1] + delta[i]

    except ZeroDivisionError as e:
        print(f"ERROR: {e} Matrix is likely singular.")
        return None, 0

    print("--- Calculation finished ---")
    return y, determinant