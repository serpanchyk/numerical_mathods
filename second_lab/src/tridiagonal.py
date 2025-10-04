import numpy as np
from decimal import Decimal, getcontext

def solve_tridiagonal(A, f):
    n = len(A)
    y = None

    getcontext().prec = 100
    determinant = Decimal(1)

    print("--- Starting Tridiagonal Matrix Algorithm (TDMA) ---")

    print("1. Checking: Tridiagonal matrix structure.")
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and not A[i, j] == 0:
                print("ERROR: Matrix is not tridiagonal!")
                return y, 0
    print("   -> OK. Matrix is tridiagonal.")

    c = np.diag(A)
    a_inner = np.diag(A, k=-1)
    b_inner = np.diag(A, k=1)

    try:
        chi1 = -A[0, 1] / A[0, 0]
        mu1 = f[0] / A[0, 0]
    except (IndexError, ZeroDivisionError):
        print("ERROR: Invalid matrix structure for first boundary condition (A[0, 0] == 0).")
        return y, 0

    try:
        chi2 = -A[n - 1, n - 2] / A[n - 1, n - 1]
        mu2 = f[n - 1] / A[n - 1, n - 1]
    except (IndexError, ZeroDivisionError):
        print("ERROR: Invalid matrix structure for last boundary condition (A[n-1, n-1] == 0).")
        return y, 0

    print(f"   -> Boundary conditions: chi1={chi1:.4f}, mu1={mu1:.4f}, chi2={chi2:.4f}, mu2={mu2:.4f}")

    print("2. Checking: Stability conditions.")
    non_strict_dominance = True
    for i in range(1, n - 1):
        if not (abs(c[i]) >= abs(a_inner[i - 1]) + abs(b_inner[i])):
            non_strict_dominance = False
            break

    non_strict_boundary = (abs(chi1) <= 1) and (abs(chi2) <= 1)

    strict_dominance = any(abs(c[i]) > abs(a_inner[i - 1]) + abs(b_inner[i]) for i in range(1, n - 1))
    strict_boundary = (abs(chi1) < 1) or (abs(chi2) < 1)

    all_non_strict = non_strict_dominance and non_strict_boundary
    at_least_one_strict = strict_dominance or strict_boundary

    if not (all_non_strict and at_least_one_strict):
        print("   -> WARNING! Stability conditions (sufficient) NOT met. Calculation proceeds, but may be unstable.")
    else:
        print("   -> OK. Stability conditions met.")

    alfa = np.zeros(n)
    beta = np.zeros(n)

    alfa[0] = chi1
    beta[0] = mu1

    try:
        print("3. Forward Elimination...")

        D0 = c[1] - a_inner[0] * alfa[0]
        alfa[1] = b_inner[1] / D0
        beta[1] = (f[1] - a_inner[0] * beta[0]) / D0
        determinant *= Decimal(D0)
        if np.isclose(D0, 0): raise ZeroDivisionError

        for i in range(2, n):
            D = c[i] - a_inner[i - 1] * alfa[i - 1]

            if i < n - 1:
                alfa[i] = b_inner[i] / D
            else:
                alfa[i] = 0

            beta[i] = (f[i] - a_inner[i - 1] * beta[i - 1]) / D

            determinant *= Decimal(D)

            if np.isclose(D, 0):
                raise ZeroDivisionError

        y = np.zeros(n)

        print("4. Backward Substitution...")

        D_final = (1 - chi2 * alfa[n - 1])
        determinant *= Decimal(D_final)

        if np.isclose(D_final, 0):
            raise ZeroDivisionError

        y[n - 1] = (mu2 + chi2 * beta[n - 1]) / D_final

        for i in range(n - 2, -1, -1):
            y[i] = alfa[i] * y[i + 1] + beta[i]

    except ZeroDivisionError:
        print("ERROR: Zero division encountered during calculation. Matrix is likely singular.")
        return None, 0

    print("--- Calculation finished ---")

    return y, determinant