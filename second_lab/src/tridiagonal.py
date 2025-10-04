import numpy as np
from decimal import Decimal, getcontext

def solve_tridiagonal(A, f):
    n = A.shape[0]
    tol = 1e-12
    y = None

    print("--- Starting Tridiagonal Matrix Algorithm (TDMA) ---")

    print("1. Checking: Tridiagonal matrix structure.")
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and not A[i, j] < tol:
                print("ERROR: Matrix is not tridiagonal!")
                return y, 0
    print("   -> OK. Matrix is tridiagonal.")

    if n == 0:
        return np.array([]), 0

    c = np.array([A[i, i] for i in range(n)])
    a = np.array([A[i, i - 1] for i in range(1, n)])
    b = np.array([A[i, i + 1] for i in range(0, n - 1)])


    print("2. Checking: Stability conditions.")

    if abs(c[0]) < tol:
        print("c[0] (A[0,0]) to close to zero.")

    non_strict_dominance = True
    for i in range(1, n - 1):
        if not (abs(c[i]) >= abs(a[i - 1]) + abs(b[i])):
            non_strict_dominance = False
            break

    if not (non_strict_dominance):
        print("   -> WARNING! Stability conditions (sufficient) NOT met. Calculation proceeds, but may be unstable.")
    else:
        print("   -> OK. Stability conditions met.")

    getcontext().prec = 80
    determinant = Decimal(c[0])

    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = - b[0] / c[0] if n > 1 else 0.0
    beta[0] = f[0] / c[0]

    try:
        print("3. Forward Elimination...")

        for i in range(1, n):
            denom = c[i] + a[i - 1] * alpha[i - 1]

            determinant *= Decimal(denom)

            if i < n - 1:
                alpha[i] = - b[i] / denom
            else:
                alpha[i] = 0.0
            beta[i] = (f[i] - a[i - 1] * beta[i - 1]) / denom

        print("4. Backward Substitution...")

        y = np.zeros(n)
        y[n - 1] = beta[n - 1]
        for i in range(n - 2, -1, -1):
            y[i] = alpha[i] * y[i + 1] + beta[i]

    except ZeroDivisionError:
        print("ERROR: Zero division encountered during calculation. Matrix is likely singular.")
        return None, 0

    print("--- Calculation finished ---")

    return y, determinant