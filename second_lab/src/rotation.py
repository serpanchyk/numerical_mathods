import numpy as np
from decimal import Decimal, getcontext

def solve_rotation(a, b, det_evaluation=True):
    print("Starting Gaussian elimination...")
    n = len(a)
    a = a.copy()
    b = b.copy()
    det = None

    try:
        print("Forward elimination...")
        for k in range(n - 1):
            for i in range(k+1, n):
                r = np.sqrt(a[k,k] ** 2 + a[i,k] ** 2)
                c = a[k,k] / r
                s = a[i,k] / r

                row_k = a[k, k:].copy()
                row_i = a[i, k:].copy()
                a[k, k:] = c * row_k + s * row_i
                a[i, k:] = -s * row_k + c * row_i

                b_k_old = b[k]
                b_i_old = b[i]
                b[k] = c * b_k_old + s * b_i_old
                b[i] = -s * b_k_old + c * b_i_old
        print("Forward elimination finished.")

        if det_evaluation:
            print("Calculating determinant...")
            getcontext().prec = n * 5
            det = Decimal(1)
            for i in range(n):
                det *= Decimal(a[i][i])
            print("Determinant calculated.")

        print("Backward substitution...")
        x = np.zeros(n)
        x[n - 1] = b[n - 1] / a[n - 1, n - 1]

        for k in range(n - 2, -1, -1):
            x[k] = (b[k] - np.sum(a[k, k + 1:] * x[k + 1:])) / a[k, k]
        print("Solution found.")

    except (FloatingPointError, ZeroDivisionError):
        print("Error: Matrix is singular. No solution.")
        return None, 0

    print("Calculations finished.")
    return x, det
