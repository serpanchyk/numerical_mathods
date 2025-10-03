import numpy as np
from decimal import Decimal, getcontext

def solve_gaussian(a, b, type='partial', det_evaluation=True):
    print("Starting Gaussian elimination...")
    n = len(a)
    a = a.copy()
    b = b.copy()
    det = None
    variable_order = np.arange(n)
    permutation_count = 0

    try:
        print("Forward elimination...")
        for k in range(n - 1):
            if type == 'partial':
                pivot_row = np.argmax(np.abs(a[k:, k])) + k
                if pivot_row != k:
                    a[[k, pivot_row]] = a[[pivot_row, k]]
                    b[[k, pivot_row]] = b[[pivot_row, k]]
                    permutation_count += 1
            elif type == 'full':
                pivot_coords = np.unravel_index(np.argmax(np.abs(a[k:, k:])), a[k:, k:].shape)
                pivot_row, pivot_col = pivot_coords[0] + k, pivot_coords[1] + k
                if pivot_row != k:
                    b[[k, pivot_row]] = b[[pivot_row, k]]
                    a[[k, pivot_row]] = a[[pivot_row, k]]
                    permutation_count += 1
                if a[k, pivot_col] != a[k, k]:
                    a[:, [k, pivot_col]] = a[:, [pivot_col, k]]
                    variable_order[[k, pivot_col]] = variable_order[[pivot_col, k]]
                    permutation_count += 1

            for i in range(k + 1, n):
                factor = a[i, k] / a[k, k]
                b[i] -= factor * b[k]
                a[i, k:] -= factor * a[k, k:]
        print("Forward elimination finished.")

        if det_evaluation:
            print("Calculating determinant...")
            getcontext().prec = n * 5
            det = Decimal(1)
            for i in range(n):
                det *= Decimal(a[i][i])
            det *= (-1) ** permutation_count
            print("Determinant calculated.")

        print("Backward substitution...")
        temp_x = np.zeros(n)
        temp_x[n - 1] = b[n - 1] / a[n - 1, n - 1]

        for k in range(n - 2, -1, -1):
            temp_x[k] = (b[k] - np.sum(a[k, k + 1:] * temp_x[k + 1:])) / a[k, k]
        print("Solution found.")

    except (FloatingPointError, ZeroDivisionError):
        print("Error: Matrix is singular. No solution.")
        return None, 0

    x = np.zeros(n)
    x[variable_order] = temp_x

    print("Calculations finished.")
    return x, det
