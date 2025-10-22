import numpy as np
from pathlib import Path

n = 1024
ZERO_PERCENTAGE = 0
INPUT_ID = 5
GENERATE_RIGHT_MATRIX = True

def generate_strictly_diagonally_dominant(n, alpha=1.5, offdiag_range=(-1.0, 1.0), b_range=(-10, 10)):
    A = np.random.uniform(offdiag_range[0], offdiag_range[1], size=(n, n))
    np.fill_diagonal(A, 0.0)
    off_diag_sums = np.sum(np.abs(A), axis=1)
    diag_vals = off_diag_sums * alpha + np.finfo(float).eps * np.ones(n)
    signs = np.where(np.random.rand(n) < 0.5, -1.0, 1.0)
    diag_vals = diag_vals * signs
    np.fill_diagonal(A, diag_vals)
    b = np.random.uniform(b_range[0], b_range[1], size=n)
    return A, b

if GENERATE_RIGHT_MATRIX:
    alpha = 1.6
    matrix_A, b = generate_strictly_diagonally_dominant(n, alpha=alpha)
    matrix = np.hstack([matrix_A, b.reshape(-1, 1)])
else:
    random_mask = np.random.rand(n, n + 1)
    zero_mask = random_mask < ZERO_PERCENTAGE
    matrix = np.random.uniform(-1000, 1000, size=(n, n + 1))
    matrix[zero_mask] = 0

output_dir = Path("..") / "inputs"
output_dir.mkdir(parents=True, exist_ok=True)

if n > 1000:
    file_path = output_dir / f"input{INPUT_ID}.npz"
    np.savez_compressed(file_path, a=matrix[:, :n], b=matrix[:, n])
    print(f"Matrices saved to file {file_path}")
else:
    file_path = output_dir / f"input{INPUT_ID}.txt"
    with open(file_path, 'w') as f:
        f.write(f"{n}\n")
        np.savetxt(f, matrix)
    print(f"Matrix saved to file {file_path}")
