import numpy as np
from pathlib import Path

n = 1000
ZERO_PERCENTAGE = 0
INPUT_ID = 7
GENERATE_TRIDIAGONAL = False

if GENERATE_TRIDIAGONAL:
    sub_diag = np.random.uniform(-100, 100, n - 1)
    super_diag = np.random.uniform(-100, 100, n - 1)

    sub_diag_padded = np.concatenate(([0], sub_diag))
    super_diag_padded = np.concatenate((super_diag, [0]))

    required_c_magnitude = np.abs(sub_diag_padded) + np.abs(super_diag_padded)

    stability_buffer = np.random.uniform(1.0, 100.0, n)
    main_diag_magnitude = required_c_magnitude + stability_buffer

    main_diag_sign = np.random.choice([-1, 1], n)
    main_diag = main_diag_magnitude * main_diag_sign

    a = np.zeros((n, n))
    np.fill_diagonal(a, main_diag)
    np.fill_diagonal(a[1:], sub_diag)
    np.fill_diagonal(a[:, 1:], super_diag)
    a[0, 0] = 1
    a[-1, -1] = 1

    f = np.random.uniform(-100, 100, n)

    matrix = np.hstack((a, f.reshape(-1, 1)))

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
