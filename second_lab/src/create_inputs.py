import numpy as np
from pathlib import Path

n = 1000
zero_percentage = 0
input_id = 2

random_mask = np.random.rand(n, n + 1)
zero_mask = random_mask < zero_percentage
matrix = np.random.uniform(-1000, 1000, size=(n, n + 1))
matrix[zero_mask] = 0

output_dir = Path("..") / "inputs"
output_dir.mkdir(parents=True, exist_ok=True)

if n > 1000:
    file_path = output_dir / f"input{input_id}.npz"
    np.savez_compressed(file_path, a=matrix[:, :n], b=matrix[:, n])
    print(f"Matrices saved to file {file_path}")
else:
    file_path = output_dir / f"input{input_id}.txt"
    with open(file_path, 'w') as f:
        f.write(f"{n}\n")
        np.savetxt(f, matrix)
    print(f"Matrix saved to file {file_path}")