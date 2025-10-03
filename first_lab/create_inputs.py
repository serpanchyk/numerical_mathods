import numpy as np

n = 2000
zero_percentage = 0.1
input_id = 8

random_mask = np.random.rand(n, n + 1)
zero_mask = random_mask < zero_percentage
matrix = np.random.uniform(-1000, 1000, size=(n, n + 1))
matrix[zero_mask] = 0

if n > 1000:
    np.savez_compressed(f"inputs/input{input_id}.npz", a=matrix[:, :n], b=matrix[:, n])
    print(f"Matrices saved to file inputs/input{input_id}.npz")
else:
    with open(f"inputs/input{input_id}.txt", 'w') as f:
        f.write(f"{n}\n")
        np.savetxt(f, matrix)
    print(f"Matrix saved to file inputs/input{input_id}.txt")