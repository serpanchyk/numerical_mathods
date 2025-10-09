import numpy as np
from decimal import Decimal
from rotation import solve_rotation
from pathlib import Path
import os

def read_sole_data(input_id):
    input_dir = Path("..") / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    npz_path = input_dir / f"input{input_id}.npz"
    txt_path = input_dir / f"input{input_id}.txt"

    try:
        if os.path.exists(npz_path):
            print(f"Reading data from file {npz_path}...")
            data = np.load(npz_path)
            if 'a' in data and 'b' in data:
                a = data['a']
                b = data['b']
                print("Data loaded successfully.")
                return a, b
            else:
                print("Error: NPZ file does not contain keys 'a' and 'b'.")
                return None, None

        elif os.path.exists(txt_path):
            print(f"Reading data from file {txt_path}...")
            full_data = np.genfromtxt(txt_path, skip_header=1, delimiter=None, filling_values=0)
            if full_data.ndim < 2:
                print("Error: Not enough data for matrix and vector.")
                return None, None
            a = full_data[:, :-1].copy()
            b = full_data[:, -1].copy()
            if a.shape[0] != a.shape[1] or a.shape[0] != b.shape[0]:
                print("Error: dimensions do not match.")
                return None, None
            print("Data loaded successfully.")
            return a, b

        else:
            print("Error: Data file not found.")
            return None, None

    except Exception as error:
        print(f"Error reading file: {error}")
        return None, None

def method_evaluation(a, x, b, det, execution_time, input_id, decimal_places, method):
    evaluation_dir = Path("..") / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    txt_path = evaluation_dir / f"evaluation{input_id}.txt"

    print("Starting method evaluation...")
    print("Step 1: Calculating benchmark determinant (np.linalg.det)...")
    try:
        det_benchmark = np.linalg.det(a)
        print("Benchmark determinant successfully calculated.")
    except RuntimeWarning as e:
        det_benchmark = None
        print(f"Error calculating benchmark determinant: {e}")

    print("Step 2: Calculating benchmark solution (np.linalg.solve)...")
    try:
        x_benchmark = np.linalg.solve(a, b)
        print("Benchmark solution successfully calculated.")
    except np.linalg.LinAlgError:
        x_benchmark = None
        print("Error: Matrix is singular. Cannot calculate benchmark solution.")

    print(f"Step 3: Opening file to write results: {txt_path}")
    with open(txt_path, 'w') as f:
        f.write(f"Method name: {method.__name__}\n")
        f.write(f"Matrix size: {a.shape[0]}\n")
        f.write(f"Execution time: {execution_time}\n")
        if det is not None:
            f.write(f"Determinant: {det:.{decimal_places}g}\n")
        if det_benchmark is not None:
            det_benchmark = Decimal(str(det_benchmark))
            f.write(f"Benchmark determinant: {det_benchmark:.{decimal_places}g}\n")
            abs_error_det = abs(det - det_benchmark)
            rel_error_det = abs_error_det / abs(det_benchmark)
            f.write(f"Absolute error of determinant: {abs_error_det}\n")
            f.write(f"Relative error of determinant: {rel_error_det}\n")

        if x is not None:
            print("Step 4: Calculating stability error...")
            epsilon = 1e-8
            b_perturbed = b + epsilon * np.random.randn(*b.shape)
            x_perturbed, det_perturbed = method(a, b_perturbed,)
            stability_error = np.linalg.norm(x_perturbed - x)
            f.write(f"Stability error: {stability_error}\n")
            print("Stability error calculated.")

            print("Step 5: Calculating residual norm...")
            residual_norm = np.linalg.norm(a @ x - b)
            relative_residual_norm = residual_norm / np.linalg.norm(a) / np.linalg.norm(x)
            f.write(f"Residual norm of solution: {residual_norm}\n")
            f.write(f"Relative residual norm: {relative_residual_norm}\n")


        print("Step 6: Calculating condition number of matrix Cond(A)...")
        cond_a = np.linalg.cond(a)
        f.write(f"Cond A: {cond_a}\n")
        print("Condition number calculated.")
        if x is not None and x_benchmark is not None:
            print("Step 7: Calculating solution errors (absolute and relative)...")
            abs_error_solution = np.linalg.norm(x - x_benchmark)
            rel_error_solution = abs_error_solution / np.linalg.norm(x_benchmark)
            f.write(f"Absolute error of solution: {abs_error_solution}\n")
            f.write(f"Relative error of solution: {rel_error_solution}\n")
            print("Solution errors calculated.")

        f.write(f"Machine error for IEEE 754 standard ε ≈ {2.2e-16}")
    print(f"Evaluation complete. Results saved to {txt_path}'.")

def save_solution(x, input_id, decimal_places):
    output_dir = Path("..") / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"output{input_id}.npz"
    txt_path = output_dir / f"output{input_id}.txt"

    if x is not None:
        n = len(x)
        if n > 1000:
            np.savez_compressed(npz_path, x=x)
            print(f"Solution saved to {npz_path}")
        else:
            np.savetxt(txt_path, x, fmt=f'%.{decimal_places}g')
            print(f"Solution saved to {txt_path}")
    else:
        with open(txt_path, 'w') as f:
            f.write("Matrix is singular.")
        print(f"Matrix is singular. Saved to {txt_path}")