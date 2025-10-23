import numpy as np
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

def method_evaluation(a, x, b, evals, eval_options, execution_time, input_id, method):
    evaluation_dir = Path("..") / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    txt_path = evaluation_dir / f"evaluation{input_id}.txt"

    with open(txt_path, 'w') as f:
        f.write(f"Method name: {getattr(method, '__name__', str(method))}\n")
        f.write(f"Machine error for IEEE 754 standard ε ≈ {2.2e-16}\n")
        f.write(f"Matrix size: {evals.get('matrix_size', a.shape[0])}\n")
        f.write(f"Execution time: {execution_time}\n")
        f.write(f"Iterations: {evals.get('iterations', 'N/A')}\n")
        f.write(f"Converged: {evals.get('converged', 'N/A')}\n")
        f.write(f"Epsilon used: {evals.get('epsilon', 'N/A')}\n\n")

        scalar_keys = ['spectral_radius', 'cond', 'residual_norm', 'relative_residual_norm',
                       'a_priori_iterations_estimate', 'diagonal_dominance']
        for key in scalar_keys:
            if key in evals:
                f.write(f"{key}: {evals[key]}\n")

        if 'norms' in evals and isinstance(evals['norms'], dict):
            f.write("\nNorms of iteration matrix C:\n")
            for nkey, nval in evals['norms'].items():
                f.write(f"  {nkey}: {nval}\n")

        if eval_options.get('benchmark', True):
            try:
                x_benchmark = np.linalg.solve(a, b)
                abs_err = float(np.linalg.norm(x - x_benchmark))
                rel_err = float(abs_err / np.linalg.norm(x_benchmark)) if np.linalg.norm(x_benchmark) != 0 else None
                f.write("\nBenchmark solution available.\n")
                f.write(f"Absolute error vs benchmark: {abs_err}\n")
                f.write(f"Relative error vs benchmark: {rel_err}\n")
            except Exception:
                f.write("\nBenchmark solution could not be computed (singular or unstable matrix).\n")

        if eval_options.get('stability_error', True) and x is not None:
            try:
                epsilon = 1e-8
                b_perturbed = b + epsilon * np.random.randn(*b.shape)
                x_perturbed, _ = method(a, b_perturbed)
                stability_error = float(np.linalg.norm(x_perturbed - x))
                f.write(f"Stability error (solving perturbed system): {stability_error}\n")
            except Exception:
                f.write("Stability error: Could not be calculated due to singularity or instability.\n")

        f.write("\nEnd of evaluation.\n")

    print(f"Evaluation complete. Results saved to {txt_path}")

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