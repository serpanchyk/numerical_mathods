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
            if 'a' in data:
                a = data['a']
                print("Data loaded successfully.")
                return a
            else:
                print("Error: NPZ file does not contain keys 'a' and 'b'.")
                return None

        elif os.path.exists(txt_path):
            print(f"Reading data from file {txt_path}...")
            full_data = np.genfromtxt(txt_path, skip_header=1, delimiter=None, filling_values=0)
            if full_data.ndim < 2:
                print("Error: Not enough data for matrix.")
                return None, None
            a = full_data.copy()
            if a.shape[0] != a.shape[1]:
                print("Error: dimensions do not match.")
                return None
            print("Data loaded successfully.")
            return a

        else:
            print("Error: Data file not found.")
            return None, None

    except Exception as error:
        print(f"Error reading file: {error}")
        return None


def method_evaluation(evals, input_id, method):
    evaluation_dir = Path("..") / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    txt_path = evaluation_dir / f"evaluation{input_id}.txt"

    with open(txt_path, 'w') as f:
        f.write(f"Method name: {getattr(method, '__name__', str(method))}\n")
        f.write(f"Machine error for IEEE 754 standard ε ≈ {2.2e-16}\n")

        for key, val in evals.items():
            f.write(f"{key.capitalize()}: {val}\n")

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
            with open(txt_path, 'w') as f:
                for eigen_pair in x:
                    f.write(f"{eigen_pair[0]:.16f}\n")
                    f.write(f"\n")
                    for val in eigen_pair[1]:
                        f.write(f"{val:.16f}\n")
                    f.write(f"\n")

            print(f"Solution saved to {txt_path}")
    else:
        with open(txt_path, 'w') as f:
            f.write("Matrix is problematic.")