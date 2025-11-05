import warnings
import numpy as np
from fourth_lab.src.RQI import solve_inverse_iteration
from jacobi import solve_jacobi
from data_handler import read_sole_data, method_evaluation, save_solution

np.seterr(divide='raise', invalid='raise')
warnings.simplefilter('error', RuntimeWarning)

INPUT_ID = 2
DECIMAL_PLACES = 60
METHOD = solve_inverse_iteration
EPSILON = 1e-6

EVAL_OPTIONS = {
    'cond': True,
    'benchmark': True,
    'iterations': True,
}

def main():
    a = read_sole_data(INPUT_ID)
    if a is None:
        return

    eigen_pair, evals = METHOD(a, tol=EPSILON, eval_options=EVAL_OPTIONS)

    method_evaluation(evals, INPUT_ID, METHOD)

    save_solution(eigen_pair, INPUT_ID, DECIMAL_PLACES)

if __name__ == '__main__':
    main()
