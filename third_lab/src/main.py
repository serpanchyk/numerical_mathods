import warnings
import time
import numpy as np
from jacobi import solve_jacobi
from data_handler import read_sole_data, method_evaluation, save_solution

np.seterr(divide='raise', invalid='raise')
warnings.simplefilter('error', RuntimeWarning)

INPUT_ID = 2
DECIMAL_PLACES = 60
METHOD = solve_jacobi
EPSILON = 10e-15

EVAL_OPTIONS = {
    'cond': True,
    'spectral_radius': True,
    'norms': True,
    'residual': True,
    'benchmark': True,
    'iterations': True,
    'a_priori': True,
    'stability_error': True
}

def main():
    a, b = read_sole_data(INPUT_ID)
    if a is None or b is None:
        return

    start_time = time.time()
    x, evals = METHOD(a, b, eps=EPSILON, eval_options=EVAL_OPTIONS)
    end_time = time.time()
    execution_time = end_time - start_time

    method_evaluation(a, x, b, evals, EVAL_OPTIONS, execution_time, INPUT_ID, METHOD)

    save_solution(x, INPUT_ID, DECIMAL_PLACES)

if __name__ == '__main__':
    main()
