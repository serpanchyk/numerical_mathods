import warnings
import time
import numpy as np
from gaussian import solve_gaussian
from data_handler import read_sole_data, method_evaluation, save_solution

np.seterr(divide='raise', invalid='raise')
warnings.simplefilter('error', RuntimeWarning)
input_id = 6
decimal_places = 60
pivoting_type = "partial"
is_evaluate = True

def main():
    a, b = read_sole_data(input_id)
    if a is None or b is None:
        return

    start_time = time.time()
    x, det = solve_gaussian(a, b, type=pivoting_type, det_evaluation=is_evaluate)
    end_time = time.time()
    execution_time = end_time - start_time

    if x is not None:
        if is_evaluate:
            method_evaluation(a, x, b, det, execution_time, input_id, pivoting_type, decimal_places)
        else:
            print(f"Execution time: {execution_time}\n")

    save_solution(x, input_id, pivoting_type, decimal_places)

if __name__ == '__main__':
    main()