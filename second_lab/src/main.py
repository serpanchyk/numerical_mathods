import warnings
import time
import numpy as np
from rotation import solve_rotation
from tridiagonal import solve_tridiagonal
from data_handler import read_sole_data, method_evaluation, save_solution

np.seterr(divide='raise', invalid='raise')
warnings.simplefilter('error', RuntimeWarning)

INPUT_ID = 5
DECIMAL_PLACES = 60
IS_EVALUATE = True
METHOD = solve_tridiagonal

def main():
    a, b = read_sole_data(INPUT_ID)
    if a is None or b is None:
        return

    start_time = time.time()
    x, det = METHOD(a, b)
    end_time = time.time()
    execution_time = end_time - start_time

    if x is not None:
        if IS_EVALUATE:
            method_evaluation(a, x, b, det, execution_time, INPUT_ID, DECIMAL_PLACES, METHOD)
        else:
            print(f"Execution time: {execution_time}\n")

    save_solution(x, INPUT_ID, DECIMAL_PLACES)

if __name__ == '__main__':
    main()
