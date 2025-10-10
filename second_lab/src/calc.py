# T = 6.966634273529053
# T_b = 1236.2821786403656
# n = 1000
# n_b = 10000
# O = lambda a: 4/3 * a ** 3 + 3 * a ** 2 - 16/3 * a
# tau = T / O(n)
# tau_b = T_b / O(n_b)
# print(tau * O(n_b) / 60 / 5)
import numpy as np

# import numpy as np
#
# npz_path_input = "../inputs/input6.npz"
#
# data_1 = np.load(npz_path_input)
# a = data_1['a']
# b = data_1['b']
#
# npz_output = "../outputs/output6.npz"
#
# data_2 = np.load(npz_output)
# x = data_2['x']
#
# residual_norm = np.linalg.norm(a @ x - b)
# relative_residual_norm = residual_norm / np.linalg.norm(a) / np.linalg.norm(x)
# print(f"Residual norm of solution: {residual_norm}\n")
# print(f"Relative residual norm: {relative_residual_norm}\n")


data = np.load("../inputs/input5.npz")
a = data["a"]
b = data["b"]
a[0, 0] = 1
a[-1, -1] = 1
n = 10000
np.savez_compressed("../inputs/input5.npz", a=a, b=b)

