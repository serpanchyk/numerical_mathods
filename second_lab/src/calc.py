T = 6.966634273529053
T_b = 1236.2821786403656
n = 1000
n_b = 10000
O = lambda a: 4/3 * a ** 3 + 3 * a ** 2 - 16/3 * a
tau = T / O(n)
tau_b = T_b / O(n_b)
print(tau * O(n_b) / 60 / 5)

