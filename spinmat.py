import numpy as np


# sigma functions

sigma_x =  np.array([[0, 1],[1, 0]])
sigma_y =  np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])
id_mat = np.array([[1, 0],[0, 1]])
zero_mat = np.array([[0, 0],[0, 0]])
# spin ladder operators

sigma_plus = np.array([[0, 1],[0, 0]])
sigma_minus = np.array([[0, 0],[1, 0]])


