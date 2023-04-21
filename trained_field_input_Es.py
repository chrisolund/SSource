import os
import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
from toolbox import figutils

# GLOBAL CONSTANTS
sweeps = 1000

# =================== FUNCTIONS ===================
s = 'Etf_L0=16_layer=1_hx=0.554784736034_Jx=0.0_hz_tf=0.01_sweeps=1000_run=7.csv'


def get_value(name, param):
    pos = name.find(param)
    pos = pos + len(param) + 1
    digit = name[pos]
    val = ''
    while digit.isdigit() or digit == '.':
        val = val + digit
        pos += 1
        digit = name[pos]
    return float(val)


get_value(s, 'hx')
get_value(s, 'hz_tf')

files = os.walk('Data/TrainingFieldL0/').next()[2]
energies_dict = {}
for f in files:
    if 'L0=8' in f and f[-4:] == '.csv' and 'sweeps=1000' in f:
        key = (get_value(f, 'hx'), get_value(f, 'hz_tf'))
        es = np.loadtxt('Data/TrainingFieldL0/' + f)
        if key not in energies_dict:
            energies_dict[key] = []
        energies_dict[key].append(np.min(es))

best_energies_dict = {}
for key in energies_dict:
    best_energies_dict[key] = np.min(energies_dict[key])

hxs = np.sort(np.unique([key[0] for key in best_energies_dict]))
hzs = np.sort(np.unique([key[1] for key in best_energies_dict]))


def EExactTFIM(h, J, L):
    CC = sparse.diags(h*np.ones(L)).toarray()+sparse.diags(J*np.ones(L-1), -1)
    H2 = CC.dot(CC.T)
    return -np.sum(np.abs(linalg.eigh(H2, eigvals_only=True))**.5)


exact_energies = [EExactTFIM(hx, 1.0, 16) for hx in hxs]

len(energies_dict[energies_dict.keys()[0]])

energies_list = []
for hz in hzs:
    energies_list.append(np.array([best_energies_dict[(hx, hz)]
                                   for hx in hxs]))

energy_errors = []
for energies in energies_list:
    errors = 1 - energies/exact_energies
    energy_errors.append(errors)

figutils.make_figure(hxs, energy_errors, 'hx', 'Error',
                     [str(hz)[:4] for hz in hzs])
