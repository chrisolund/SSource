from readable_wf_to_tensor import *
import numpy as np
import mps
import pickle
from toolbox import slicer
import spinmat
import matplotlib.pyplot as plt

wf_dir = '/home/max/DMRG-itensor/near-critical-prep/'

hx_name, hxs_L = 'hx', np.linspace(0.9, 1.1, 21)
hxs_2L = (hxs_L-1)/2 + 1
Jx_name, Jxs = 'Jx', [0.0]
Jx_inds = [0]
param_space_L = [hxs_L, Jxs]
param_space_2L = [hxs_2L, Jxs]
param_space_list = [param_space_L, param_space_2L]
param_names = [hx_name, Jx_name]
param_sizes = [len(hxs_L), len(Jxs)]

opnames = ['SvN', 'Sz', 'Sz_Sz']
sim_prefix_list = ['N8_', 'N16_']

target_inds = range(len(hxs_L))


def load_readable_wfs(wf_dir, sim_prefix, target_inds):
    readable_wfs = []
    for i in target_inds:
        fname = wf_dir + sim_prefix + str(i) + '_state_0.txt'
        with open(fname) as f:
            readable_wfs.append([x.strip() for x in f.readlines()])
    return readable_wfs


L8_readable_wfs = load_readable_wfs(wf_dir, 'N8_', target_inds)
L16_readable_wfs = load_readable_wfs(wf_dir, 'N16_', target_inds)

bond_dim = 100
spin_dim = 2

L8_mps_list = convert_readable_wfs(L8_readable_wfs, 8, spin_dim, bond_dim,
                                   rwf_inds=[0, 2, 1])
L16_mps_list = convert_readable_wfs(L16_readable_wfs, 16, spin_dim, bond_dim,
                                    rwf_inds=[0, 2, 1])


def calc_avg_Sz(mps_list, N):
    avg_Sz = []
    for loc_mps in mps_list:
        exp_vals = [mps.MPS1Point(i, spinmat.sigma_z, N, loc_mps)
                    for i in range(N)]
        avg_Sz.append(np.mean(exp_vals))
    return avg_Sz


# rough checks for errors: expectation values

# L8 check
for Jx_ind in Jx_inds:
    plot_inds = target_inds
    avg_Sz = calc_avg_Sz([L8_mps_list[i] for i in plot_inds], 8)
    plt.plot(hxs_L, np.array(avg_Sz)/2, label='Jx=0')
    plt.ylim([-0.05, 0.05])
    plt.legend()

# L16 check
for Jx_ind in Jx_inds:
    plot_inds = target_inds
    avg_Sz = calc_avg_Sz([L16_mps_list[i] for i in plot_inds], 16)
    plt.plot(hxs_2L, np.array(avg_Sz)/2, label='Jx=0')
    plt.ylim([-0.05, 0.05])
    plt.legend()

L8_tensor_list = [loc_mps.MPS for loc_mps in L8_mps_list]
L16_tensor_list = [loc_mps.MPS for loc_mps in L16_mps_list]

with open('pickles/near_crit_L8_tensor_data.p', 'w') as f:
    pickle.dump(L8_tensor_list, f)

with open('pickles/near_crit_L16_tensor_data.p', 'w') as f:
    pickle.dump(L16_tensor_list, f)
