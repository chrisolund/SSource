from readable_wf_to_tensor import *
import numpy as np
import mps
import pickle
import spinmat
import matplotlib.pyplot as plt

wf_dir = '/home/max/DMRG-itensor/parity-guarded-cat-prep/'

hx_name, hxs = 'hx', np.logspace(np.log(0.2)/np.log(10), np.log(2)/np.log(10),
                                 41)
Jx_name, Jxs = 'Jx', [0.1]
param_space = [hxs, Jxs]
param_names = [hx_name, Jx_name]
param_sizes = [len(hxs), len(Jxs)]

Ns = [8, 16, 32, 64]
sim_prefix_list = ['N8_Jx_', 'N16_Jx_', 'N32_Jx_', 'N64_Jx_']
target_inds = range(len(hxs))


def load_readable_wfs(wf_dir, sim_prefix, target_inds):
    readable_wfs = []
    for i in target_inds:
        fname = wf_dir + sim_prefix + str(i) + '_state_0.txt'
        with open(fname) as f:
            readable_wfs.append([x.strip() for x in f.readlines()])
    return readable_wfs


readable_wfs = [load_readable_wfs(wf_dir, prefix, target_inds) for
                prefix in sim_prefix_list]

bond_dim = 100
spin_dim = 2

mps_lists = [convert_readable_wfs(r_wf, N, spin_dim, bond_dim,
                                  rwf_inds=[0, 2, 1])
             for r_wf, N in zip(readable_wfs, Ns)]


def calc_avg_Sz(mps_list, N):
    avg_Sz = []
    for loc_mps in mps_list:
        exp_vals = [mps.MPS1Point(i, spinmat.sigma_z, N, loc_mps)
                    for i in range(N)]
        avg_Sz.append(np.mean(exp_vals))
    return avg_Sz


# rough checks for errors: expectation values

Jx_inds = [0]

# <Sz> Check
for mps_list, N in zip(mps_lists, Ns):
    for Jx_ind in Jx_inds:
        plot_inds = target_inds
        avg_Sz = calc_avg_Sz([mps_list[i] for i in plot_inds], N)
        plt.plot(hxs, np.array(avg_Sz)/2, label='Jx=0')
        plt.ylim([-0.05, 0.05])
        plt.legend()

tensor_lists = [[loc_mps.MPS for loc_mps in mps_list] for mps_list
                in mps_lists]
names = ['parity_guarded_Jx_L8', 'parity_guarded_Jx_L16',
         'parity_guarded_Jx_L32', 'parity_guarded_Jx_L64']
for tensor_list, name in zip(tensor_lists, names):
    with open('pickles/{}_tensor_data.p'.format(name), 'w') as f:
        pickle.dump(tensor_list, f)
