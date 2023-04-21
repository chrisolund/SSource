from readable_wf_to_tensor import *
import numpy as np
import mps
import pickle
from toolbox import slicer
import spinmat
import matplotlib.pyplot as plt

# used for both off critical ssource input and critical ssource input
# off critical sim prefix: N8_, N16_ etc..
# critical sim prefix: cN8_, cN16_ etc...
# off critical pickle filename: mixed_L8_tensor_data etc..
# critical pickle filename: cmixed_L8_tensor_data etc..

wf_dir = '/home/max/DMRG-itensor/cat-like-prep/'

# # off critical params
# hx_name, hxs = 'hx', np.array([0.8, 0.4])
# hz_name, hzs = 'hz', np.linspace(-0.5, 0.5, 21)
# param_space = [hxs, hzs]
# param_names = [hx_name, hz_name]
# param_sizes = [len(hxs), len(hzs)]

# criticical params
hx_name, hxs = 'hx', np.logspace(np.log(0.5)/np.log(10), np.log(1)/np.log(10),
                                 21)
Jx_name, Jxs = 'Jx', [0]
param_space = [hxs, Jxs]
param_names = [hx_name, Jx_name]
param_sizes = [len(hxs), len(Jxs)]

raw_inds = range(param_sizes[0] * param_sizes[1])
ind_slicer = slicer.Slicer(raw_inds, param_names, param_sizes)
hx_inds = range(len(hxs))
Jx_inds = range(len(Jxs))
target_inds, _ = ind_slicer.slice(hx=hx_inds, Jx=Jx_inds)
target_inds


def load_readable_wfs(wf_dir, sim_prefix, target_inds):
    readable_wfs = []
    for i in target_inds:
        fname = wf_dir + sim_prefix + str(i) + '_state_0.txt'
        with open(fname) as f:
            readable_wfs.append([x.strip() for x in f.readlines()])
    return readable_wfs


L16_readable_wfs = load_readable_wfs(wf_dir, 'N16_', target_inds)
L32_readable_wfs = load_readable_wfs(wf_dir, 'N32_', target_inds)

bond_dim = 100
spin_dim = 2

L16_mps_list = convert_readable_wfs(L16_readable_wfs, 16, spin_dim, bond_dim,
                                    rwf_inds=[0, 2, 1])
L32_mps_list = convert_readable_wfs(L32_readable_wfs, 32, spin_dim, bond_dim,
                                    rwf_inds=[0, 2, 1])


def calc_avg_Sz(mps_list, N):
    avg_Sz = []
    for loc_mps in mps_list:
        exp_vals = [mps.MPS1Point(i, spinmat.sigma_z, N, loc_mps)
                    for i in range(N)]
        avg_Sz.append(np.mean(exp_vals))
    return avg_Sz


# rough checks for errors: expectation values

# L16 check
for Jx_ind in Jx_inds:
    plot_inds, _ = ind_slicer.slice(hx=hx_inds, Jx=[Jx_ind])
    avg_Sz = calc_avg_Sz([L16_mps_list[i] for i in plot_inds], 16)
    plt.plot(hxs, np.array(avg_Sz)/2, label='Jx=0')
    plt.legend()

# L32 check
for Jx_ind in Jx_inds:
    plot_inds, _ = ind_slicer.slice(hx=hx_inds, Jx=[Jx_ind])
    avg_Sz = calc_avg_Sz([L32_mps_list[i] for i in plot_inds], 32)
    plt.plot(hxs, np.array(avg_Sz)/2, label='Jx=0')
    plt.legend()

L16_tensor_list = [loc_mps.MPS for loc_mps in L16_mps_list]
L32_tensor_list = [loc_mps.MPS for loc_mps in L32_mps_list]

with open('pickles/cat_like_L16_tensor_data.p', 'w') as f:
    pickle.dump(L16_tensor_list, f)

with open('pickles/cat_like_L32_tensor_data.p', 'w') as f:
    pickle.dump(L32_tensor_list, f)
