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

wf_dir = '/home/max/DMRG-itensor/mixed-field-prep/'

# # off critical params
# hx_name, hxs = 'hx', np.array([0.8, 0.4])
# hz_name, hzs = 'hz', np.linspace(-0.5, 0.5, 21)
# param_space = [hxs, hzs]
# param_names = [hx_name, hz_name]
# param_sizes = [len(hxs), len(hzs)]

# criticical params
hx_name, hxs = 'hx', np.array([1.0])
hz_name, hzs = 'hz', np.linspace(-0.02, 0.02, 21)
param_space = [hxs, hzs]
param_names = [hx_name, hz_name]
param_sizes = [len(hxs), len(hzs)]

raw_inds = range(param_sizes[0] * param_sizes[1])
ind_slicer = slicer.Slicer(raw_inds, param_names, param_sizes)
hx_inds = range(len(hxs))
hz_inds = range(len(hzs))
target_inds, _ = ind_slicer.slice(hx=hx_inds, hz=hz_inds)
target_inds


def load_readable_wfs(wf_dir, sim_prefix, target_inds):
    readable_wfs = []
    for i in target_inds:
        fname = wf_dir + sim_prefix + str(i) + '_state_0.txt'
        with open(fname) as f:
            readable_wfs.append([x.strip() for x in f.readlines()])
    return readable_wfs


L8_readable_wfs = load_readable_wfs(wf_dir, 'cN8_', target_inds)
L16_readable_wfs = load_readable_wfs(wf_dir, 'cN16_', target_inds)
L32_readable_wfs = load_readable_wfs(wf_dir, 'cN32_', target_inds)

bond_dim = 8
spin_dim = 2

L8_mps_list = convert_readable_wfs(L8_readable_wfs, 8, spin_dim, bond_dim)
L16_mps_list = convert_readable_wfs(L16_readable_wfs, 16, spin_dim, bond_dim)
L32_mps_list = convert_readable_wfs(L32_readable_wfs, 32, spin_dim, bond_dim)

for x in L16_mps_list[-1].MPS:
    print(x.data.shape)

def calc_avg_Sz(mps_list, N):
    avg_Sz = []
    for loc_mps in mps_list:
        exp_vals = [mps.MPS1Point(i, spinmat.sigma_z, N, loc_mps)
                    for i in range(N)]
        avg_Sz.append(np.mean(exp_vals))
    return avg_Sz


# rough checks for errors: expectation values

# L=8 check
for hx_ind in hx_inds:
    plot_inds, _ = ind_slicer.slice(hz=hz_inds, hx=[hx_ind])
    avg_Sz = calc_avg_Sz([L8_mps_list[i] for i in plot_inds], 8)
    plt.plot(hzs, np.array(avg_Sz)/2, label='hx='+str(hxs[hx_ind])[:4])
    plt.legend()

# L16 check
for hx_ind in hx_inds:
    plot_inds, _ = ind_slicer.slice(hz=hz_inds, hx=[hx_ind])
    avg_Sz = calc_avg_Sz([L16_mps_list[i] for i in plot_inds], 16)
    plt.plot(hzs, np.array(avg_Sz)/2, label='hx='+str(hxs[hx_ind])[:4])
    plt.legend()

# L32 check
for hx_ind in hx_inds:
    plot_inds, _ = ind_slicer.slice(hz=hz_inds, hx=[hx_ind])
    avg_Sz = calc_avg_Sz([L32_mps_list[i] for i in plot_inds], 32)
    plt.plot(hzs, np.array(avg_Sz)/2, label='hx='+str(hxs[hx_ind])[:4])
    plt.legend()

L8_tensor_list = [loc_mps.MPS for loc_mps in L8_mps_list]
L16_tensor_list = [loc_mps.MPS for loc_mps in L16_mps_list]
L32_tensor_list = [loc_mps.MPS for loc_mps in L32_mps_list]

with open('pickles/cmixed_L8_tensor_data.p', 'w') as f:
    pickle.dump(L8_tensor_list, f)

with open('pickles/cmixed_L16_tensor_data.p', 'w') as f:
    pickle.dump(L16_tensor_list, f)

with open('pickles/cmixed_L32_tensor_data.p', 'w') as f:
    pickle.dump(L32_tensor_list, f)
