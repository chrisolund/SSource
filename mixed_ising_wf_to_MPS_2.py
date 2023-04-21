from readable_wf_to_tensor import *
import numpy as np
import mps
import pickle
import spinmat
import matplotlib.pyplot as plt

# used for hz hx correlated sweep ssource input

wf_dir = '/home/max/DMRG-itensor/mixed-field-prep/'
base = np.linspace(-0.2, 0.2, 21)
hxs = 1.0 + base
hzs = 0.1 * base
param_space = list(zip(hxs, hzs))
param_names = ['hx_hz']
param_sizes = [len(param_space)]
target_inds = range(param_sizes[0])


def load_readable_wfs(wf_dir, sim_prefix, target_inds):
    readable_wfs = []
    for i in target_inds:
        fname = wf_dir + sim_prefix + str(i) + '_state_0.txt'
        with open(fname) as f:
            readable_wfs.append([x.strip() for x in f.readlines()])
    return readable_wfs


L8_readable_wfs = load_readable_wfs(wf_dir, 'xzN8_', target_inds)
L16_readable_wfs = load_readable_wfs(wf_dir, 'xzN16_', target_inds)
L32_readable_wfs = load_readable_wfs(wf_dir, 'xzN32_', target_inds)

bond_dim = 32
spin_dim = 2

L8_mps_list = convert_readable_wfs(L8_readable_wfs, 8, spin_dim, bond_dim)
L16_mps_list = convert_readable_wfs(L16_readable_wfs, 16, spin_dim, bond_dim)
L32_mps_list = convert_readable_wfs(L32_readable_wfs, 32, spin_dim, bond_dim)

def calc_avg_Sz(mps_list, N):
    avg_Sz = []
    for loc_mps in mps_list:
        exp_vals = [mps.MPS1Point(i, spinmat.sigma_z, N, loc_mps)
                    for i in range(N)]
        avg_Sz.append(np.mean(exp_vals))
    return avg_Sz


# rough checks for errors: expectation values

# L=8 check
plot_inds = target_inds
avg_Sz = calc_avg_Sz([L8_mps_list[i] for i in plot_inds], 8)
plt.plot(hzs, np.array(avg_Sz)/2, label='')

# L16 check
plot_inds = target_inds
avg_Sz = calc_avg_Sz([L16_mps_list[i] for i in plot_inds], 16)
plt.plot(hzs, np.array(avg_Sz)/2, label='')

# L32 check
plot_inds = target_inds
avg_Sz = calc_avg_Sz([L32_mps_list[i] for i in plot_inds], 32)
plt.plot(hzs, np.array(avg_Sz)/2, label='')

L8_tensor_list = [loc_mps.MPS for loc_mps in L8_mps_list]
L16_tensor_list = [loc_mps.MPS for loc_mps in L16_mps_list]
L32_tensor_list = [loc_mps.MPS for loc_mps in L32_mps_list]

with open('pickles/xz_L8_tensor_data.p', 'w') as f:
    pickle.dump(L8_tensor_list, f)

with open('pickles/xz_L16_tensor_data.p', 'w') as f:
    pickle.dump(L16_tensor_list, f)

with open('pickles/xz_L32_tensor_data.p', 'w') as f:
    pickle.dump(L32_tensor_list, f)
