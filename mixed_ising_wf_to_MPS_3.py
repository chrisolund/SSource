from readable_wf_to_tensor import *
import numpy as np
import mps
import pickle
import spinmat
import matplotlib.pyplot as plt

# used for hz hx correlated sweep ssource input

wf_dir = '/home/max/DMRG-itensor/mixed-field-prep/'
hx_name, hxs = 'hx', np.array([1.0])
hz_name, hzs = 'hz', np.linspace(0.025, 0.1, 16)
alt_hzs = np.array([0.000625, 0.00125, 0.00185])
param_space = [hxs, hzs]
alt_param_space = [hxs, alt_hzs]
param_names = [hx_name, hz_name]
param_sizes = [len(hxs), len(hzs)]
alt_param_sizes = [len(hxs), len(alt_hzs)]
pspaces = [param_space, param_space, alt_param_space]
psizes = [param_sizes, param_sizes, alt_param_sizes]


def load_readable_wfs(wf_dir, sim_prefix, target_inds):
    readable_wfs = []
    for i in target_inds:
        fname = wf_dir + sim_prefix + str(i) + '_state_0.txt'
        with open(fname) as f:
            readable_wfs.append([x.strip() for x in f.readlines()])
    return readable_wfs


L8_readable_wfs = load_readable_wfs(wf_dir, 'extN8_', range(len(hzs)))
L16_readable_wfs = load_readable_wfs(wf_dir, 'extN16_', range(len(hzs)))
L32_readable_wfs = load_readable_wfs(wf_dir, 'extN32_', range(len(alt_hzs)))

bond_dim = 64
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
plot_inds = range(len(hzs))
avg_Sz = calc_avg_Sz([L8_mps_list[i] for i in plot_inds], 8)
plt.plot(hzs, np.array(avg_Sz)/2, label='')

# L16 check
plot_inds = range(len(hzs))
avg_Sz = calc_avg_Sz([L16_mps_list[i] for i in plot_inds], 16)
plt.plot(hzs, np.array(avg_Sz)/2, label='')

# L32 check
plot_inds = range(len(alt_hzs))
avg_Sz = calc_avg_Sz([L32_mps_list[i] for i in plot_inds], 32)
plt.plot(alt_hzs, np.array(avg_Sz)/2, label='')

L8_tensor_list = [loc_mps.MPS for loc_mps in L8_mps_list]
L16_tensor_list = [loc_mps.MPS for loc_mps in L16_mps_list]
L32_tensor_list = [loc_mps.MPS for loc_mps in L32_mps_list]

with open('pickles/ext_L8_tensor_data.p', 'w') as f:
    pickle.dump(L8_tensor_list, f)

with open('pickles/ext_L16_tensor_data.p', 'w') as f:
    pickle.dump(L16_tensor_list, f)

with open('pickles/ext_L32_tensor_data.p', 'w') as f:
    pickle.dump(L32_tensor_list, f)
