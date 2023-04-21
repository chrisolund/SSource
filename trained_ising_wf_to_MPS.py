import numpy as np
import mps
import tensor
import pickle
from toolbox import slicer
import spinmat
import matplotlib.pyplot as plt


wf_dir = '/home/max/DMRG-itensor/training-field-prep/'

hx_name, hxs = 'hx', np.logspace(np.log(0.5)/np.log(10), np.log(2)/np.log(10),
                                 41)
hz_name, hzs = 'hz', np.logspace(-2, 0, 5)
param_space = [hxs, hzs]
param_names = [hx_name, hz_name]
param_sizes = [len(hxs), len(hzs)]
raw_inds = range(param_sizes[0] * param_sizes[1])
ind_slicer = slicer.Slicer(raw_inds, param_names, param_sizes)
hx_inds = range(len(hxs))
hz_inds = range(len(hzs))
target_inds, _ = ind_slicer.slice(hx=hx_inds, hz=hz_inds)


def load_readable_wfs(wf_dir, sim_prefix, target_inds):
    readable_wfs = []
    for i in target_inds:
        fname = wf_dir + sim_prefix + str(i) + '_state_0.txt'
        with open(fname) as f:
            readable_wfs.append([x.strip() for x in f.readlines()])
    return readable_wfs


L8_readable_wfs = load_readable_wfs(wf_dir, 'N8_', target_inds)
L16_readable_wfs = load_readable_wfs(wf_dir, 'N16_', target_inds)
L32_readable_wfs = load_readable_wfs(wf_dir, 'N32_', target_inds)


# mps convention spin left right
# itensor convention (right) spin (left)


def convert_dims(readable_wf_dims, N):
    data_dims = []
    for site, dims in enumerate(readable_wf_dims):
        if site == 0:
            # no left leg
            data_dims.append([dims[1], 1, dims[0]])
        elif site == N-1:
            # no right leg
            data_dims.append([dims[0], dims[1], 1])
        else:
            data_dims.append([dims[1], dims[2], dims[0]])
    return data_dims


def convert_inds(readable_wf_inds, site, N):
    if site == 0:
        assert len(readable_wf_inds) == 3
        right_ind = int(readable_wf_inds[1]) - 1
        left_ind = 0
        spin_ind = int(readable_wf_inds[2]) - 1
    elif site == N-1:
        assert len(readable_wf_inds) == 3
        right_ind = 0
        left_ind = int(readable_wf_inds[2]) - 1
        spin_ind = int(readable_wf_inds[1]) - 1
    else:
        assert len(readable_wf_inds) == 4
        right_ind = int(readable_wf_inds[1]) - 1
        left_ind = int(readable_wf_inds[3]) - 1
        spin_ind = int(readable_wf_inds[2]) - 1
    return spin_ind, left_ind, right_ind


def add_bonds(data):
    tensor_data = []
    for site, array in enumerate(data):
        shape = array.shape
        bonds = [tensor.Bond(mps.spin_bond_name(site), shape[0], False),
                 tensor.Bond(mps.mps_bond_left_name(site), shape[1], False),
                 tensor.Bond(mps.mps_bond_right_name(site), shape[2], False)]
        tensor_data.append(tensor.Tensor(array, bonds))
    return tensor_data


def parse_readable_wf_line(line):
    readable_wf_inds, value = line.split(';')
    readable_wf_inds = readable_wf_inds.split(',')
    value = np.dot(np.array([float(x) for x in value.split(',')]),
                   [1.0, 1j])
    site = int(readable_wf_inds[0])-1
    return site, readable_wf_inds, value


def convert_readable_wfs(readable_wfs, N, spin_dim, bond_dim):
    mps_list = []
    for k, wf in enumerate(readable_wfs):
        readable_wf_dims = [x for x in wf[0].split(';')]
        readable_wf_dims = [[int(y.strip()) for y in x.split(',')]
                            for x in readable_wf_dims]
        data_dims = convert_dims(readable_wf_dims, N)
        data = [np.zeros(dims, dtype=complex) for dims in data_dims]
        loc_mps = mps.mps(bond_dim, spin_dim, N)
        for line in wf[1:]:
            site, readable_wf_inds, value = parse_readable_wf_line(line)
            spin, left, right = convert_inds(readable_wf_inds, site, N)
            data[site][spin, left, right] = value
        loc_mps.MPS = add_bonds(data)
        mps_list.append(loc_mps)
    return mps_list


bond_dim = 8
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
hx_name, hxs = 'hx', np.logspace(np.log(0.5)/np.log(10), np.log(2)/np.log(10),
                                 41)
# L=8 check
for hz_ind in hz_inds:
    plot_inds, _ = ind_slicer.slice(hx=hx_inds, hz=[hz_ind])
    avg_Sz = calc_avg_Sz([L8_mps_list[i] for i in plot_inds], 8)
    plt.plot(hxs, np.array(avg_Sz)/2, label='hz='+str(hzs[hz_ind])[:4])
    plt.legend()

# L16 check
for hz_ind in hz_inds:
    plot_inds, _ = ind_slicer.slice(hx=hx_inds, hz=[hz_ind])
    avg_Sz = calc_avg_Sz([L16_mps_list[i] for i in plot_inds], 16)
    plt.plot(hxs, np.array(avg_Sz)/2, label='hz='+str(hzs[hz_ind])[:4])
    plt.legend()

# L32 check
for hz_ind in hz_inds:
    plot_inds, _ = ind_slicer.slice(hx=hx_inds, hz=[hz_ind])
    avg_Sz = calc_avg_Sz([L32_mps_list[i] for i in plot_inds], 32)
    plt.plot(hxs, np.array(avg_Sz)/2, label='hz='+str(hzs[hz_ind])[:4])
    plt.legend()

L8_tensor_list = [loc_mps.MPS for loc_mps in L8_mps_list]
L16_tensor_list = [loc_mps.MPS for loc_mps in L16_mps_list]
L32_tensor_list = [loc_mps.MPS for loc_mps in L32_mps_list]

with open('hz_L8_tensor_data.p', 'w') as f:
    pickle.dump(L8_tensor_list, f)

with open('hz_L16_tensor_data.p', 'w') as f:
    pickle.dump(L16_tensor_list, f)

with open('hz_L32_tensor_data.p', 'w') as f:
    pickle.dump(L32_tensor_list, f)
