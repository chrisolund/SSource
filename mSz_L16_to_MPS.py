import numpy as np
import mps
import tensor
import pickle

mSz_wf_dir = '/home/max/DMRG-itensor/L16-maxSz/'

# mps convention spin left right
# itensor convention (right) spin (left)

readable_wfs = []
for i in list(range(41)):
    fname = mSz_wf_dir + 'mSz' + str(i) + '_state_0.txt'
    with open(fname) as f:
        readable_wfs.append([x.strip() for x in f.readlines()])

mSz_mps = []
bond_dim = 4
spin_d = 2
N = 16
for k, wf in enumerate(readable_wfs):
    readable_wf_dims = [x for x in wf[0].split(';')]
    readable_wf_dims = [[int(y.strip()) for y in x.split(',')]
                        for x in readable_wf_dims]
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
    # print(data_dims)
    data = [np.zeros(dims, dtype=complex) for dims in data_dims]
    loc_mps = mps.mps(bond_dim, spin_d, N)  # spin-half, chi=8 fixed by DMRG
    for line in wf[1:]:
        indices, value = line.split(';')
        indices = indices.split(',')
        value = np.dot(np.array([float(x) for x in value.split(',')]),
                       [1.0, 1j])
        site = int(indices[0])-1
        if site == 0:
            assert len(indices) == 3
            right_ind = int(indices[1]) - 1
            left_ind = 0
            spin_ind = int(indices[2]) - 1
        elif site == N-1:
            assert len(indices) == 3
            right_ind = 0
            left_ind = int(indices[2]) - 1
            spin_ind = int(indices[1]) - 1
        else:
            assert len(indices) == 4
            right_ind = int(indices[1]) - 1
            left_ind = int(indices[3]) - 1
            spin_ind = int(indices[2]) - 1
        data[site][spin_ind, left_ind, right_ind] = value
    loc_mps.MPS = data  # wrong - need to add bonds
    mSz_mps.append(loc_mps)

# \/ MUTATES!
tensor_data_list = []
for loc_mps in mSz_mps:
    tensor_data = []
    for site, array in enumerate(loc_mps.MPS):
        shape = array.shape
        bonds = [tensor.Bond(mps.spin_bond_name(site), shape[0], False),
                 tensor.Bond(mps.mps_bond_left_name(site), shape[1], False),
                 tensor.Bond(mps.mps_bond_right_name(site), shape[2], False)]
        tensor_data.append(tensor.Tensor(array, bonds))
    loc_mps.MPS = tensor_data
    tensor_data_list.append(tensor_data)

with open('mSz_L16_tensor_data_list.p', 'w') as f:
    pickle.dump(tensor_data_list, f)
