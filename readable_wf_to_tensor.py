import tensor
import mps
import numpy as np

# mps convention spin left right
# itensor convention (right) spin (left)


def _left_most(rwf_inds):
    _rwf_inds = rwf_inds[0:3:2]
    if rwf_inds[1] == 1:
        _rwf_inds = map(lambda x: max(x-1, 0), _rwf_inds)
    elif rwf_inds[1] == 0:
        _rwf_inds = map(lambda x: x-1, _rwf_inds)
    return _rwf_inds


def _right_most(rwf_inds):
    _rwf_inds = rwf_inds[:2]
    if rwf_inds[2] == 1:
        _rwf_inds = map(lambda x: max(x-1, 0), _rwf_inds)
    elif rwf_inds[2] == 0:
        _rwf_inds = map(lambda x: x-1, _rwf_inds)
    return _rwf_inds

# rwf_inds = [rwf_spin, rwf_left, rwf_right]


def convert_dims(readable_wf_dims, N, rwf_inds=[1, 2, 0]):

    assert sorted(rwf_inds) == [0, 1, 2]

    data_dims = []
    for site, dims in enumerate(readable_wf_dims):
        if site == 0:
            # no left leg
            _rwf_inds = _left_most(rwf_inds)
            data_dims.append([dims[_rwf_inds[0]], 1, dims[_rwf_inds[1]]])
        elif site == N-1:
            # no right leg
            _rwf_inds = _right_most(rwf_inds)
            data_dims.append([dims[_rwf_inds[0]], dims[_rwf_inds[1]], 1])
        else:
            data_dims.append([dims[rwf_inds[0]], dims[rwf_inds[1]],
                              dims[rwf_inds[2]]])
    return data_dims


def convert_inds(readable_wf_inds, site, N, rwf_inds=[1, 2, 0]):
    if site == 0:
        assert len(readable_wf_inds) == 3
        _rwf_inds = _left_most(rwf_inds)
        right_ind = int(readable_wf_inds[_rwf_inds[1]+1]) - 1
        left_ind = 0
        spin_ind = int(readable_wf_inds[_rwf_inds[0]+1]) - 1
    elif site == N-1:
        assert len(readable_wf_inds) == 3
        _rwf_inds = _right_most(rwf_inds)
        right_ind = 0
        left_ind = int(readable_wf_inds[_rwf_inds[1]+1]) - 1
        spin_ind = int(readable_wf_inds[_rwf_inds[0]+1]) - 1
    else:
        assert len(readable_wf_inds) == 4
        right_ind = int(readable_wf_inds[rwf_inds[2]+1]) - 1
        left_ind = int(readable_wf_inds[rwf_inds[1]+1]) - 1
        spin_ind = int(readable_wf_inds[rwf_inds[0]+1]) - 1
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


def convert_readable_wfs(readable_wfs, N, spin_dim, bond_dim,
                         rwf_inds=[1, 2, 0]):
    mps_list = []
    for k, wf in enumerate(readable_wfs):
        readable_wf_dims = [x for x in wf[0].split(';')]
        readable_wf_dims = [[int(y.strip()) for y in x.split(',')]
                            for x in readable_wf_dims]
        data_dims = convert_dims(readable_wf_dims, N, rwf_inds=rwf_inds)
        data = [np.zeros(dims, dtype=complex) for dims in data_dims]
        loc_mps = mps.mps(bond_dim, spin_dim, N)
        for line in wf[1:]:
            site, readable_wf_inds, value = parse_readable_wf_line(line)
            spin, left, right = convert_inds(readable_wf_inds, site, N,
                                             rwf_inds=rwf_inds)
            data[site][spin, left, right] = value
        loc_mps.MPS = add_bonds(data)
        mps_list.append(loc_mps)
    return mps_list
