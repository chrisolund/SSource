import os
import numpy as np
import RGMPS_shift
import pickle
from toolbox import figutils


# generalizing one time script for computing expvals of RGMPS on mSz L=8 input
# states


# GLOBAL CONSTANTS
sweeps = 1000

# =================== FUNCTIONS ===================


def parse_name(name):
    # pname: element, L0, hx, Jx, sweeps, run
    pname = name.split('_')
    pname = filter(lambda x: ('layer' not in x and 'Lf' not in x), pname)
    for i, entry in enumerate(pname[1:]):
        key, val = entry.split('=')
        if key in ['hx', 'Jx']:
            val = float(val)
        elif key in ['sweeps', 'L0']:
            val = int(val)
        elif key == 'run':
            val = int(val.split('.')[0])
        else:
            print(key, 'not found')
        pname[i + 1] = val
    return pname


def construct_name(pname):
    # doesn't construct energy name properly
    # pname: element, L0, hx, Jx, sweeps, run
    keys = ['hx', 'Jx', 'sweeps', 'run']
    L0 = pname[1]
    Lf = int(2 * L0)  # only single layer
    name = pname[0] + '_L0=' + str(L0) + '_Lf=' + str(Lf)
    for i, entry in enumerate(pname[2:]):
        name += '_' + keys[i] + '=' + str(entry)
    name += '.p'
    return name


def find_best_runs(folder, target_L0, target_runs=None):
    best_runs = {}
    fnames = list(os.walk(folder).next()[2])
    for fname in fnames:
        if '.csv' in fname:
            _, L0, hx, Jx, loc_sweeps, run = parse_name(fname)
            compare = (L0 == target_L0)
            compare *= (loc_sweeps == sweeps)
            if target_runs is not None:
                compare *= (run in target_runs)
            if compare:
                with open(folder + fname) as file:
                    Es = np.loadtxt(file)
                    key = (hx, Jx)
                    candidate = Es[-1]
                    if key not in best_runs:
                        best_runs[key] = (run, candidate)
                    else:
                        current = best_runs[key][1]
                        if candidate < current:
                            best_runs[key] = (run, candidate)
    return best_runs


def copy_best_RGMPS_sh(fname, orig_folder, final_folder, target_L0, best_runs):
    with open(fname, 'w') as file:
        file.write('#!/bin/bash\n')
        for key, val in best_runs.iteritems():
            for element in ['Aiso', 'A', 'B', 'C', 'D']:
                name = construct_name([element, target_L0, key[0], key[1],
                                       sweeps, val[0]])
                file.write('cp ' + orig_folder + name +
                           ' ' + final_folder + name + '\n')


def load_RGMPS(folder, target_L0, hx, Jx, best_runs, tensors=None):
    if tensors is not None:
        DMRG_sweeps = 1
        params = {"hx": hx, "hz": 0.0, "Jx": Jx, "Jz": 0.0, "sweeps":
                  DMRG_sweeps, "D": 8, "L": 8, "PBC": False}
        renorm = RGMPS_shift.RGMPS(params, modelname="MFieldAndCoupling")
        renorm.MPS.MPS = tensors  # palm-to-face
    else:
        # TODO: grab DMRG params chris used from odyssey scripts
        DMRG_sweeps = 1
        params = {"hx": hx, "hz": 0.0, "Jx": Jx, "Jz": 0.0, "sweeps":
                  DMRG_sweeps, "D": 8, "L": 8, "PBC": False}
        renorm = RGMPS_shift.RGMPS(params, modelname="MFieldAndCoupling")

    run = best_runs[(hx, Jx)][0]

    f = open(folder + construct_name(['A', target_L0, hx, Jx, sweeps, run]),
             "rb")
    renorm.UTurnOnWithHA = pickle.load(f)
    f.close()
    f = open(folder + construct_name(['Aiso', target_L0, hx, Jx, sweeps, run]),
             "rb")
    renorm.IsoTurnOnWithH = pickle.load(f)
    f.close()
    f = open(folder + construct_name(['B', hx, target_L0, Jx, sweeps, run]),
             "rb")
    renorm.UTurnOffB = pickle.load(f)
    f.close()
    f = open(folder + construct_name(['C', hx, target_L0, Jx, sweeps, run]),
             "rb")
    renorm.UTurnOnWithHC = pickle.load(f)
    f.close()
    f = open(folder + construct_name(['D', target_L0, hx, Jx, sweeps, run]),
             "rb")
    renorm.UTurnOffD = pickle.load(f)
    f.close()

    return renorm


def expvals(folder, target_L0, best_runs, valid_inds=None, tensors_list=None):
    hxs = np.sort([x[0] for x in best_runs.keys()])
    Jx = 0.0  # HACK
    if valid_inds is None:
        valid_inds = range(len(hxs))
    Es, Xs, Zs, XXs, ZZs = [], [], [], [], []
    for tind, i in enumerate(valid_inds):
        if tensors_list:
            tensors = tensors_list[tind]
        else:
            tensors = None
        hx = hxs[i]
        E = best_runs[(hx, Jx)][0]
        renorm = load_RGMPS(folder, target_L0, hx, Jx, best_runs,
                            tensors=tensors)
        assert np.islclose(E, renorm.calcenergy(1))
        Es.append(E)
        Xs.append(renorm.calcExpO(1, "X"))
        Zs.append(renorm.calcExpO(1, "Z"))
        XXs.append(renorm.calcExpO(1, "XX"))
        ZZs.append(renorm.calcExpO(1, "ZZ"))
    return Es, Xs, Zs, XXs, ZZs


# =================== ASSEMBLY ===================

mSz_L0_8_tensors_list = pickle.load(open('L8_tensor_data_list.p'))
L0_16_tensors_list = pickle.load(open('L16_tensor_data_list.p'))

noncat_Es_folder = 'Data/NonCat/'
remote_noncat_folder = 'SavedRGMPSNonCat/'
loc_noncat_folder = 'SavedRGMPSNonCat-best-runs/'

std_Es_folder = 'Data/Std/'
remote_std_folder = 'SavedRGMPSShift/'
loc_std_folder = 'SavedRGMPSShift-best-runs/'

mSz_L0_8_inds = [0, 1] + list(range(3, 41))
mSz_L0_8_best_runs = find_best_runs(noncat_Es_folder, 8)
# skip copying files over etc... did that for this split yesterday
mSz_L0_8_Es, _, mSz_L0_8_Zs, _, _ = expvals(loc_noncat_folder, 8,
                                            mSz_L0_8_best_runs,
                                            valid_inds=mSz_L0_8_inds,
                                            tensors_list=mSz_L0_8_tensors_list)

# =================== PLOTTING ===================
mSz_L0_8_in_Zs = np.loadtxt('mSz_L8_avg_Sz.txt')

# figutils.make_figure(hxs[np.array(valid_inds)], [np.array(Zs).real, in_Zs],
#                      'hx', '<Z>', ['RGMPS', 'MPS in'], marker='.',
#                      title='L0=8 Lf=16', filename='RGMPS_vs_MPS_in_Sz.png')
#
# figutils.make_figure(hxs[np.array(valid_inds)],
#                      [np.abs(np.array(Zs).real), in_Zs],
#                      'hx', '|<Z>|', ['RGMPS', 'MPS in'], marker='.',
#                      title='L0=8 Lf=16',
#                      filename='RGMPS_vs_MPS_in_abs_Sz.png')
