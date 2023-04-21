import numpy as np
import time
import RGMPS_shift
import cPickle as pickle

hxs = np.logspace(np.log(0.5)/np.log(10), np.log(2)/np.log(10), 10)
Jxs = np.logspace(np.log(0.1)/np.log(10), np.log(0.2)/np.log(10), 2)
ind_hx = 5
ind_Jx = 0
Jz = 1.0
hz = 0.0
hx = hxs[ind_hx]
Jx = Jxs[ind_Jx]
L0 = 8
Lf = 16

# construct hamiltonian operators
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
s0 = np.array([[1, 0], [0, 1]])
XIII = np.kron(np.kron(np.kron(sx, s0), s0), s0).reshape(2, 2, 2, 2,
                                                         2, 2, 2, 2)
ZIII = np.kron(np.kron(np.kron(sz, s0), s0), s0).reshape(2, 2, 2, 2,
                                                         2, 2, 2, 2)
XXII = np.kron(np.kron(np.kron(sx, sx), s0), s0).reshape(2, 2, 2, 2,
                                                         2, 2, 2, 2)
ZZII = np.kron(np.kron(np.kron(sz, sz), s0), s0).reshape(2, 2, 2, 2,
                                                         2, 2, 2, 2)
XIXI = np.kron(np.kron(np.kron(sx, s0), sx), s0).reshape(2, 2, 2, 2,
                                                         2, 2, 2, 2)
ZIZI = np.kron(np.kron(np.kron(sz, s0), sz), s0).reshape(2, 2, 2, 2,
                                                         2, 2, 2, 2)
XIIX = np.kron(np.kron(np.kron(sx, s0), s0), sx).reshape(2, 2, 2, 2,
                                                         2, 2, 2, 2)
ZIIZ = np.kron(np.kron(np.kron(sz, s0), s0), sz).reshape(2, 2, 2, 2,
                                                         2, 2, 2, 2)
XXops = [XXII, XIXI, XIIX]
ZZops = [ZZII, ZIZI, ZIIZ]


# hyper parameters - shoudn't need dmrg parameters
sweeps = 2  # same 5-D behavior for 20
DRMG_sweeps = 1
DRMG_bond_dim = 8

# initialze ssource network
params = {"hx": hx, "hz": hz, "Jx": Jx, "Jz": Jz, "sweeps": DRMG_sweeps,
          "D": DRMG_bond_dim, "L": L0, "PBC": False}
renorm = RGMPS_shift.RGMPS(params, modelname="MFieldAndCoupling")

# load MPS input
f = open("L8_tensor_data_list.p", "rb")
blah = pickle.load(f)
for i in range(L0):
    renorm.MPS.MPS[i] = blah[ind_hx][i]

# optimize s-source generation of Lf state
for i in range(1, 1+int(np.log2(Lf/L0))):
    print("Layer "+str(i)+":")
    tic = time.time()
    renorm.setUstotallyrandom(i)

    renorm.test = True
    Es = renorm.optimizeUGrad(i, sweeps, noise=0.0, decay=0.0, suppress=True)
    toc = time.time()
    print("Time to optimize to "+str((2**i)*L0)+":")
    print(toc-tic)
    print('')
    np.savetxt("Data/NonCat/E_L0="+str(L0)+"_layer="+str(i) +
               "_hx="+str(hx) + "_Jx="+str(Jx)+"_sweeps="+str(sweeps) +
               "_run=loc.csv", np.squeeze(Es))  # example will be run locally

with open("Data/NonCat/RGMPS_L0="+str(L0)+"_Lf="+str(Lf)+"_hx="+str(hx) +
          "_Jx="+str(Jx)+"_sweeps="+str(sweeps) + "_run=loc.p", "wb") as f:
    pickle.dump(renorm, f)
