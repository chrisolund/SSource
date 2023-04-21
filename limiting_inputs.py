import numpy as np
import mps
import pickle

bond_dim = 8
site_dim = 2
nsites = 32  # 8
rand_mps = mps.mps(bond_dim, site_dim, nsites)
rand_mps.makeLeftCanonical()
rand_tensors = rand_mps.MPS
for t in rand_tensors:
    print(t.data)
with open('rand_input.p', 'wb') as f:
    pickle.dump(rand_tensors, f)

bond_dim = 1
prod_mps = mps.mps(bond_dim, site_dim, nsites)
for t in prod_mps.MPS:
    shape = t.data.shape
    new_data = np.array([1, 0]).reshape(shape)
    t.data = new_data
prod_tensors = prod_mps.MPS
for t in prod_tensors:
    print(t.data)
# fname = 'prod_input.p'
# fname = 'prod_input_16.p'
fname = 'prod_input_32.p'
with open(fname, 'wb') as f:
    pickle.dump(prod_tensors, f)

nsites = 64
bond_dim = 1
prod_mps = mps.mps(bond_dim, site_dim, nsites)
for t in prod_mps.MPS:
    shape = t.data.shape
    new_data = np.array([1, 0]).reshape(shape)
    t.data = new_data
prod_tensors = prod_mps.MPS
for t in prod_tensors:
    print(t.data)
# fname = 'prod_input.p'
# fname = 'prod_input_16.p'
fname = 'prod_input_64.p'
with open(fname, 'wb') as f:
    pickle.dump(prod_tensors, f)
