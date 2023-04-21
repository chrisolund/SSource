import numpy as np
import tensor as ten
import copy


def spin_bond_name(n):
    return "spin"+str(n)


def mpo_bond_left_name(n, PBC=False, L=0):
    if PBC and n == 0:
        return "mpo_"+str(L-1)+"_"+str(0)
    else:
        return "mpo_"+str(n-1)+"_"+str(n)


def mpo_bond_right_name(n, PBC=False, L=0):
    if PBC and n == L-1:
        return "mpo_"+str(L-1)+"_"+str(0)
    else:
        return "mpo_"+str(n)+"_"+str(n+1)


def mps_bond_left_name(n, PBC=False, L=0):
    if PBC and n == 0:
        return "mps_"+str(L-1)+"_"+str(0)
    else:
        return "mps_"+str(n-1)+"_"+str(n)


def mps_bond_right_name(n, PBC=False, L=0):
    if PBC and n == L-1:
        return "mps_"+str(L-1)+"_"+str(0)
    else:
        return "mps_"+str(n)+"_"+str(n+1)


class mps:
    def __init__(self, D, d, L, PBC=False, ferro=False):
        assert L % 2 == 0
        self.D = D
        self.d = d
        self.L = L
        self.MPS = []
        if not PBC:
            for n in range(self.L):
                powleft = n if n < self.L/2 else (self.L-1)-(n-1)
                powright = n+1 if n < self.L/2 else (self.L-1)-n
                dimleft = min(2**powleft, self.D)
                dimright = min(2**powright, self.D)
                bonds = [ten.Bond(spin_bond_name(n), 2, False), ten.Bond(mps_bond_left_name(
                    n), dimleft, False), ten.Bond(mps_bond_right_name(n), dimright, False)]
                self.MPS.append(ten.Tensor(np.random.uniform(
                    size=[self.d, dimleft, dimright]), bonds))
        else:
            dimleft = self.D
            dimright = self.D
            # dat=np.random.uniform(size=[self.d,dimleft,dimright])
            # dat2=np.random.uniform(size=[self.d,dimleft,dimright])
            for n in range(self.L):
                if(ferro):
                    dat = np.array([np.ones((self.D, self.D)), np.zeros((self.D, self.D))])
                else:
                    dat = np.random.uniform(size=[self.d, dimleft, dimright])
                bonds = [ten.Bond(spin_bond_name(n), 2, False), ten.Bond(mps_bond_left_name(
                    n, True, self.L), dimleft, False), ten.Bond(mps_bond_right_name(n, True, self.L), dimright, False)]
                # if n%2==0:
                self.MPS.append(ten.Tensor(dat, bonds))
                # else:
                # self.MPS.append(ten.Tensor(dat2,bonds))
        self.isLeftCanonical = False
        self.isRightCanonical = False
        self.isMixedCanonical = False

    def makeLeftCanonicalSite(self, n):
        shapeMPS = self.MPS[n].data.shape
        # qr Decomposition
        q, r = np.linalg.qr(np.reshape(self.MPS[n].data, (self.d*shapeMPS[1], shapeMPS[2])))
        # Reshape to M
        mat_qr = np.reshape(q, (self.d, shapeMPS[1], shapeMPS[2]))
        if (n < self.L-1):
            for d in range(self.d):
                self.MPS[n+1].data[d] = np.dot(r, self.MPS[n+1].data[d])
        self.MPS[n].data = mat_qr

    def makeRightCanonicalSite(self, n):
        shapeMPS = self.MPS[n].data.shape
        # QR Decomposition
        r_tr = np.transpose(self.MPS[n].data, (0, 2, 1))
        q, r = np.linalg.qr(np.reshape(r_tr, (self.d*shapeMPS[2], shapeMPS[1])))
        q_reshape = np.reshape(q, (self.d, shapeMPS[2], shapeMPS[1]))
        q_tr = np.transpose(q_reshape, (0, 2, 1))
        if (n > 0):
            for d in range(self.d):
                self.MPS[n-1].data[d] = np.dot(self.MPS[n-1].data[d], r)
        # Reshape to M
        self.MPS[n].data = q_tr

    def makeLeftCanonical(self):
        for n in range(self.L):
            self.makeLeftCanonicalSite(n)

    def makeRightCanonical(self):
        for n in reversed(range(self.L)):
            self.makeRightCanonicalSite(n)

    def makeMixedCanonical(self, n):
        for n in range(n):
            self.makeLeftCanonicalSite(n)
        for n in reversed(range(n, self.L)):
            self.makeRightCanonicalSite(n)

    def checkLeftCanonical(self):
        for n in range(self.L):
            shapeMPS = self.MPS[n].shape
            # print product
            print np.allclose(np.sum([np.dot(self.MPS[n].data[d].conj().T, self.MPS[n].data[d])
                                      for d in range(self.d)], axis=0), np.eye(shapeMPS[2]))

    def checkRightCanonical(self):
        for n in range(self.L):
            shapeMPS = self.MPS[n].shape
            # print product
            print np.allclose(np.sum([np.dot(self.MPS[n].data[d].conj(), self.MPS[n].data[d].T)
                                      for d in range(self.d)], axis=0), np.eye(shapeMPS[1]))

    def getMPSn(self, n):
        return self.MPS[n]

    def getMPSnDagger(self, n):
        tensor_dag = ten.Tensor.TensorCopy(self.MPS[n])
        tensor_dag.data.conjugate()
        for b in tensor_dag.bonds:
            b.prime = True
        return tensor_dag


def MPS1Point(i, O1, L, MPS):
    Q = copy.deepcopy(MPS.getMPSn(0))
    if(i == 0):
        bonds = [ten.Bond("spin"+str(0), 2), ten.Bond("spin"+str(0), 2, True)]
        A = ten.Tensor(O1, bonds)
        Q = ten.contract(Q, A)
    else:
        Q.setBondPrime("spin"+str(0), True)
    Q = ten.contract(Q, MPS.getMPSnDagger(0))

    for k in range(L-1):
        Q = ten.contract(Q, copy.deepcopy(MPS.getMPSn(k+1)))
        if(k+1 == i):
            bonds = [ten.Bond("spin"+str(k+1), 2), ten.Bond("spin"+str(k+1), 2, True)]
            A = ten.Tensor(O1, bonds)
            Q = ten.contract(Q, A)
        else:
            Q.setBondPrime("spin"+str(k+1), True)
        Q = ten.contract(Q, MPS.getMPSnDagger(k+1))
    # Q.printBonds()
    return float(Q.data.real)


def MPS2Point(i, j, O1, O2, L, MPS):
    Q = copy.deepcopy(MPS.getMPSn(0))
    if(i == 0):
        bonds = [ten.Bond("spin"+str(0), 2), ten.Bond("spin"+str(0), 2, True)]
        A = ten.Tensor(O1, bonds)
        Q = ten.contract(Q, A)
    elif(j == 0):
        bonds = [ten.Bond("spin"+str(0), 2), ten.Bond("spin"+str(0), 2, True)]
        A = ten.Tensor(O2, bonds)
        Q = ten.contract(Q, A)
    else:
        Q.setBondPrime("spin"+str(0), True)
    Q = ten.contract(Q, MPS.getMPSnDagger(0))

    for k in range(L-1):
        Q = ten.contract(Q, copy.deepcopy(MPS.getMPSn(k+1)))
        if(k+1 == i):
            bonds = [ten.Bond("spin"+str(k+1), 2), ten.Bond("spin"+str(k+1), 2, True)]
            A = ten.Tensor(O1, bonds)
            Q = ten.contract(Q, A)
        elif(k+1 == j):
            bonds = [ten.Bond("spin"+str(k+1), 2), ten.Bond("spin"+str(k+1), 2, True)]
            A = ten.Tensor(O2, bonds)
            Q = ten.contract(Q, A)
        else:
            Q.setBondPrime("spin"+str(k+1), True)
        Q = ten.contract(Q, MPS.getMPSnDagger(k+1))
    # Q.printBonds()
    return float(Q.data.real)
