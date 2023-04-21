import numpy as np
import copy
import itertools
from operator import mul, itemgetter

class Bond:
    def __init__(self,name,size,prime=False):
        self.name=name
        self.size=size
        self.prime=prime
    def makePrime(self):
        new_bond=Bond(self.name,self.size)
        new_bond.prime=True
        return new_bond
    def makeUnPrime(self):
        new_bond=Bond(self.name,self.size)
        new_bond.prime=False
        return new_bond

class Tensor:
    def __init__(self,data,bonds):
        self.data=np.array(data)
        self.bonds=bonds
        assert all([a==b for a,b in zip(self.data.shape,[b.size for b in self.bonds])])
    def checkIfBondExist(self,bond):
        for b in self.bonds:
            if b.name==bond.name and b.prime==bond.prime:
                return True
        return False
    def getBond(self,bond):
        assert self.checkIfBondExist(bond)
        for i,b in enumerate(self.bonds):
            if b.name==bond.name and b.prime==bond.prime:
                return b
    def getBondIndex(self,bond):
        assert self.checkIfBondExist(bond)
        for i,b in enumerate(self.bonds):
            if b.name==bond.name and b.prime==bond.prime:
                return i
    def groupTensor(self,groups):
        newBonds,sizes,indices = zip(*[(Bond(g["bond_name"],reduce(mul,[b.size for b in g["group_bonds"]])),[self.getBondIndexByName(b.name) for b in g["group_bonds"]]) for g in groups])
        self.data=np.reshape(np.transpose(self.data,indices),sizes)
        self.bonds=[Bond(n,s) for n,s in zip(newBonds,sizes)]
    def checkIfBondExistByNameAndPrime(self,bond_name,bond_prime=False):
        for b in self.bonds:
            if b.name==bond_name and b.prime==bond_prime:
                return True
        return False
    def checkIfBondExistByName(self,bond_name):
        for b in self.bonds:
            if b.name==bond_name:
                return True
        return False
    def getBondByName(self,bond_name,bond_prime=False):
        assert self.checkIfBondExistByNameAndPrime(bond_name,bond_prime)
        for i,b in enumerate(self.bonds):
            if b.name==bond_name and b.prime==bond_prime:
                return b
    def setBondPrime(self,bond_name,bond_prime):
        assert self.checkIfBondExistByName(bond_name)
        for i,b in enumerate(self.bonds):
            if b.name==bond_name:
                b.prime=bond_prime
    def getBondIndexByName(self,bond_name,bond_prime=False):
        assert self.checkIfBondExistByNameAndPrime(bond_name,bond_prime)
        for i,b in enumerate(self.bonds):
            if b.name==bond_name and b.prime==bond_prime:
                return i
    def groupTensor(self,groups):
        newBonds,indices = zip(*[(Bond(g["bond_name"],reduce(mul,[b.size for b in g["group_bonds"]])),[self.getBondIndex(b) for b in g["group_bonds"]]) for g in groups])
        self.data=np.reshape(np.transpose(self.data,list(itertools.chain(*indices))),[b.size for b in newBonds])
        self.bonds=newBonds
    def unGroupTensor(self,bonds):
        sizes = [b.size for b in bonds]
        self.data=np.reshape(self.data,sizes)
        self.bonds=bonds
    def changeBondName(self,new_name,old_name):
        for i,b in enumerate(self.bonds):
            if b.name==old_name:
                b.name=new_name
    def printBonds(self):
        for i,b in enumerate(self.bonds):
            print "name, ",b.name," prime ",b.prime
    def reorderBonds(self,bonds_order):
        new_index=[]
        for b in bonds_order:
            new_index.append(self.getBondIndexByName(b.name,b.prime))
        self.data=np.transpose(self.data,axes=tuple(new_index))
        self.bonds=bonds_order
    @classmethod
    def randTensor(cls,bonds):
        sizes=[x.size for x in bonds]
        return cls(np.random.rand(tuple(sizes)),bonds)
    @classmethod
    def zeroTensor(cls,bonds):
        sizes=[x.size for x in bonds]
        return cls(np.zeros(tuple(sizes)),bonds)
    @classmethod
    def TensorCopy(cls,tensor):
        return copy.deepcopy(tensor)
def contract(Ta,Tb):
    axes_cont_A=[]
    axes_cont_B=[]
    axes_no_cont_A=[]
    b_bonds=list(Tb.bonds)
    for ia,ba in enumerate(Ta.bonds):
        if Tb.checkIfBondExist(ba):
                ib=Tb.getBondIndex(ba)
                axes_cont_A.append(ia)
                axes_cont_B.append(ib)
                continue
        axes_no_cont_A.append(ba)
    new_bonds=axes_no_cont_A+[b for n,b in enumerate(Tb.bonds) if n not in axes_cont_B]
    return Tensor(np.tensordot(Ta.data,Tb.data,axes=(axes_cont_A,axes_cont_B)),new_bonds)
