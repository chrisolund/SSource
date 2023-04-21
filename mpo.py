import numpy as np

import tensor as ten
import spinmat
from mps import spin_bond_name,mpo_bond_left_name,mpo_bond_right_name


## Heisenberg

def MpoHeisenberg(h,J,Jz):    
    data=np.array([[spinmat.id_mat, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat]
            ,[spinmat.sigma_plus, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat]
            ,[spinmat.sigma_minus, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat]
            ,[spinmat.sigma_z, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat]
            ,[-h* spinmat.sigma_z,(J/2.)* spinmat.sigma_minus,(J/2.)* spinmat.sigma_plus,Jz* spinmat.sigma_z, spinmat.id_mat]
            ])
    return {"mpo":data,"MPOD":5}
def MpoTFieldIsing(h,J):    
    data=np.array([[spinmat.id_mat, spinmat.zero_mat, spinmat.zero_mat]
            ,[spinmat.sigma_z, spinmat.zero_mat, spinmat.zero_mat]
            ,[-h* spinmat.sigma_x,-J* spinmat.sigma_z, spinmat.id_mat]
            ])
    #data0: different mpo at site 0 due to PBCs
    data0=np.array([[-h*spinmat.sigma_x,-J*spinmat.sigma_z,spinmat.id_mat],
                    [spinmat.zero_mat,spinmat.zero_mat,spinmat.sigma_z],
                    [spinmat.zero_mat,spinmat.zero_mat,spinmat.zero_mat]])
    return {"mpo":data,"mpo0":data0,"MPOD":3,"h":h,"J":J}
def MpoMFieldIsing(hx,hz,J):
    data=np.array([[spinmat.id_mat, spinmat.zero_mat, spinmat.zero_mat]
            ,[spinmat.sigma_z, spinmat.zero_mat, spinmat.zero_mat]
            ,[-hx*spinmat.sigma_x-hz*spinmat.sigma_z,-J* spinmat.sigma_z, spinmat.id_mat]
            ])
    #data0: different mpo at site 0 due to PBCs
    data0=np.array([[-hx*spinmat.sigma_x-hz*spinmat.sigma_z,-J*spinmat.sigma_z,spinmat.id_mat],
                    [spinmat.zero_mat,spinmat.zero_mat,spinmat.sigma_z],
                    [spinmat.zero_mat,spinmat.zero_mat,spinmat.zero_mat]])
    return {"mpo":data,"mpo0":data0,"MPOD":3,"hx":hx,"hz":hz,"J":J}

def MpoMFieldAndCoupling(hx,hz,Jx,Jz):
    data=np.array([[spinmat.id_mat, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat]
            ,[spinmat.sigma_z, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat]
            ,[spinmat.sigma_x, spinmat.zero_mat, spinmat.zero_mat, spinmat.zero_mat]
            ,[-hx*spinmat.sigma_x-hz*spinmat.sigma_z,-Jz*spinmat.sigma_z, -Jx*spinmat.sigma_x, spinmat.id_mat]
            ])
    #data0: different mpo at site 0 due to PBCs
    data0=np.array([[-hx*spinmat.sigma_x-hz*spinmat.sigma_z,-Jz*spinmat.sigma_z,-Jx*spinmat.sigma_x,spinmat.id_mat],
                    [spinmat.zero_mat,spinmat.zero_mat,spinmat.zero_mat,spinmat.sigma_z],
                    [spinmat.zero_mat,spinmat.zero_mat,spinmat.zero_mat,spinmat.sigma_x],
                    [spinmat.zero_mat,spinmat.zero_mat,spinmat.zero_mat,spinmat.zero_mat]])
    return {"mpo":data,"mpo0":data0,"MPOD":4,"hx":hx,"hz":hz,"Jx":Jx,"Jz":Jz}

def MpoId():
    data=np.array([[spinmat.id_mat]])
    return {"mpo":data,"MPOD":1}
def MpoSx(h):
    data=np.array([[spinmat.id_mat, spinmat.zero_mat],[-h*spinmat.sigma_x,spinmat.id_mat]])
    return {"mpo":data,"MPOD":2}

class mpo:    
    def __init__(self,L,model,PBC=False):
        self.MPO=[]
        self.MPOD=model["MPOD"]
        for n in range(L):
            bonds=[ten.Bond(mpo_bond_left_name(n,PBC,L),self.MPOD),ten.Bond(mpo_bond_right_name(n,PBC,L),self.MPOD),ten.Bond(spin_bond_name(n),2),ten.Bond(spin_bond_name(n),2,True)]
            if n>0 or not PBC or self.MPOD==1:
                self.MPO.append(ten.Tensor(model["mpo"],bonds))
            else:
                self.MPO.append(ten.Tensor(model["mpo0"],bonds))
                #self.MPO.append(ten.Tensor(np.tensordot(PBCFix,model["mpo"],axes=([1],[0])),bonds))
    

    @classmethod
    def MpoTFieldIsing(cls,L,h,J,PBC=False):
        data=MpoTFieldIsing(h,J)
        return cls(L,data,3,PBC)
    @classmethod
    def MpoHeisenberg(cls,L,h,J,Jz,PBC=False):
        data=MpoHeisenberg(h,J,Jz)
        return cls(L,data,5,PBC)
    @classmethod
    def MpoId(cls,L,PBC=False):
        data=MpoId()
        return cls(L,data,1,PBC)
    @classmethod
    def MpoSx(cls,L,h,PBC=False):
        data=MpoSx(h)
        return cls(L,data,2,PBC)
    def getMPOn(self,n):
        return self.MPO[n]
            