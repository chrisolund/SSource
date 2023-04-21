'''
Created on May 4, 2014

@author: snirgaz
'''

import numpy as np
from scipy import linalg as LA

import mpo
from mps import spin_bond_name,mpo_bond_left_name,mpo_bond_right_name,mps_bond_left_name,mps_bond_right_name
import mps as mps
import spinmat
import tensor as ten


class minimizeMPO(object):
    '''
    classdocs
    '''    
    
    def __init__(self, params,PBC=False,modelname="TFieldIsing"):
        '''
        Constructor
        '''
        self.params=params
        #if np.abs(self.params["h"])<=np.abs(self.params["J"]):
        #self.mps= mps.mps(self.params["D"],self.params["d"],self.params["L"],PBC=PBC,ferro=True)
        #else:
        self.mps= mps.mps(self.params["D"],self.params["d"],self.params["L"],PBC=PBC)
        #model=mpo.MpoHeisenberg(self.params["h"],self.params["J"],self.params["Jz"])
        if modelname=="TFieldIsing":
            model= mpo.MpoTFieldIsing(self.params["h"],self.params["J"])
        elif modelname=="MFieldIsing":
            model= mpo.MpoMFieldIsing(self.params["hx"],self.params["hz"],self.params["J"])
        elif modelname=="MFieldAndCoupling":
            model= mpo.MpoMFieldAndCoupling(self.params["hx"],self.params["hz"],self.params["Jx"],self.params["Jz"])
        else:
            print("Invalid model name")
        #model= mpo.MpoSx(self.params["h"])
        self.mpo= mpo.mpo(self.params["L"],model,PBC)
        self.mpoid=mpo.mpo(self.params["L"],mpo.MpoId(),PBC)
        self.HLeft=[]
        self.HRight=[]
        self.NLeft=[]
        self.NRight=[]
        for n,(cMPS,cMPO) in enumerate(zip(self.mps.MPS,self.mpo.MPO)):
            left_bond_mps=cMPS.getBondByName(mps_bond_left_name(n,PBC,self.params["L"]))
            left_bond_mpo=cMPO.getBondByName(mpo_bond_left_name(n,PBC,self.params["L"]))
            bonds_left=[left_bond_mps,left_bond_mpo,left_bond_mps.makePrime()]
            self.HLeft.append(ten.Tensor.zeroTensor(bonds_left))
            self.NLeft.append(ten.Tensor.zeroTensor(bonds_left))
            right_bond_mps=cMPS.getBondByName(mps_bond_right_name(n,PBC,self.params["L"]))
            right_bond_mpo=cMPO.getBondByName(mpo_bond_right_name(n,PBC,self.params["L"]))
            bonds_right=[right_bond_mps,right_bond_mpo,right_bond_mps.makePrime()]
            self.HRight.append(ten.Tensor.zeroTensor(bonds_right))
            self.NRight.append(ten.Tensor.zeroTensor(bonds_right))
        if PBC:
            self.normalize_MPS_PBC()
        
            A=ten.contract(self.mps.getMPSnDagger(0),self.mpo.getMPOn(0));
            A=ten.contract(A,self.mps.getMPSn(0))
            self.HLeft[1]=A
            A=ten.contract(self.mps.getMPSnDagger(self.params["L"]-1),self.mpo.getMPOn(self.params["L"]-1));
            A=ten.contract(A,self.mps.getMPSn(self.params["L"]-1))
            self.HRight[self.params["L"]-2]=A
            
            A=ten.contract(self.mps.getMPSnDagger(0),self.mpoid.getMPOn(0));
            A=ten.contract(A,self.mps.getMPSn(0))
            self.NLeft[1]=A
            A=ten.contract(self.mps.getMPSnDagger(self.params["L"]-1),self.mpoid.getMPOn(self.params["L"]-1));
            A=ten.contract(A,self.mps.getMPSn(self.params["L"]-1))
            self.NRight[self.params["L"]-2]=A
        else:
            self.HLeft[0].data[0,self.params["DMPO"]-1,0]=1.
            self.HRight[self.params["L"]-1].data[0,0,0]=1.
        self.E=0
    def updateHLeft(self,n,PBC=False):
        if PBC and n==0:
            A=ten.contract(self.mps.getMPSnDagger(n),self.mpo.getMPOn(n))
            A=ten.contract(A, self.mps.getMPSn(n))
        else:
            A=ten.contract(self.HLeft[n], self.mps.getMPSnDagger(n))
            A=ten.contract(A, self.mpo.getMPOn(n))
            A=ten.contract(A, self.mps.getMPSn(n))
        #assert A.data.shape == self.HLeft[n+1].data.shape
        self.HLeft[n+1]=A
    def updateHRight(self,n,PBC=False):
        if PBC and n==self.params["L"]-1:
            A=ten.contract(self.mps.getMPSnDagger(n), self.mpo.getMPOn(n))
            A=ten.contract(A, self.mps.getMPSn(n))
        else:
            A=ten.contract(self.HRight[n], self.mps.getMPSnDagger(n))
            A=ten.contract(A, self.mpo.getMPOn(n))
            A=ten.contract(A, self.mps.getMPSn(n))
        #assert A.data.shape == self.HRight[n-1].data.shape
        self.HRight[n-1]=A
    def updateNLeft(self,n,PBC=False):
        if PBC and n==0:
            A=ten.contract(self.mps.getMPSnDagger(n), self.mpoid.getMPOn(n))
            A=ten.contract(A, self.mps.getMPSn(n))
        else:
            A=ten.contract(self.NLeft[n], self.mps.getMPSnDagger(n))
            A=ten.contract(A, self.mpoid.getMPOn(n))
            A=ten.contract(A, self.mps.getMPSn(n))
        #assert A.data.shape == self.NLeft[n+1].data.shape
        self.NLeft[n+1]=A
    def updateNRight(self,n,PBC=False):
        if PBC and n==self.params["L"]-1:
            A=ten.contract(self.mps.getMPSnDagger(n), self.mpoid.getMPOn(n))
            A=ten.contract(A, self.mps.getMPSn(n))
        else:
            A=ten.contract(self.NRight[n], self.mps.getMPSnDagger(n))
            A=ten.contract(A, self.mpoid.getMPOn(n))
            A=ten.contract(A, self.mps.getMPSn(n))
        #assert A.data.shape == self.NRight[n-1].data.shape
        self.NRight[n-1]=A
    def initializeMinimization(self,PBC=False):
        # Make left canonical and build Hamiltonian
        for n in range(self.params["L"]):
            if not PBC:
                self.mps.makeLeftCanonicalSite(n)
            if (n<self.params["L"]-1 and ((not PBC) or n>0)):
                self.updateHLeft(n)
                if PBC:
                    self.updateHRight(self.params["L"]-1-n)
                    self.updateNLeft(n)
                    self.updateNRight(self.params["L"]-1-n)

    def normalize_MPS_PBC(self):
        N=ten.contract(self.mps.getMPSnDagger(0),self.mpoid.getMPOn(0));
        N=ten.contract(N,self.mps.getMPSn(0))
        for n in range(1,self.params["L"]):
            N=ten.contract(N,self.mps.getMPSnDagger(n))
            N=ten.contract(N,self.mpoid.getMPOn(n))
            N=ten.contract(N,self.mps.getMPSn(n))
        for n in range(0,self.params["L"]):
            self.mps.MPS[n].data/=((N.data)**(0.5/self.params["L"]))

    def minimizeH_MPS_Site(self,n,PBC=False):
        if not PBC:
            mpo=self.mpo.getMPOn(n)
            H=ten.contract(self.HLeft[n], mpo)
            H=ten.contract(H,self.HRight[n])
            HBonds=[H.getBondByName(spin_bond_name(n)),H.getBondByName(mps_bond_left_name(n)),H.getBondByName(mps_bond_right_name(n))]
            HPrimeBonds=[H.getBondByName(spin_bond_name(n),True),H.getBondByName(mps_bond_left_name(n),True),H.getBondByName(mps_bond_right_name(n),True)]
            H.groupTensor([{"bond_name":"H","group_bonds":HBonds}
                         ,{"bond_name":"H_prime","group_bonds":HPrimeBonds}])
            w,v=np.linalg.eigh(H.data)
            self.E=w[0]
            M=np.reshape(v[:,0],self.mps.getMPSn(n).data.shape)
            M=M+np.random.normal(0,self.params["noise"], M.shape)
            self.mps.MPS[n]=ten.Tensor(M,HBonds)
        else:
            mpo=self.mpo.getMPOn(n)
            mpoid=self.mpoid.getMPOn(n)
            if n>0:
                N=ten.contract(self.NLeft[n], mpoid)
                H=ten.contract(self.HLeft[n], mpo)
            else:
                N=mpoid
                H=mpo
            if n<self.params["L"]-1:
                N=ten.contract(N,self.NRight[n])
                H=ten.contract(H,self.HRight[n])
            HBonds=[H.getBondByName(spin_bond_name(n)),H.getBondByName(mps_bond_left_name(n,True,self.params["L"])),H.getBondByName(mps_bond_right_name(n,True,self.params["L"]))]
            HPrimeBonds=[H.getBondByName(spin_bond_name(n),True),H.getBondByName(mps_bond_left_name(n,True,self.params["L"]),True),H.getBondByName(mps_bond_right_name(n,True,self.params["L"]),True)]
            H.groupTensor([{"bond_name":"H","group_bonds":HBonds}
                           ,{"bond_name":"H_prime","group_bonds":HPrimeBonds}])
            N.groupTensor([{"bond_name":"N","group_bonds":HBonds}
                           ,{"bond_name":"N_prime","group_bonds":HPrimeBonds}])
            eps=1e-12
            H.data=H.data+(eps**.5)*np.eye(len(H.data))
            N.data=N.data+eps*np.eye(len(N.data))
            w,v=LA.eigh(.5*(H.data+H.data.conjugate().T),.5*(N.data+N.data.conjugate().T))
    
            self.E=w[0]
            M=np.reshape(v[:,0],self.mps.getMPSn(n).data.shape)
            M=M+np.random.normal(0,self.params["noise"], M.shape)
            self.mps.MPS[n]=ten.Tensor(M,HBonds)
            if n>0:
                N=ten.contract(self.NLeft[n], mpoid)
            else:
                N=mpoid
            if n<self.params["L"]-1:
                N=ten.contract(N,self.NRight[n])
            norm=ten.contract(ten.contract(N,self.mps.getMPSn(n)),self.mps.getMPSnDagger(n))
            self.mps.MPS[n].data/=((norm.data)**0.5)


    def minimizeH_MPS_PBC_TI_BLAHTEMP(self):
        mpo=self.mpo.getMPOn(0)
        mpoid=self.mpoid.getMPOn(0)
        H=ten.contract(self.mps.getMPSnDagger(1),self.mpo.getMPOn(1));
        H=ten.contract(H,self.mps.getMPSn(1))
        N=ten.contract(self.mps.getMPSnDagger(1),self.mpoid.getMPOn(1));
        N=ten.contract(N,self.mps.getMPSn(1))
        
        for n in range(2,self.params["L"]):
            H=ten.contract(H,self.mps.getMPSnDagger(n))
            H=ten.contract(H,self.mpo.getMPOn(n))
            H=ten.contract(H,self.mps.getMPSn(n))
            N=ten.contract(N,self.mps.getMPSnDagger(n))
            N=ten.contract(N,self.mpoid.getMPOn(n))
            N=ten.contract(N,self.mps.getMPSn(n))

        H=ten.contract(H,mpo)
        N=ten.contract(N,mpoid)
        
        #norm=(ten.contract(ten.contract(N,self.mps.getMPSnDagger(0)),self.mps.getMPSn(0)))**.5
        
        
        HBonds=[H.getBondByName(spin_bond_name(0)),H.getBondByName(mps_bond_left_name(0,True,self.params["L"])),H.getBondByName(mps_bond_right_name(0,True,self.params["L"]))]
        HPrimeBonds=[H.getBondByName(spin_bond_name(0),True),H.getBondByName(mps_bond_left_name(0,True,self.params["L"]),True),H.getBondByName(mps_bond_right_name(0,True,self.params["L"]),True)]
        H.groupTensor([{"bond_name":"H","group_bonds":HBonds}
                       ,{"bond_name":"H_prime","group_bonds":HPrimeBonds}])
        N.groupTensor([{"bond_name":"N","group_bonds":HBonds}
                       ,{"bond_name":"N_prime","group_bonds":HPrimeBonds}])
                       #print(LA.norm(H.data-H.data.conjugate().T))
                       #print(LA.norm(N.data-N.data.conjugate().T))
        eps=1e-10
        #H.data=.5*(H.data+H.data.conjugate().T)+(eps**.5)*np.eye(len(H.data))
        #N.data=.5*(N.data+N.data.conjugate().T)+eps*np.eye(len(N.data))
        H.data=H.data+(eps**.5)*np.eye(len(H.data))
        N.data=N.data+eps*np.eye(len(N.data))
        #M=N.data[0:(self.params["D"]*self.params["D"]),0:(self.params["D"]*self.params["D"])]
        #U, s, V = LA.svd(M)
        #print(U)
        #print(s)
        #print(V)
        #N1=(s[0]**-.5)*(U[:,0].reshape(self.params["D"],self.params["D"]))
        #N2=(s[0]**-.5)*(V[0].reshape(self.params["D"],self.params["D"]))
        #X=LA.sqrtm(LA.pinv(N1))
        #Y=LA.sqrtm(LA.pinv(N2))
        #IXY=np.kron(np.eye(2),np.kron(X,Y))
        #H.data=np.dot(IXY,np.dot(H.data,(IXY.conjugate().T)))
        #N.data=np.dot(IXY,np.dot(N.data,(IXY.conjugate().T)))
        #w,v=LA.eigh(.5*(H.data+H.data.conjugate().T)+(eps**.5)*np.eye(len(H.data)),.5*(N.data+N.data.conjugate().T)+eps*np.eye(len(N.data)))
        #print(N.data-N.data.conjugate().T)
        #print(H.data-H.data.conjugate().T)
        w,v=LA.eigh(.5*(H.data+H.data.conjugate().T),.5*(N.data+N.data.conjugate().T))
        
        self.E=w[0]
        #print(w[0])
        #print(LA.norm(w))
        
        #M=np.reshape(np.dot((IXY.conjugate().T),v[:,0]),self.mps.getMPSn(0).data.shape)
        M=np.reshape(v[:,0],self.mps.getMPSn(0).data.shape)
        M=M+np.random.normal(0,self.params["noise"], M.shape)
        for n, cMPS in enumerate(self.mps.MPS):
            bonds=[cMPS.getBondByName(spin_bond_name(n)),cMPS.getBondByName(mps_bond_left_name(n,True,self.params["L"])),cMPS.getBondByName(mps_bond_right_name(n,True,self.params["L"]))]
            self.mps.MPS[n]=ten.Tensor(M,bonds)
        self.normalize_MPS_PBC()


    def minimizeH_MPS_PBC_TI(self):
        #minimize even sites
        mpo=self.mpo.getMPOn(0)
        mpoid=self.mpoid.getMPOn(0)
        H=ten.contract(self.mps.getMPSnDagger(1),self.mpo.getMPOn(1));
        H=ten.contract(H,self.mps.getMPSn(1))
        N=ten.contract(self.mps.getMPSnDagger(1), self.mpoid.getMPOn(1));
        N=ten.contract(N,self.mps.getMPSn(1))
        for n in range(2,self.params["L"]):
            H=ten.contract(H,self.mps.getMPSnDagger(n))
            H=ten.contract(H,self.mpo.getMPOn(n))
            H=ten.contract(H,self.mps.getMPSn(n))
            N=ten.contract(N,self.mps.getMPSnDagger(n))
            N=ten.contract(N,self.mpoid.getMPOn(n))
            N=ten.contract(N,self.mps.getMPSn(n))
        H=ten.contract(H,mpo)
        N=ten.contract(N,mpoid)
        HBonds=[H.getBondByName(spin_bond_name(0)),H.getBondByName(mps_bond_left_name(0,True,self.params["L"])),H.getBondByName(mps_bond_right_name(0,True,self.params["L"]))]
        HPrimeBonds=[H.getBondByName(spin_bond_name(0),True),H.getBondByName(mps_bond_left_name(0,True,self.params["L"]),True),H.getBondByName(mps_bond_right_name(0,True,self.params["L"]),True)]
        H.groupTensor([{"bond_name":"H","group_bonds":HBonds}
                       ,{"bond_name":"H_prime","group_bonds":HPrimeBonds}])
        N.groupTensor([{"bond_name":"N","group_bonds":HBonds}
                       ,{"bond_name":"N_prime","group_bonds":HPrimeBonds}])
        eps=1e-10
        H.data=H.data+(eps**.5)*np.eye(len(H.data))
        N.data=N.data+eps*np.eye(len(N.data))
        
        w,v=LA.eigh(.5*(H.data+H.data.conjugate().T),.5*(N.data+N.data.conjugate().T))
        self.E=w[0]
        M=np.reshape(v[:,0],self.mps.getMPSn(0).data.shape)
        M=M+np.random.normal(0,self.params["noise"], M.shape)
        for n, cMPS in enumerate(self.mps.MPS):
            if n%2==0:
                bonds=[cMPS.getBondByName(spin_bond_name(n)),cMPS.getBondByName(mps_bond_left_name(n,True,self.params["L"])),cMPS.getBondByName(mps_bond_right_name(n,True,self.params["L"]))]
                self.mps.MPS[n]=ten.Tensor(M,bonds)
        self.normalize_MPS_PBC()
        #minimize odd sites
        mpo=self.mpo.getMPOn(1)
        mpoid=self.mpoid.getMPOn(1)
        H=ten.contract(self.mps.getMPSnDagger(0),self.mpo.getMPOn(0));
        H=ten.contract(H,self.mps.getMPSn(0))
        N=ten.contract(self.mps.getMPSnDagger(0),self.mpoid.getMPOn(0));
        N=ten.contract(N,self.mps.getMPSn(0))
        for n in range(2,self.params["L"]):
            H=ten.contract(H,self.mps.getMPSnDagger(self.params["L"]+1-n))
            H=ten.contract(H,self.mpo.getMPOn(self.params["L"]+1-n))
            H=ten.contract(H,self.mps.getMPSn(self.params["L"]+1-n))
            N=ten.contract(N,self.mps.getMPSnDagger(self.params["L"]+1-n))
            N=ten.contract(N,self.mpoid.getMPOn(self.params["L"]+1-n))
            N=ten.contract(N,self.mps.getMPSn(self.params["L"]+1-n))
        H=ten.contract(H,mpo)
        N=ten.contract(N,mpoid)
        HBonds=[H.getBondByName(spin_bond_name(1)),H.getBondByName(mps_bond_left_name(1,True,self.params["L"])),H.getBondByName(mps_bond_right_name(1,True,self.params["L"]))]
        HPrimeBonds=[H.getBondByName(spin_bond_name(1),True),H.getBondByName(mps_bond_left_name(1,True,self.params["L"]),True),H.getBondByName(mps_bond_right_name(1,True,self.params["L"]),True)]
        H.groupTensor([{"bond_name":"H","group_bonds":HBonds}
                       ,{"bond_name":"H_prime","group_bonds":HPrimeBonds}])
        N.groupTensor([{"bond_name":"N","group_bonds":HBonds}
                       ,{"bond_name":"N_prime","group_bonds":HPrimeBonds}])
        H.data=H.data+(eps**.5)*np.eye(len(H.data))
        N.data=N.data+eps*np.eye(len(N.data))
                       
        w,v=LA.eigh(.5*(H.data+H.data.conjugate().T),.5*(N.data+N.data.conjugate().T))
        #self.E=(self.E+w[0])/2.0
        self.E=w[0]
        M=np.reshape(v[:,0],self.mps.getMPSn(1).data.shape)
        M=M+np.random.normal(0,self.params["noise"], M.shape)
        for n, cMPS in enumerate(self.mps.MPS):
            if n%2==1:
                bonds=[cMPS.getBondByName(spin_bond_name(n)),cMPS.getBondByName(mps_bond_left_name(n,True,self.params["L"])),cMPS.getBondByName(mps_bond_right_name(n,True,self.params["L"]))]
                self.mps.MPS[n]=ten.Tensor(M,bonds)
        self.normalize_MPS_PBC()

    def calcSz(self):
        sz=[]
        for n in range(self.params["L"]):
            mps=self.mps.getMPSn(n)
            mps_prime=ten.Tensor.TensorCopy(mps)
            mps_prime.getBondByName(spin_bond_name(n)).prime=True
            A=ten.contract(mps, mps_prime)
            SzTensor=ten.Tensor(spinmat.sigma_z,[ten.Bond(spin_bond_name(n),2),ten.Bond(spin_bond_name(n),2,True)])
            result=ten.contract(SzTensor,A).data
            sz.append(result)
        return np.sum(sz)/float(len(sz))
    def sweepRightToLeft(self,PBC=False):
        for n in reversed(range(self.params["L"])):
            self.minimizeH_MPS_Site(n,PBC)
            self.mps.makeRightCanonicalSite(n)
            if (n>0):
                self.updateHRight(n,PBC)
                if PBC:
                    self.updateNRight(n,PBC)
    def sweepLeftToRight(self,PBC=False):
        for n in (range(self.params["L"])):
            self.minimizeH_MPS_Site(n,PBC)
            if not PBC:
                self.mps.makeLeftCanonicalSite(n)
            if (n<self.params["L"]-1):
                self.updateHLeft(n,PBC)
                if PBC:
                    self.updateNLeft(n,PBC)
    def sweep(self,PBC=False):
        for s in range(self.params["sweeps"]):
            #if not PBC:
            self.sweepRightToLeft(PBC)
            self.sweepLeftToRight(PBC)
                #else:
                #self.minimizeH_MPS_PBC_TI()
            self.params["noise"]=0.5*self.params["noise"]
        return self.E