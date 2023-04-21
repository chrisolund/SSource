# -*- coding: utf-8 -*-

import numpy as np

import tensor as ten
import minimizeMPO as mini
import math
import copy
from scipy import linalg as LA
from scipy import optimize as opt
from scipy.sparse import diags

UId=np.eye(4).reshape(2,2,2,2)
sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0],[0,-1]])
s0=np.array([[1,0],[0,1]])
basis=[np.kron(sx,sx),np.kron(sx,sy),np.kron(sx,sz),np.kron(sx,s0),
       np.kron(sy,sx),np.kron(sy,sy),np.kron(sy,sz),np.kron(sy,s0),
       np.kron(sz,sx),np.kron(sz,sy),np.kron(sz,sz),np.kron(sz,s0),
       np.kron(s0,sx),np.kron(s0,sy),np.kron(s0,sz)]
basis2=[np.kron(sx,sx),np.kron(sx,sy),np.kron(sx,sz),np.kron(sx,s0),
       np.kron(sy,sx),np.kron(sy,sy),np.kron(sy,sz),np.kron(sy,s0),
       np.kron(sz,sx),np.kron(sz,sy),np.kron(sz,sz),np.kron(sz,s0),
       np.kron(s0,sx),np.kron(s0,sy),np.kron(s0,sz),np.kron(s0,s0)]
XIII=np.kron(np.kron(np.kron(sx,s0),s0),s0).reshape(2,2,2,2,2,2,2,2)
YIII=np.kron(np.kron(np.kron(sy,s0),s0),s0).reshape(2,2,2,2,2,2,2,2)
ZIII=np.kron(np.kron(np.kron(sz,s0),s0),s0).reshape(2,2,2,2,2,2,2,2)
XXII=np.kron(np.kron(np.kron(sx,sx),s0),s0).reshape(2,2,2,2,2,2,2,2)
YYII=np.kron(np.kron(np.kron(sy,sy),s0),s0).reshape(2,2,2,2,2,2,2,2)
ZZII=np.kron(np.kron(np.kron(sz,sz),s0),s0).reshape(2,2,2,2,2,2,2,2)
XYII=np.kron(np.kron(np.kron(sx,sy),s0),s0).reshape(2,2,2,2,2,2,2,2)
XZII=np.kron(np.kron(np.kron(sx,sz),s0),s0).reshape(2,2,2,2,2,2,2,2)
YXII=np.kron(np.kron(np.kron(sy,sx),s0),s0).reshape(2,2,2,2,2,2,2,2)
YZII=np.kron(np.kron(np.kron(sy,sz),s0),s0).reshape(2,2,2,2,2,2,2,2)
ZXII=np.kron(np.kron(np.kron(sz,sx),s0),s0).reshape(2,2,2,2,2,2,2,2)
ZYII=np.kron(np.kron(np.kron(sz,sy),s0),s0).reshape(2,2,2,2,2,2,2,2)
XIIIII=np.kron(np.kron(np.kron(np.kron(np.kron(sx,s0),s0),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IXIIII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,sx),s0),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIXIII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),sx),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIXII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),sx),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIIXI=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),s0),sx),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIIIX=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),s0),s0),sx).reshape(2,2,2,2,2,2,2,2,2,2,2,2)

ZIIIII=np.kron(np.kron(np.kron(np.kron(np.kron(sz,s0),s0),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IZIIII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,sz),s0),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIZIII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),sz),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIZII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),sz),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIIZI=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),s0),sz),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIIIZ=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),s0),s0),sz).reshape(2,2,2,2,2,2,2,2,2,2,2,2)

ZZIIII=np.kron(np.kron(np.kron(np.kron(np.kron(sz,sz),s0),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IZZIII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,sz),sz),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIZZII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),sz),sz),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIZZI=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),sz),sz),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIIZZ=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),s0),sz),sz).reshape(2,2,2,2,2,2,2,2,2,2,2,2)

XXIIII=np.kron(np.kron(np.kron(np.kron(np.kron(sx,sx),s0),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IXXIII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,sx),sx),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIXXII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),sx),sx),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIXXI=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),sx),sx),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIIIXX=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),s0),sx),sx).reshape(2,2,2,2,2,2,2,2,2,2,2,2)

XIII=np.kron(np.kron(np.kron(sx,s0),s0),s0).reshape(2,2,2,2,2,2,2,2)
IXII=np.kron(np.kron(np.kron(s0,sx),s0),s0).reshape(2,2,2,2,2,2,2,2)
IIXI=np.kron(np.kron(np.kron(s0,s0),sx),s0).reshape(2,2,2,2,2,2,2,2)
IIIX=np.kron(np.kron(np.kron(s0,s0),s0),sx).reshape(2,2,2,2,2,2,2,2)

ZIII=np.kron(np.kron(np.kron(sz,s0),s0),s0).reshape(2,2,2,2,2,2,2,2)
IZII=np.kron(np.kron(np.kron(s0,sz),s0),s0).reshape(2,2,2,2,2,2,2,2)
IIZI=np.kron(np.kron(np.kron(s0,s0),sz),s0).reshape(2,2,2,2,2,2,2,2)
IIIZ=np.kron(np.kron(np.kron(s0,s0),s0),sz).reshape(2,2,2,2,2,2,2,2)

ZZII=np.kron(np.kron(np.kron(sz,sz),s0),s0).reshape(2,2,2,2,2,2,2,2)
IZZI=np.kron(np.kron(np.kron(s0,sz),sz),s0).reshape(2,2,2,2,2,2,2,2)
IIZZ=np.kron(np.kron(np.kron(s0,s0),sz),sz).reshape(2,2,2,2,2,2,2,2)

XXII=np.kron(np.kron(np.kron(sx,sx),s0),s0).reshape(2,2,2,2,2,2,2,2)
IXXI=np.kron(np.kron(np.kron(s0,sx),sx),s0).reshape(2,2,2,2,2,2,2,2)
IIXX=np.kron(np.kron(np.kron(s0,s0),sx),sx).reshape(2,2,2,2,2,2,2,2)

IIIIII=np.kron(np.kron(np.kron(np.kron(np.kron(s0,s0),s0),s0),s0),s0).reshape(2,2,2,2,2,2,2,2,2,2,2,2)
IIII=np.kron(np.kron(np.kron(s0,s0),s0),s0).reshape(2,2,2,2,2,2,2,2)

class RGMPS:
    #def setParams(self,params):
    #    self.params=dict(params)
    def __init__(self,params,modelname="TFieldIsing"):
        """
        Initializes the RG circuit
        :return:
        Nothing
        """
        self.test=False
        self.reg=False
        self.reglayer=1
        self.params=params
        self.modelname=modelname
        self.coeffsOnA={}
        self.coeffsOffB={}
        self.coeffsOnC={}
        self.coeffsOffD={}
        #self.minresults=[opt.optimize.OptimizeResult()]
        self.minresultsA={}
        self.minresultsB={}
        self.minresultsC={}
        self.minresultsD={}
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        # unitary for turning "off" the J coupling
        self.UTurnOffB = {}
        self.UTurnOffD = {}

        # unitary for turning "on" the J coupling , 0.5 --> because we slip the h propagation between even and odd sites
        #self.UTurnOnWithH = ten.Tensor(genU.genURotSpinODE(self.params["J"],0.5*self.params["h"],self.params["T"],False,False),bonds)
        self.UTurnOnWithHA = {}
        self.UTurnOnWithHC = {}
        for i in range(self.params["L"]):
            self.coeffsOnA["s1_"+str(2*i)+"_"+str(2*i+1)]=np.array([0]*15)
            self.minresultsA["s1_"+str(2*i)+"_"+str(2*i+1)]=None#opt.optimize.OptimizeResult()
            self.UTurnOnWithHA["s1_"+str(2*i)+"_"+str(2*i+1)]=ten.Tensor(UId,bonds)
        for i in range(self.params["L"]/2):
            self.coeffsOffB["s1_"+str(4*i)+"_"+str(4*i+2)]=np.array([0]*15)
            self.minresultsB["s1_"+str(4*i)+"_"+str(4*i+2)]=None#opt.optimize.OptimizeResult()
            self.UTurnOffB["s1_"+str(4*i)+"_"+str(4*i+2)]=ten.Tensor(UId,bonds)
        for i in range(1,self.params["L"]):
            self.coeffsOnC["s1_"+str(2*i)+"_"+str(2*i-1)]=np.array([0]*15)
            self.minresultsC["s1_"+str(2*i)+"_"+str(2*i-1)]=None#opt.optimize.OptimizeResult()
            self.UTurnOnWithHC["s1_"+str(2*i)+"_"+str(2*i-1)]=ten.Tensor(UId,bonds)
        for i in range(1,self.params["L"]/2):
            self.coeffsOffD["s1_"+str(4*i)+"_"+str(4*i-2)]=np.array([0]*15)
            self.minresultsD["s1_"+str(4*i)+"_"+str(4*i-2)]=None#opt.optimize.OptimizeResult()
            self.UTurnOffD["s1_"+str(4*i)+"_"+str(4*i-2)]=ten.Tensor(UId,bonds)

        # "up" spin in the sigma_z basis on the right in bond
        bondspin=[ten.Bond("spin_in_right",2)]
        self.ancilla=np.array([1./math.sqrt(2), 1./math.sqrt(2)])
        if (modelname=="MFieldIsing" or modelname=="MFieldAndCoupling") and self.params["hz"]!=0:
            self.setancilla(math.atan2(self.params["hx"],self.params["hz"]),0)
        upspin_right=ten.Tensor(self.ancilla,bondspin)
        # Isometry for turning "on" the J coupling and rotation the on site spin
        self.IsoTurnOnWithH={}
        for i in range(self.params["L"]):
            self.IsoTurnOnWithH["s1_"+str(2*i)+"_"+str(2*i+1)]=ten.contract(self.UTurnOnWithHA["s1_"+str(2*i)+"_"+str(2*i+1)],upspin_right)
        #self.init = {"1":True}
        self.init={}

        self.ferro=False
        #if np.abs(self.params["h"])<=np.abs(self.params["J"]):
        #    self.ferro=True
        #else:
        #    self.ferro=False

        self.noise=1E-10


        PBC=self.params["PBC"]

        redo=1
        if self.ferro:
            while redo > 0:
                redo=0
                for i in range(8):
                    if self.MPS1Point(i,sz,8,self.MPS8) < 0:
                        redo=1
                        break
                if redo > 0:
                    mpssim8=mini.minimizeMPO(paramsMPS8,PBC)
                    mpssim8.initializeMinimization(PBC)
                    e8=mpssim8.sweep(PBC)
                    self.MPS8=mpssim8.mps
        '''

        for i in range(8):
            self.MPS8.MPS[i].changeBondName("spin_f"+str(i),"spin"+str(i))
        '''

        if modelname=="TFieldIsing":
            paramsMPS={"L":self.params["L"],"D":self.params["D"],"d":2,"DMPO":3,"h":self.params["h"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        elif modelname=="MFieldIsing":
            paramsMPS={"L":self.params["L"],"D":self.params["D"],"d":2,"DMPO":3,"hx":self.params["hx"],"hz":self.params["hz"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        elif modelname=="MFieldAndCoupling":
            paramsMPS={"L":self.params["L"],"D":self.params["D"],"d":2,"DMPO":4, "hx":self.params["hx"],"hz":self.params["hz"],"Jx":self.params["Jx"],"Jz":self.params["Jz"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        else:
            print("Invalid model name")
        mpssim=mini.minimizeMPO(paramsMPS,PBC,modelname)
        mpssim.initializeMinimization(PBC)
        self.e=mpssim.sweep(PBC)
        self.MPS=mpssim.mps

        '''
        redo=1
        if self.ferro:
            while redo > 0:
                redo=0
                for i in range(self.params["L"]):
                    if self.MPS1Point(i,sz,self.params["L"],self.MPS) < 0:
                        redo=1
                        break
                if redo > 0:
                    mpssim=mini.minimizeMPO(paramsMPS,PBC)
                    mpssim.initializeMinimization(PBC)
                    self.e=mpssim.sweep(PBC)
                    self.MPS=mpssim.mps
        '''


        self.MPSCircuitA=[]
        self.MPSCircuitB=[]
        self.MPSCircuitC=[]
        self.MPSCircuitD=[]

        self.optL=self.params["L"]

    #creates an n x n Haar random unitary
    def randomU(self,n):
        Q, R = LA.qr(np.random.random((n,n))+1j*np.random.random((n,n)))
        return np.dot(Q,np.diag(np.sign(np.diag(R))))

    def setancilla(self,theta,phi):
        self.ancilla=np.array([np.cos(theta/2),np.exp(1j*phi)*np.sin(theta/2)])
        #self.setcoeffsA(self.coeffsOnA[0],1)

    def setcoeffsA(self,coeffs,layer,leg1,ghost=False):
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        bondspin=[ten.Bond("spin_in_right",2)]
        upspin_right=ten.Tensor(self.ancilla,bondspin)
        if not ghost:
            self.coeffsOnA["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=np.array(coeffs)
            TT=ten.Tensor(self.Uparam(coeffs),bonds)
            self.UTurnOnWithHA["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=TT
            self.IsoTurnOnWithH["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=ten.contract(TT,upspin_right)
        else:
            self.coeffsOnA["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=np.array(coeffs)
            TT=ten.Tensor(self.Uparam(coeffs),bonds)
            self.UTurnOnWithHA["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=TT
            self.IsoTurnOnWithH["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=ten.contract(TT,upspin_right)

    def setcoeffsB(self,coeffs,layer,leg1,ghost=False):
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        if not ghost:
            self.coeffsOffB["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)]=np.array(coeffs)
            self.UTurnOffB["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)]=ten.Tensor(self.Uparam(coeffs),bonds)
        else:
            self.coeffsOffB["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)]=np.array(coeffs)
            self.UTurnOffB["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)]=ten.Tensor(self.Uparam(coeffs),bonds)

    def setcoeffsC(self,coeffs,layer,leg1,ghost=False):
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        if not ghost:
            self.coeffsOnC["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)]=np.array(coeffs)
            self.UTurnOnWithHC["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)]=ten.Tensor(self.Uparam(coeffs),bonds)
        else:
            self.coeffsOnC["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)]=np.array(coeffs)
            self.UTurnOnWithHC["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)]=ten.Tensor(self.Uparam(coeffs),bonds)

    def setcoeffsD(self,coeffs,layer,leg1,ghost=False):
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        if not ghost:
            self.coeffsOffD["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)]=np.array(coeffs)
            self.UTurnOffD["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)]=ten.Tensor(self.Uparam(coeffs),bonds)
        else:
            self.coeffsOffD["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)]=np.array(coeffs)
            self.UTurnOffD["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)]=ten.Tensor(self.Uparam(coeffs),bonds)


    #For use with gradient updates since these give new Us but not new Hs directly. Saved coeffs will temporarily be unupdated.
    def setUA(self,U,layer,leg1,ghost=False):
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        bondspin=[ten.Bond("spin_in_right",2)]
        upspin_right=ten.Tensor(self.ancilla,bondspin)
        TT=ten.Tensor(U,bonds)
        if not ghost:
            self.UTurnOnWithHA["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=TT
            self.IsoTurnOnWithH["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=ten.contract(TT,upspin_right)
        else:
            self.UTurnOnWithHA["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=TT
            self.IsoTurnOnWithH["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=ten.contract(TT,upspin_right)

    #sets correct coeffs for a given U (throwing away the identity component, so these will be off by a phase)
    def fixUA(self,layer,leg1):
        U=self.UTurnOnWithHA["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)].data
        cs=LA.solve(np.array(basis2).reshape(16,16).T,1j*LA.logm(U.reshape(4,4)).reshape(16)).real
        self.coeffsOnA["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=cs[0:15]
        if self.reg:
            U=self.UTurnOnWithHA["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)].data
            cs=LA.solve(np.array(basis2).reshape(16,16).T,1j*LA.logm(U.reshape(4,4)).reshape(16)).real
            self.coeffsOnA["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+1)]=cs[0:15]

    def setUB(self,U,layer,leg1,ghost=False):
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        if not ghost:
            self.UTurnOffB["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)]=ten.Tensor(U,bonds)
        else:
            self.UTurnOffB["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)]=ten.Tensor(U,bonds)

    def fixUB(self,layer,leg1):
        U=self.UTurnOffB["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)].data
        cs=LA.solve(np.array(basis2).reshape(16,16).T,1j*LA.logm(U.reshape(4,4)).reshape(16)).real
        self.coeffsOffB["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)]=cs[0:15]
        if self.reg:
            U=self.UTurnOffB["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)].data
            cs=LA.solve(np.array(basis2).reshape(16,16).T,1j*LA.logm(U.reshape(4,4)).reshape(16)).real
            self.coeffsOffB["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1+2)]=cs[0:15]

    def setUC(self,U,layer,leg1,ghost=False):
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        if not ghost:
            self.UTurnOnWithHC["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)]=ten.Tensor(U,bonds)
        else:
            self.UTurnOnWithHC["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)]=ten.Tensor(U,bonds)

    def fixUC(self,layer,leg1):
        U=self.UTurnOnWithHC["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)].data
        cs=LA.solve(np.array(basis2).reshape(16,16).T,1j*LA.logm(U.reshape(4,4)).reshape(16)).real
        self.coeffsOnC["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)]=cs[0:15]
        if self.reg:
            U=self.UTurnOnWithHC["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)].data
            cs=LA.solve(np.array(basis2).reshape(16,16).T,1j*LA.logm(U.reshape(4,4)).reshape(16)).real
            self.coeffsOnC["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-1)]=cs[0:15]

    def setUD(self,U,layer,leg1,ghost=False):
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        if not ghost:
            self.UTurnOffD["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)]=ten.Tensor(U,bonds)
        else:
            self.UTurnOffD["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)]=ten.Tensor(U,bonds)

    def fixUD(self,layer,leg1):
        U=self.UTurnOffD["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)].data
        cs=LA.solve(np.array(basis2).reshape(16,16).T,1j*LA.logm(U.reshape(4,4)).reshape(16)).real
        self.coeffsOffD["s"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)]=cs[0:15]
        if self.reg:
            U=self.UTurnOffD["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)].data
            cs=LA.solve(np.array(basis2).reshape(16,16).T,1j*LA.logm(U.reshape(4,4)).reshape(16)).real
            self.coeffsOffD["ghosts"+str(layer)+"_"+str(leg1)+"_"+str(leg1-2)]=cs[0:15]

    def setcoeffs(self,coeffs,layer):
        for i in range((2**(layer-1))*self.params["L"]):
            self.setcoeffsA(coeffs[0:15],layer,2*i)
        for i in range(((2**(layer-1))*self.params["L"])/2):
            self.setcoeffsB(coeffs[15:30],layer,4*i)
        for i in range(1,(2**(layer-1))*self.params["L"]):
            self.setcoeffsC(coeffs[30:45],layer,2*i)
        for i in range(1,((2**(layer-1))*self.params["L"])/2):
            self.setcoeffsD(coeffs[45:60],layer,4*i)
        self.init[str(layer)]=True

    #clones ghost coeffs/unitaries from normal coeffs/unitaries; some redundancy here to avoid screwing up the phases due to not storing an identity component
    #in the effective hamiltonians (which actually matters if U and U_ghost are allowed to be different)
    def setcoeffsghost(self,layer):
        for i in range((2**(layer-1))*self.params["L"]):
            self.setcoeffsA(self.coeffsOnA["s"+str(layer)+"_"+str(2*i)+"_"+str(2*i+1)],layer,2*i,ghost=True)
            self.UTurnOnWithHA["ghosts"+str(layer)+"_"+str(2*i)+"_"+str(2*i+1)]=copy.deepcopy(self.UTurnOnWithHA["s"+str(layer)+"_"+str(2*i)+"_"+str(2*i+1)])
            self.IsoTurnOnWithH["ghosts"+str(layer)+"_"+str(2*i)+"_"+str(2*i+1)]=copy.deepcopy(self.IsoTurnOnWithH["s"+str(layer)+"_"+str(2*i)+"_"+str(2*i+1)])
        for i in range(((2**(layer-1))*self.params["L"])/2):
            self.setcoeffsB(self.coeffsOffB["s"+str(layer)+"_"+str(4*i)+"_"+str(4*i+2)],layer,4*i,ghost=True)
            self.UTurnOffB["ghosts"+str(layer)+"_"+str(4*i)+"_"+str(4*i+2)]=copy.deepcopy(self.UTurnOffB["s"+str(layer)+"_"+str(4*i)+"_"+str(4*i+2)])
        for i in range(1,(2**(layer-1))*self.params["L"]):
            self.setcoeffsC(self.coeffsOnC["s"+str(layer)+"_"+str(2*i)+"_"+str(2*i-1)],layer,2*i,ghost=True)
            self.UTurnOnWithHC["ghosts"+str(layer)+"_"+str(2*i)+"_"+str(2*i-1)]=copy.deepcopy(self.UTurnOnWithHC["s"+str(layer)+"_"+str(2*i)+"_"+str(2*i-1)])
        for i in range(1,((2**(layer-1))*self.params["L"])/2):
            self.setcoeffsD(self.coeffsOffD["s"+str(layer)+"_"+str(4*i)+"_"+str(4*i-2)],layer,4*i,ghost=True)
            self.UTurnOffD["ghosts"+str(layer)+"_"+str(4*i)+"_"+str(4*i-2)]=copy.deepcopy(self.UTurnOffD["s"+str(layer)+"_"+str(4*i)+"_"+str(4*i-2)])
        self.init[str(layer)]=True

    #random coeffs *near 0* (width of order eps) and thus random unitaries *near the identity*
    def setcoeffsrandom(self,eps,layer,same=True):
        for i in range((2**(layer-1))*self.params["L"]):
            csA=np.random.normal(0,eps,15)
            self.setcoeffsA(csA,layer,2*i)
            if self.reg:
                if same:
                    self.setcoeffsA(csA,layer,2*i,ghost=True)
                else:
                    self.setcoeffsA(np.random.normal(0,eps,15),layer,2*i,ghost=True)
        for i in range(((2**(layer-1))*self.params["L"])/2):
            csB=np.random.normal(0,eps,15)
            self.setcoeffsB(csB,layer,4*i)
            if self.reg:
                if same:
                    self.setcoeffsB(csB,layer,4*i,ghost=True)
                else:
                    self.setcoeffsB(np.random.normal(0,eps,15),layer,4*i,ghost=True)
        for i in range(1,(2**(layer-1))*self.params["L"]):
            csC=np.random.normal(0,eps,15)
            self.setcoeffsC(csC,layer,2*i)
            if self.reg:
                if same:
                    self.setcoeffsC(csC,layer,2*i,ghost=True)
                else:
                    self.setcoeffsC(np.random.normal(0,eps,15),layer,2*i,ghost=True)
        for i in range(1,((2**(layer-1))*self.params["L"])/2):
            csD=np.random.normal(0,eps,15)
            self.setcoeffsD(csD,layer,4*i)
            if self.reg:
                if same:
                    self.setcoeffsD(csD,layer,4*i,ghost=True)
                else:
                    self.setcoeffsD(np.random.normal(0,eps,15),layer,4*i,ghost=True)
        self.init[str(layer)]=True

    #Haar random starting unitaries
    def setUstotallyrandom(self,layer,same=True):
        for i in range((2**(layer-1))*self.params["L"]):
            uA=self.randomU(4).reshape(2,2,2,2)
            self.setUA(uA,layer,2*i,ghost=False)
            if self.reg:
                if same:
                    self.setUA(uA,layer,2*i,ghost=True)
                else:
                    self.setUA(self.randomU(4).reshape(2,2,2,2),layer,2*i,ghost=True)
            self.fixUA(layer,2*i)
        for i in range(((2**(layer-1))*self.params["L"])/2):
            uB=self.randomU(4).reshape(2,2,2,2)
            self.setUB(uB,layer,4*i)
            if self.reg:
                if same:
                    self.setUB(uB,layer,4*i,ghost=True)
                else:
                    self.setUB(self.randomU(4).reshape(2,2,2,2),layer,4*i,ghost=True)
            self.fixUB(layer,4*i)
        for i in range(1,(2**(layer-1))*self.params["L"]):
            uC=self.randomU(4).reshape(2,2,2,2)
            self.setUC(uC,layer,2*i)
            if self.reg:
                if same:
                    self.setUC(uC,layer,2*i,ghost=True)
                else:
                    self.setUC(self.randomU(4).reshape(2,2,2,2),layer,2*i,ghost=True)
            self.fixUC(layer,2*i)
        for i in range(1,((2**(layer-1))*self.params["L"])/2):
            uD=self.randomU(4).reshape(2,2,2,2)
            self.setUD(uD,layer,4*i)
            if self.reg:
                if same:
                    self.setUD(uD,layer,4*i,ghost=True)
                else:
                    self.setUD(self.randomU(4).reshape(2,2,2,2),layer,4*i,ghost=True)
            self.fixUD(layer,4*i)
        self.init[str(layer)]=True

    def setcoeffsanalytic(self,layer):#sets unitaries to the leading order analytic answer for large h/J dropping all T terms
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            cs=-J/(8.0*h)*np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
            for i in range((2**(layer-1))*self.params["L"]):
                self.setcoeffsA(cs,layer,2*i)
            for i in range(((2**(layer-1))*self.params["L"])/2):
                self.setcoeffsB(-cs,layer,4*i)
            for i in range(1,(2**(layer-1))*self.params["L"]):
                self.setcoeffsC(cs,layer,2*i)
            for i in range(1,((2**(layer-1))*self.params["L"])/2):
                self.setcoeffsD(-cs,layer,4*i)
            if self.reg:
                for i in range((2**(layer-1))*self.params["L"]):
                    self.setcoeffsA(cs,layer,2*i,ghost=True)
                for i in range(((2**(layer-1))*self.params["L"])/2):
                    self.setcoeffsB(-cs,layer,4*i,ghost=True)
                for i in range(1,(2**(layer-1))*self.params["L"]):
                    self.setcoeffsC(cs,layer,2*i,ghost=True)
                for i in range(1,((2**(layer-1))*self.params["L"])/2):
                    self.setcoeffsD(-cs,layer,4*i,ghost=True)
            self.init[str(layer)]=True
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            h=(hx**2+hz**2)**.5
            eta=math.atan2(hz,hx)
            cxy=(Jx/(32.*h))*(7*np.sin(eta)+3*np.sin(3*eta))-(3*Jz/(8.*h))*np.sin(eta)*(np.cos(eta)**2)
            cyz=(3*Jx/(8.*h))*(np.sin(eta)**2)*np.cos(eta)-(Jz/(32.*h))*(7*np.cos(eta)-3*np.cos(3*eta))
            cs=np.array([0.0,cxy,0.0,0.0,cxy,0.0,cyz,0.0,0.0,cyz,0.0,0.0,0.0,0.0,0.0])
            for i in range((2**(layer-1))*self.params["L"]):
                self.setcoeffsA(cs,layer,2*i)
            for i in range(((2**(layer-1))*self.params["L"])/2):
                self.setcoeffsB(-cs,layer,4*i)
            for i in range(1,(2**(layer-1))*self.params["L"]):
                self.setcoeffsC(cs,layer,2*i)
            for i in range(1,((2**(layer-1))*self.params["L"])/2):
                self.setcoeffsD(-cs,layer,4*i)
            if self.reg:
                for i in range((2**(layer-1))*self.params["L"]):
                    self.setcoeffsA(cs,layer,2*i,ghost=True)
                for i in range(((2**(layer-1))*self.params["L"])/2):
                    self.setcoeffsB(-cs,layer,4*i,ghost=True)
                for i in range(1,(2**(layer-1))*self.params["L"]):
                    self.setcoeffsC(cs,layer,2*i,ghost=True)
                for i in range(1,((2**(layer-1))*self.params["L"])/2):
                    self.setcoeffsD(-cs,layer,4*i,ghost=True)
            self.init[str(layer)]=True
        else:
            print("Not implemented for this model!")

    def addnoisecoeffs(self,eps,layer):
        for i in range((2**(layer-1))*self.params["L"]):
            cA=self.coeffsOnA["s"+str(layer)+"_"+str(2*i)+"_"+str(2*i+1)];
            self.setcoeffsA(cA+eps*np.random.random(15),layer,2*i)
        for i in range(((2**(layer-1))*self.params["L"])/2):
            cB=self.coeffsOffB["s"+str(layer)+"_"+str(4*i)+"_"+str(4*i+2)]
            self.setcoeffsB(cB+eps*np.random.random(15),layer,4*i)
        for i in range(1,(2**(layer-1))*self.params["L"]):
            cC=self.coeffsOnC["s"+str(layer)+"_"+str(2*i)+"_"+str(2*i-1)]
            self.setcoeffsC(cC+eps*np.random.random(15),layer,2*i)
        for i in range(1,((2**(layer-1))*self.params["L"])/2):
            cD=self.coeffsOffD["s"+str(layer)+"_"+str(4*i)+"_"+str(4*i-2)]
            self.setcoeffsD(cD+eps*np.random.random(15),layer,4*i)

    def setcoeffsprevlayer(self,layer):
        if layer<2:
            print("There is no previous layer!")
            return None
        for i in range((2**(layer-2))*self.params["L"]):
            self.setcoeffsA(self.coeffsOnA["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i+1)],layer,4*i)
            self.setcoeffsA(self.coeffsOnA["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i+1)],layer,4*i+2)
        for i in range(((2**(layer-2))*self.params["L"])/2):
            self.setcoeffsB(self.coeffsOffB["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i+2)],layer,8*i)
            self.setcoeffsB(self.coeffsOffB["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i+2)],layer,8*i+4)
        for i in range(1,(2**(layer-2))*self.params["L"]/2):
            self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i-1)],layer,4*i-2)
            self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i-1)],layer,4*i)
        q=(2**(layer-2))*self.params["L"]/2
        self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*q)+"_"+str(2*q-1)],layer,4*q-2)
        self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*q)+"_"+str(2*q-1)],layer,4*q)
        self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*q)+"_"+str(2*q-1)],layer,4*q+2)
        for i in range(q+1,(2**(layer-2))*self.params["L"]):
            self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i-1)],layer,4*i)
            self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i-1)],layer,4*i+2)
        for i in range(1,((2**(layer-2))*self.params["L"])/4):
            self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i-2)],layer,8*i-4)
            self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i-2)],layer,8*i)
        q=(2**(layer-2))*self.params["L"]/4
        self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*q)+"_"+str(4*q-2)],layer,8*q-4)
        self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*q)+"_"+str(4*q-2)],layer,8*q)
        self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*q)+"_"+str(4*q-2)],layer,8*q+4)
        for i in range(q+1,((2**(layer-2))*self.params["L"])/2):
            self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i-2)],layer,8*i)
            self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i-2)],layer,8*i+4)

        if self.reg:
            for i in range((2**(layer-2))*self.params["L"]):
                self.setcoeffsA(self.coeffsOnA["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i+1)],layer,4*i,ghost=True)
                self.setcoeffsA(self.coeffsOnA["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i+1)],layer,4*i+2,ghost=True)
            for i in range(((2**(layer-2))*self.params["L"])/2):
                self.setcoeffsB(self.coeffsOffB["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i+2)],layer,8*i,ghost=True)
                self.setcoeffsB(self.coeffsOffB["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i+2)],layer,8*i+4,ghost=True)
            for i in range(1,(2**(layer-2))*self.params["L"]/2):
                self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i-1)],layer,4*i-2,ghost=True)
                self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i-1)],layer,4*i,ghost=True)
            q=(2**(layer-2))*self.params["L"]/2
            self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*q)+"_"+str(2*q-1)],layer,4*q-2,ghost=True)
            self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*q)+"_"+str(2*q-1)],layer,4*q,ghost=True)
            self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*q)+"_"+str(2*q-1)],layer,4*q+2,ghost=True)
            for i in range(q+1,(2**(layer-2))*self.params["L"]):
                self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i-1)],layer,4*i,ghost=True)
                self.setcoeffsC(self.coeffsOnC["s"+str(layer-1)+"_"+str(2*i)+"_"+str(2*i-1)],layer,4*i+2,ghost=True)
            for i in range(1,((2**(layer-2))*self.params["L"])/4):
                self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i-2)],layer,8*i-4,ghost=True)
                self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i-2)],layer,8*i,ghost=True)
            q=(2**(layer-2))*self.params["L"]/4
            self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*q)+"_"+str(4*q-2)],layer,8*q-4,ghost=True)
            self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*q)+"_"+str(4*q-2)],layer,8*q,ghost=True)
            self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*q)+"_"+str(4*q-2)],layer,8*q+4,ghost=True)
            for i in range(q+1,((2**(layer-2))*self.params["L"])/2):
                self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i-2)],layer,8*i,ghost=True)
                self.setcoeffsD(self.coeffsOffD["s"+str(layer-1)+"_"+str(4*i)+"_"+str(4*i-2)],layer,8*i+4,ghost=True)
        self.init[str(layer)]=True


    def infidelity4to8(self,coeffs,full=True):
        PBC=self.params["PBC"]
        coeffsA=coeffs[0:15]
        coeffsB=coeffs[15:30]
        if full:
            coeffsC=coeffs[30:45]
            coeffsD=coeffs[45:60]
        else:
            coeffsC=coeffs[0:15]
            coeffsD=coeffs[15:30]
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        self.UTurnOnWithHA = ten.Tensor(self.Uparam(coeffsA),bonds)
        self.UTurnOnWithHC = ten.Tensor(self.Uparam(coeffsC),bonds)
        self.UTurnOffB= ten.Tensor(self.Uparam(coeffsB),bonds)
        self.UTurnOffD= ten.Tensor(self.Uparam(coeffsD),bonds)
        bondspin=[ten.Bond("spin_in_right",2)]
        upspin_right=ten.Tensor(self.ancilla,bondspin)
        # Isometry for turning "on" the J coupling and rotation the on site spin
        self.IsoTurnOnWithH=ten.contract(self.UTurnOnWithHA,upspin_right)

        self.MPS4CircuitA=[]
        self.MPS4CircuitB=[]
        self.MPS4CircuitC=[]
        self.MPS4CircuitD=[]
        for i in range(4):
            OnA=self.buildTurnOnIso(1,2*i,2*i,2*i+1)
            OnA.changeBondName("spin"+str(i),"l0_"+str(2*i))
            if (not PBC and i==3):
                OnA.changeBondName("spin_f"+str(2*i+1),"l1_"+str(2*i+1))
            self.MPS4CircuitA.append(OnA)
        for i in range(2):
            OffB=self.buildTurnOff(2,4*i,4*i+2,4*i,4*i+2)
            if (not PBC and i==0):
                OffB.changeBondName("spin_f0","l2_0")
            self.MPS4CircuitB.append(OffB)
        for i in range(4):
            if (PBC or i>0):
                OnC=self.buildTurnOn(3,2*i,(2*i-1)%8,2*i,(2*i-1)%8)
                OnC.changeBondName("l1_"+str((2*i-1)%8),"l2_"+str((2*i-1)%8))
                OnC.changeBondName("spin_f"+str((2*i-1)%8),"l3_"+str((2*i-1)%8))
                if (not PBC and i==3):
                    OnC.changeBondName("spin_f"+str(2*i),"l3_"+str(2*i))
                self.MPS4CircuitC.append(OnC)
        for i in range(2):
            if (PBC or i>0):
                OnD=self.buildTurnOff(4,4*i,(4*i-2)%8,4*i,(4*i-2)%8)
                OnD.changeBondName("spin_f"+str(4*i),"l4_"+str(4*i))
                OnD.changeBondName("spin_f"+str((4*i-2)%8),"l4_"+str((4*i-2)%8))
                self.MPS4CircuitD.append(OnD)

        first=True
        for i in range(4):
            if(first):
                F=self.MPS4.getMPSn(i)
                first=False
            else:
                F=ten.contract(F,self.MPS4.getMPSn(i))
            if PBC:
                F=ten.contract(F,self.MPS4CircuitA[i])
                if(i%2==0):
                    F=ten.contract(F,self.MPS4CircuitB[i/2])
                F=ten.contract(F,self.MPS4CircuitC[i])
                if(i%2==0):
                    F=ten.contract(F,self.MPS4CircuitD[i/2])
            else:
                F=ten.contract(F,self.MPS4CircuitA[i])
                if(i%2==0):
                    F=ten.contract(F,self.MPS4CircuitB[i/2])
                if i>0:
                    F=ten.contract(F,self.MPS4CircuitC[i-1])
                    if(i%2==0):
                        F=ten.contract(F,self.MPS4CircuitD[i/2-1])
            bonds=[ten.Bond("spin_f"+str(2*i),2),ten.Bond("spin_f"+str(2*i),2,True)]
            unprimer=ten.Tensor(np.array([[1,0],[0,1]]),bonds)
            Adagn=ten.contract(self.MPS8.getMPSnDagger(2*i),unprimer)
            #print(i)
            #Adagn.printBonds()
            F=ten.contract(F,Adagn)
            bonds=[ten.Bond("spin_f"+str(2*i+1),2),ten.Bond("spin_f"+str(2*i+1),2,True)]
            unprimer=ten.Tensor(np.array([[1,0],[0,1]]),bonds)
            Adagn=ten.contract(self.MPS8.getMPSnDagger(2*i+1),unprimer)
            #Adagn.printBonds()
            F=ten.contract(F,Adagn)
        return 1-abs(np.complex_(F.data))**2

    def optimizeU4to8(self,full=True):
        if full:
            self.minresults=opt.minimize(self.infidelity4to8,np.array([0.]*60),full)
            self.coeffsOnA=self.minresults.x[0:15]
            self.coeffsOnC=self.minresults.x[30:45]
            self.coeffsOffB=self.minresults.x[15:30]
            self.coeffsOffD=self.minresults.x[45:60]
        else:
            self.minresults=opt.minimize(self.infidelity4to8,np.array([0.]*30),full)
            self.coeffsOnA=self.minresults.x[0:15]
            self.coeffsOnC=self.minresults.x[0:15]
            self.coeffsOffB=self.minresults.x[15:30]
            self.coeffsOffD=self.minresults.x[15:30]

        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        self.UTurnOnWithHA = ten.Tensor(self.Uparam(self.coeffsOnA),bonds)
        self.UTurnOnWithHC = ten.Tensor(self.Uparam(self.coeffsOnC),bonds)
        self.UTurnOffB = ten.Tensor(self.Uparam(self.coeffsOffB),bonds)
        self.UTurnOffD = ten.Tensor(self.Uparam(self.coeffsOffD),bonds)
        bondspin=[ten.Bond("spin_in_right",2)]
        upspin_right=ten.Tensor(self.ancilla,bondspin)
        #Isometry for turning "on" the J coupling and rotation the on site spin
        self.IsoTurnOnWithH=ten.contract(self.UTurnOnWithHA,upspin_right)

    def optimizeUOld(self,L0=4,full=True,local=False,guess=np.array([0.]*60),gtol=1e-5):
        self.optL=L0

        if self.modelname=="TFieldIsing":
            paramsMPSL={"L":L0,"D":self.params["D"],"d":2,"DMPO":3,"h":self.params["h"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        elif self.modelname=="MFieldIsing":
            paramsMPSL={"L":L0,"D":self.params["D"],"d":2,"DMPO":3,"hx":self.params["hx"],"hz":self.params["hz"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        elif self.modelname=="MFieldAndCoupling":
            paramsMPSL={"L":L0,"D":self.params["D"],"d":2,"DMPO":4, "hx":self.params["hx"],"hz":self.params["hz"],"Jx":self.params["Jx"],"Jz":self.params["Jz"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        else:
            print("Invalid model name")

        #paramsMPSL={"L":L0,"D":self.params["D"],"d":2,"DMPO":3,"h":self.params["h"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        # Do MPS
        PBC=self.params["PBC"]
        mpssimL=mini.minimizeMPO(paramsMPSL,PBC,self.modelname)
        mpssimL.initializeMinimization(PBC)
        eL=mpssimL.sweep(PBC)
        self.MPSL=mpssimL.mps

        if self.modelname=="TFieldIsing":
            paramsMPS2L={"L":2*L0,"D":self.params["D"],"d":2,"DMPO":3,"h":self.params["h"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        elif self.modelname=="MFieldIsing":
            paramsMPS2L={"L":2*L0,"D":self.params["D"],"d":2,"DMPO":3,"hx":self.params["hx"],"hz":self.params["hz"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        elif self.modelname=="MFieldAndCoupling":
            paramsMPS2L={"L":2*L0,"D":self.params["D"],"d":2,"DMPO":4, "hx":self.params["hx"],"hz":self.params["hz"],"Jx":self.params["Jx"],"Jz":self.params["Jz"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        else:
            print("Invalid model name")

        #paramsMPS2L={"L":2*L0,"D":self.params["D"],"d":2,"DMPO":3,"h":self.params["h"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
        # Do MPS
        mpssim2L=mini.minimizeMPO(paramsMPS2L,PBC,self.modelname)
        mpssim2L.initializeMinimization(PBC)
        e2L=mpssim2L.sweep(PBC)
        self.MPS2L=mpssim2L.mps



        if full:
            #self.minresults=opt.basinhopping(self.infidelityLto2L,np.array([0.]*60),T=.01,minimizer_kwargs={"args":(full)})
            #self.minresults=opt.minimize(self.infidelityLto2L,guess[0:60],full)
            if not local:
                self.minresults=opt.minimize(self.infidelityLto2L,guess[0:60],full,options={"gtol":gtol})
            else:
                if self.modelname=="TFieldIsing":
                    qs=self.genqs(2*L0)
                    ws=self.genws(self.params["h"],self.params["J"],qs)
                    us=self.genus(self.params["h"],self.params["J"],qs,ws,2*L0)
                    vs=self.genvs(self.params["h"],self.params["J"],qs,ws,2*L0)
                    self.Xtar=self.XExactTFIM(qs,ws,us,vs,2*L0)
                    self.Ytar=0.
                    self.Ztar=0.
                    self.XXtar=self.XXExactTFIM(qs,ws,us,vs,2*L0,1)
                    self.YYtar=self.YYExactTFIM(qs,ws,us,vs,2*L0,1)
                    self.ZZtar=self.ZZExactTFIM(qs,ws,us,vs,2*L0,1)
                    self.XYtar=0.
                    self.YZtar=0.
                    self.ZXtar=0.
                    self.XZtar=0.
                    self.YXtar=0.
                    self.ZYtar=0.
                else:
                    tx=0.
                    ty=0.
                    tz=0.
                    for j in range(2*L0):
                        tx+=self.MPS1Point(j,sx,2*L0,self.MPS2L)
                        ty+=self.MPS1Point(j,sy,2*L0,self.MPS2L)
                        tz+=self.MPS1Point(j,sz,2*L0,self.MPS2L)
                    self.Xtar=tx/(2*L0)
                    self.Ytar=ty/(2*L0)
                    self.Ztar=tz/(2*L0)
                    txx=0.
                    tyy=0.
                    tzz=0.
                    txy=0.
                    tyz=0.
                    tzx=0.
                    txz=0.
                    tyx=0.
                    tzy=0.
                    for j in range(2*L0):
                        txx+=self.MPS2Point(j,(j+1)%(2*L0),sx,sx,2*L0,self.MPS2L)
                        tyy+=self.MPS2Point(j,(j+1)%(2*L0),sy,sy,2*L0,self.MPS2L)
                        tzz+=self.MPS2Point(j,(j+1)%(2*L0),sz,sz,2*L0,self.MPS2L)
                        txy+=self.MPS2Point(j,(j+1)%(2*L0),sx,sy,2*L0,self.MPS2L)
                        tyz+=self.MPS2Point(j,(j+1)%(2*L0),sy,sz,2*L0,self.MPS2L)
                        tzx+=self.MPS2Point(j,(j+1)%(2*L0),sz,sx,2*L0,self.MPS2L)
                        txz+=self.MPS2Point(j,(j+1)%(2*L0),sx,sz,2*L0,self.MPS2L)
                        tyx+=self.MPS2Point(j,(j+1)%(2*L0),sy,sx,2*L0,self.MPS2L)
                        tzy+=self.MPS2Point(j,(j+1)%(2*L0),sz,sy,2*L0,self.MPS2L)
                    self.XXtar=txx/(2*L0)
                    self.YYtar=tyy/(2*L0)
                    self.ZZtar=tzz/(2*L0)
                    self.XYtar=txy/(2*L0)
                    self.YZtar=tyz/(2*L0)
                    self.ZXtar=tzx/(2*L0)
                    self.XZtar=txz/(2*L0)
                    self.YXtar=tyx/(2*L0)
                    self.ZYtar=tzy/(2*L0)
                self.minresults=opt.minimize(self.localErrorLto2L,guess[0:60],full,options={"gtol":gtol})
            self.coeffsOnA=self.minresults.x[0:15]
            self.coeffsOnC=self.minresults.x[30:45]
            self.coeffsOffB=self.minresults.x[15:30]
            self.coeffsOffD=self.minresults.x[45:60]
        else:
            #self.minresults=opt.basinhopping(self.infidelityLto2L,np.array([0.]*30),T=.01,minimizer_kwargs={"args":(full)})
            #self.minresults=opt.minimize(self.infidelityLto2L,guess[0:30],full)
            self.minresults=opt.minimize(self.infidelityLto2L,guess[0:30],full,options={"gtol":gtol})
            self.coeffsOnA=self.minresults.x[0:15]
            self.coeffsOnC=self.minresults.x[0:15]
            self.coeffsOffB=self.minresults.x[15:30]
            self.coeffsOffD=self.minresults.x[15:30]

        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        self.UTurnOnWithHA = ten.Tensor(self.Uparam(self.coeffsOnA),bonds)
        self.UTurnOnWithHC = ten.Tensor(self.Uparam(self.coeffsOnC),bonds)
        self.UTurnOffB = ten.Tensor(self.Uparam(self.coeffsOffB),bonds)
        self.UTurnOffD = ten.Tensor(self.Uparam(self.coeffsOffD),bonds)
        bondspin=[ten.Bond("spin_in_right",2)]
        upspin_right=ten.Tensor(self.ancilla,bondspin)
        #Isometry for turning "on" the J coupling and rotation the on site spin
        self.IsoTurnOnWithH=ten.contract(self.UTurnOnWithHA,upspin_right)

    def optimizeUTI(self,superlayer,guess=np.array([0.]*60),gtol=1e-5):
        if len(self.minresults)==superlayer-1:
            self.minresults.append(opt.minimize(self.energy,guess[0:60],superlayer,options={"gtol":gtol}))
        else:
            self.minresults[superlayer-1]=opt.minimize(self.energy,guess[0:60],superlayer,options={"gtol":gtol})
        self.setcoeffs(self.minresults[superlayer-1].x,superlayer)

    def optimizeU(self,superlayer,sweeps,guess=np.array([0.]*60),gtol=5e-6):
        if str(superlayer) not in self.init:
            if superlayer==1:
                self.setcoeffs(guess,superlayer)
            else:
                self.setcoeffsprevlayer(superlayer)
        #print(self.calcenergy(superlayer))
        Es=[]
        for s in range(sweeps):
            #print("sweep "+str(s)+":")
            for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
                curr=self.coeffsOffD["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i-2)]
                self.minresultsD["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i-2)]=opt.minimize(self.energyD,curr,(superlayer,4*i),options={"gtol":self.tolD(superlayer,4*i,gtol)})
            #print("D:")
            #print(self.calcenergy(superlayer))
            for i in range(1,(2**(superlayer-1))*self.params["L"]):
                curr=self.coeffsOnC["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i-1)]
                self.minresultsC["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i-1)]=opt.minimize(self.energyC,curr,(superlayer,2*i),options={"gtol":self.tolC(superlayer,2*i,gtol)})
            #print("C:")
            #print(self.calcenergy(superlayer))
            for i in range(((2**(superlayer-1))*self.params["L"])/2):
                curr=self.coeffsOffB["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i+2)]
                self.minresultsB["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i+2)]=opt.minimize(self.energyB,curr,(superlayer,4*i),options={"gtol":self.tolB(superlayer,4*i,gtol)})
            #print("B:")
            #print(self.calcenergy(superlayer))
            for i in range((2**(superlayer-1))*self.params["L"]):
                curr=self.coeffsOnA["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i+1)]
                self.minresultsA["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i+1)]=opt.minimize(self.energyA,curr,(superlayer,2*i),options={"gtol":self.tolA(superlayer,2*i,gtol)})
            #print("A:")
            E=self.calcenergy(superlayer)
            print(E)
            Es.append(E)
        return Es;


    def optimizeUGrad(self,superlayer,sweeps,noise=0.0,decay=0.0,suppress=False,rand=False):
        if str(superlayer) not in self.init:
            if superlayer==1:
                #self.setcoeffsrandom(.001,superlayer)
                if rand:
                    self.setcoeffsrandom(noise,superlayer)
                else:
                    self.setcoeffsanalytic(superlayer)
            else:
                self.setcoeffsprevlayer(superlayer)
        #print(self.calcenergy(superlayer))
        Es=[]
        if self.test:
            negDef=False
        else:
            negDef=True
        temp=self.test
        for s in range(sweeps):
            negCheck=True
            #print("sweep "+str(s)+":")
            for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
                W=self.DenergyD(superlayer,4*i,ghost=False).reshape(4,4).conjugate()
                U,S,V=np.linalg.svd(W)
                if not negDef:
                    #bnds=[ten.Bond("l4_"+str(4*i),2,True),ten.Bond("l4_"+str(4*i-2),2,True), ten.Bond("l3_"+str(4*i),2,True),ten.Bond("l3_"+str(4*i-2),2,True)]
                    #WW=ten.Tensor(W.reshape(2,2,2,2),bnds)
                    #QQ=self.buildTurnOff(4,4*i,4*i-2,4*i,4*i-2,superlayer,True)
                    #pE=float(ten.contract(WW,QQ).data.real)
                    pE=self.energyD(superlayer,4*i).real
                    if pE>=0:
                        negCheck=False
                U_n=self.Uparam(noise*np.random.normal(0,1.0,15)).reshape(4,4)
                #if step==1:
                self.setUD(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=False)
                #self.setUD(np.dot(U_n,np.dot(U,V)).reshape(2,2,2,2),superlayer,4*i,ghost=False)
                #else:
                    #U_old=self.UTurnOffD["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i-2)].data.reshape(4,4)
                    #U_interp=LA.fractional_matrix_power(np.dot(U,V),step).dot(LA.fractional_matrix_power(U_old,1-step))
                    #self.setUD(np.dot(U_interp,U_n).reshape(2,2,2,2),superlayer,4*i)
            #print("D:")
            #print(self.calcenergy(superlayer))
            for i in range(1,(2**(superlayer-1))*self.params["L"]):
                W=self.DenergyC(superlayer,2*i,ghost=False).reshape(4,4).conjugate()
                U,S,V=np.linalg.svd(W)
                if not negDef:
                    #bnds=[ten.Bond("l3_"+str(2*i),2,True),ten.Bond("l3_"+str(2*i-1),2,True), ten.Bond("l2_"+str(2*i),2,True),ten.Bond("l2_"+str(2*i-1),2,True)]
                    #WW=ten.Tensor(W.reshape(2,2,2,2),bnds)
                    #QQ=self.buildTurnOn(3,2*i,2*i-1,2*i,2*i-1,superlayer,True)
                    #pE=float(ten.contract(WW,QQ).data.real)
                    pE=self.energyC(superlayer,2*i).real
                    if pE>=0:
                        negCheck=False
                U_n=self.Uparam(noise*np.random.normal(0,1.0,15)).reshape(4,4)
                #if step==1:
                self.setUC(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=False)
                #self.setUC(np.dot(U_n,np.dot(U,V)).reshape(2,2,2,2),superlayer,2*i,ghost=False)
                #else:
                    #U_old=self.UTurnOnWithHC["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i-1)].data.reshape(4,4)
                    #U_interp=LA.fractional_matrix_power(np.dot(U,V),step).dot(LA.fractional_matrix_power(U_old,1-step))
                    #self.setUC(np.dot(U_interp,U_n).reshape(2,2,2,2),superlayer,2*i)
            #print("C:")
            #print(self.calcenergy(superlayer))
            for i in range(((2**(superlayer-1))*self.params["L"])/2):
                W=self.DenergyB(superlayer,4*i,ghost=False).reshape(4,4).conjugate()
                U,S,V=np.linalg.svd(W)
                if not negDef:
                    #bnds=[ten.Bond("l2_"+str(4*i),2,True),ten.Bond("l2_"+str(4*i+2),2,True), ten.Bond("l1_"+str(4*i),2,True),ten.Bond("l1_"+str(4*i+2),2,True)]
                    #WW=ten.Tensor(W.reshape(2,2,2,2),bnds)
                    #QQ=self.buildTurnOff(2,4*i,4*i+2,4*i,4*i+2,superlayer,True)
                    #pE=float(ten.contract(WW,QQ).data.real)
                    pE=self.energyB(superlayer,4*i).real
                    if pE>=0:
                        negCheck=False
                U_n=self.Uparam(noise*np.random.normal(0,1.0,15)).reshape(4,4)
                #if step==1:
                self.setUB(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=False)
                #self.setUB(np.dot(U_n,np.dot(U,V)).reshape(2,2,2,2),superlayer,4*i,ghost=False)
                #else:
                    #U_old=self.UTurnOffB["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i+2)].data.reshape(4,4)
                    #U_interp=LA.fractional_matrix_power(np.dot(U,V),step).dot(LA.fractional_matrix_power(U_old,1-step))
                    #self.setUB(np.dot(U_interp,U_n).reshape(2,2,2,2),superlayer,4*i)
            #print("B:")
            #print(self.calcenergy(superlayer))
            for i in range((2**(superlayer-1))*self.params["L"]):
                W=np.kron(self.DenergyA(superlayer,2*i,ghost=False),self.ancilla).reshape(4,4).conjugate()
                U,S,V=np.linalg.svd(W)
                if not negDef:
                    #bnds=[ten.Bond("l1_"+str(2*i),2,True),ten.Bond("l1_"+str(2*i+1),2,True), ten.Bond("l0_"+str(2*i),2,True),ten.Bond("l0_"+str(2*i+1),2,True)]
                    #WW=ten.Tensor(W.reshape(2,2,2,2),bnds)
                    #QQ=self.buildTurnOn(1,2*i,2*i+1,2*i,2*i+1,superlayer,True)
                    #pE=float(ten.contract(WW,QQ).data.real)
                    pE=self.energyA(superlayer,2*i).real
                    if pE>=0:
                        negCheck=False
                U_n=self.Uparam(noise*np.random.normal(0,1.0,15)).reshape(4,4)
                #if step==1:
                self.setUA(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=False)
                #self.setUA(np.dot(U_n,np.dot(U,V)).reshape(2,2,2,2),superlayer,2*i,ghost=False)
                #else:
                    #U_old=self.UTurnOnWithHA["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i+1)].data.reshape(4,4)
                    #U_interp=LA.fractional_matrix_power(np.dot(U,V),step).dot(LA.fractional_matrix_power(U_old,1-step))
                    #self.setUA(np.dot(U_interp,U_n).reshape(2,2,2,2),superlayer,2*i)
            #print("A:")
            E=self.calcenergy(superlayer)
            if not suppress:
                print(E)
            Es.append(E)
            noise*=decay
            if negCheck:
                negDef=True
                self.test=False
        self.test=temp
        for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
            self.fixUD(superlayer,4*i)
        for i in range(1,(2**(superlayer-1))*self.params["L"]):
            self.fixUC(superlayer,2*i)
        for i in range(((2**(superlayer-1))*self.params["L"])/2):
            self.fixUB(superlayer,4*i)
        for i in range((2**(superlayer-1))*self.params["L"]):
            self.fixUA(superlayer,2*i)
        return Es;

    def optimizeUReg(self,superlayer,sweeps,noise=0.0,decay=0.0,suppress=False,rand=False,same=True,lam=1.0):
        self.reg=True
        self.reglayer=superlayer
        if str(superlayer) not in self.init:
            if superlayer==1:
                #self.setcoeffsrandom(.001,superlayer)
                if rand:
                    self.setcoeffsrandom(noise,superlayer,same)
                else:
                    self.setcoeffsanalytic(superlayer)
            else:
                self.setcoeffsprevlayer(superlayer)
        elif same:
            self.setcoeffsghost(superlayer)
        Es=[]
        for s in range(sweeps):
            #print("sweep "+str(s)+":")
            for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
                W=-self.DenergyD(superlayer,4*i,ghost=False).reshape(4,4).conjugate()+lam*self.UTurnOffD["ghosts"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i-2)].data.reshape(4,4).conjugate()
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                #bnds=[ten.Bond("l4_"+str(4*i),2,False),ten.Bond("l4_"+str(4*i-2),2,False), ten.Bond("l3_"+str(4*i),2,False),ten.Bond("l3_"+str(4*i-2),2,False)]
                #WW=ten.Tensor(W.conjugate().T.reshape(2,2,2,2),bnds)
                #QQ=self.buildTurnOff(4,4*i,4*i-2,4*i,4*i-2,superlayer,False)
                #pE=-complex(ten.contract(WW,QQ).data)
                #upd=ten.Tensor(np.dot(U,V).reshape(2,2,2,2),bnds)
                #pEupd=-complex(ten.contract(WW,upd).data)
                #print(self.energyD([0]*15,1,4*i,False))
                #print("deltaE_D = "+str(pEupd-pE))
                #self.reg=False
                #E1=self.calcenergy(1)
                #pE1=self.energyD([0]*15,1,4*i,False)
                #print(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2))
                #print(U_n)
                self.setUD(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=False)
                #print(self.UTurnOffD["s1_"+str(4*i)+"_"+str(4*i-2)].data)
                #E2=self.calcenergy(1)
                #pE2=self.energyD([0]*15,1,4*i,False)
                #self.reg=True
                #print(E2-E1)
                #print(pE2-pE1)
                #print("")

                W=-self.DenergyD(superlayer,4*i,ghost=True).reshape(4,4)+lam*self.UTurnOffD["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i-2)].data.reshape(4,4)
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUD(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=True)

            #print("D:")
            #print(self.calcenergy(superlayer))
            for i in range(1,(2**(superlayer-1))*self.params["L"]):
                W=-self.DenergyC(superlayer,2*i,ghost=False).reshape(4,4).conjugate()+lam*self.UTurnOnWithHC["ghosts"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i-1)].data.reshape(4,4).conjugate()
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                #bnds=[ten.Bond("l3_"+str(2*i),2,False),ten.Bond("l3_"+str(2*i-1),2,False), ten.Bond("l2_"+str(2*i),2,False),ten.Bond("l2_"+str(2*i-1),2,False)]
                #WW=ten.Tensor(W.conjugate().T.reshape(2,2,2,2),bnds)
                #QQ=self.buildTurnOn(3,2*i,2*i-1,2*i,2*i-1,superlayer,False)
                #pE=-complex(ten.contract(WW,QQ).data)
                #print(self.energyC([0]*15,1,2*i,False))
                #print("pE_C =?= "+str(pE))
                self.setUC(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=False)

                W=-self.DenergyC(superlayer,2*i,ghost=True).reshape(4,4)+lam*self.UTurnOnWithHC["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i-1)].data.reshape(4,4)
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUC(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=True)
            #print("C:")
            #print(self.calcenergy(superlayer))
            for i in range(((2**(superlayer-1))*self.params["L"])/2):
                W=-self.DenergyB(superlayer,4*i,ghost=False).reshape(4,4).conjugate()+lam*self.UTurnOffB["ghosts"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i+2)].data.reshape(4,4).conjugate()
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUB(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=False)

                W=-self.DenergyB(superlayer,4*i,ghost=True).reshape(4,4)+lam*self.UTurnOffB["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i+2)].data.reshape(4,4)
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUB(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=True)
            #print("B:")
            #print(self.calcenergy(superlayer))
            for i in range((2**(superlayer-1))*self.params["L"]):
                W=-np.kron(self.DenergyA(superlayer,2*i,ghost=False),self.ancilla).reshape(4,4).conjugate()+lam*self.UTurnOnWithHA["ghosts"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i+1)].data.reshape(4,4).conjugate()
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUA(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=False)

                W=-np.kron(self.DenergyA(superlayer,2*i,ghost=True),self.ancilla).reshape(4,4)+lam*self.UTurnOnWithHA["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i+1)].data.reshape(4,4)
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUA(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=True)
            #print("A:")
            E=self.calcenergy(superlayer)
            if not suppress:
                print(E)
            Es.append(E)
            noise*=decay
            #print(self.UVerror(superlayer))
        for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
            self.fixUD(superlayer,4*i)
        for i in range(1,(2**(superlayer-1))*self.params["L"]):
            self.fixUC(superlayer,2*i)
        for i in range(((2**(superlayer-1))*self.params["L"])/2):
            self.fixUB(superlayer,4*i)
        for i in range((2**(superlayer-1))*self.params["L"]):
            self.fixUA(superlayer,2*i)
        self.reg=False
        return Es;

    def optimizeUReg2(self,superlayer,sweeps,uvsweeps,noise=0.0,decay=0.0,suppress=False,rand=False,lam=1.0,tol=1e-10):
        self.reg=True
        self.reglayer=superlayer
        if str(superlayer) not in self.init:
            if superlayer==1:
                #self.setcoeffsrandom(.001,superlayer)
                if rand:
                    self.setcoeffsrandom(noise,superlayer,True)
                else:
                    self.setcoeffsanalytic(superlayer)
            else:
                self.setcoeffsprevlayer(superlayer)
        else:
            self.setcoeffsghost(superlayer)
        Es=[]
        maxuvsweeps=0;
        actualuvsweeps=0;
        for s in range(sweeps):
            #print("sweep "+str(s)+":")
            maxuvsweeps+=uvsweeps
            actualuvsweeps+=uvsweeps
            for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
                for suv in range(uvsweeps):
                    W=-self.DenergyD(superlayer,4*i,ghost=True).reshape(4,4)+(2**suv)*lam*self.UTurnOffD["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i-2)].data.reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    #U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    #self.setUD(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=True)
                    self.setUD(np.dot(U,V).reshape(2,2,2,2),superlayer,4*i,ghost=True)

                    W=-self.DenergyD(superlayer,4*i,ghost=False).reshape(4,4).conjugate()+(2**suv)*lam*self.UTurnOffD["ghosts"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i-2)].data.reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    #U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    #self.setUD(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=False)
                    self.setUD(np.dot(U,V).reshape(2,2,2,2),superlayer,4*i,ghost=False)

                    UU=self.UTurnOffD["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i-2)].data.reshape(4,4)
                    VV=self.UTurnOffD["ghosts"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i-2)].data.reshape(4,4).conjugate().T
                    #print(np.abs(4-np.trace(np.dot(UU,VV))))
                    if np.abs(4-np.trace(np.dot(UU,VV)))<tol:
                        actualuvsweeps-=(uvsweeps-suv-1)
                        break
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                U_upd=np.dot(UU,U_n)
                self.setUD(U_upd.reshape(2,2,2,2),superlayer,4*i,True)
                self.setUD(U_upd.reshape(2,2,2,2),superlayer,4*i,False)
                #self.setUD(UU.reshape(2,2,2,2),superlayer,4*i,True)
                #print("")


            #print("D:")
            #print(self.calcenergy(superlayer))
            for i in range(1,(2**(superlayer-1))*self.params["L"]):
                maxuvsweeps+=uvsweeps
                actualuvsweeps+=uvsweeps
                for suv in range(uvsweeps):
                    W=-self.DenergyC(superlayer,2*i,ghost=True).reshape(4,4)+(2**suv)*lam*self.UTurnOnWithHC["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i-1)].data.reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    #U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    #self.setUC(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=True)
                    self.setUC(np.dot(U,V).reshape(2,2,2,2),superlayer,2*i,ghost=True)

                    W=-self.DenergyC(superlayer,2*i,ghost=False).reshape(4,4).conjugate()+(2**suv)*lam*self.UTurnOnWithHC["ghosts"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i-1)].data.reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    #U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    #self.setUC(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=False)
                    self.setUC(np.dot(U,V).reshape(2,2,2,2),superlayer,2*i,ghost=False)

                    UU=self.UTurnOnWithHC["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i-1)].data.reshape(4,4)
                    VV=self.UTurnOnWithHC["ghosts"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i-1)].data.reshape(4,4).conjugate().T
                    if np.abs(4-np.trace(np.dot(UU,VV)))<tol:
                        actualuvsweeps-=(uvsweeps-suv-1)
                        break
                    #print(np.abs(4-np.trace(np.dot(UU,VV))))
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                U_upd=np.dot(UU,U_n)
                self.setUC(U_upd.reshape(2,2,2,2),superlayer,2*i,True)
                self.setUC(U_upd.reshape(2,2,2,2),superlayer,2*i,False)
                #self.setUC(UU.reshape(2,2,2,2),superlayer,2*i,True)
                #print("")
            #print("C:")
            #print(self.calcenergy(superlayer))
            for i in range(((2**(superlayer-1))*self.params["L"])/2):
                maxuvsweeps+=uvsweeps
                actualuvsweeps+=uvsweeps
                for suv in range(uvsweeps):
                    W=-self.DenergyB(superlayer,4*i,ghost=True).reshape(4,4)+(2**suv)*lam*self.UTurnOffB["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i+2)].data.reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    #U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    #self.setUB(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=True)
                    self.setUB(np.dot(U,V).reshape(2,2,2,2),superlayer,4*i,ghost=True)

                    W=-self.DenergyB(superlayer,4*i,ghost=False).reshape(4,4).conjugate()+(2**suv)*lam*self.UTurnOffB["ghosts"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i+2)].data.reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    #U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    #self.setUB(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=False)
                    self.setUB(np.dot(U,V).reshape(2,2,2,2),superlayer,4*i,ghost=False)

                    UU=self.UTurnOffB["s"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i+2)].data.reshape(4,4)
                    VV=self.UTurnOffB["ghosts"+str(superlayer)+"_"+str(4*i)+"_"+str(4*i+2)].data.reshape(4,4).conjugate().T
                    if np.abs(4-np.trace(np.dot(UU,VV)))<tol:
                        actualuvsweeps-=(uvsweeps-suv-1)
                        break
                    #print(np.abs(4-np.trace(np.dot(UU,VV))))

                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                U_upd=np.dot(UU,U_n)
                self.setUB(U_upd.reshape(2,2,2,2),superlayer,4*i,True)
                self.setUB(U_upd.reshape(2,2,2,2),superlayer,4*i,False)
                #self.setUB(UU.reshape(2,2,2,2),superlayer,4*i,True)
                #print("")
            #print("B:")
            #print(self.calcenergy(superlayer))
            for i in range((2**(superlayer-1))*self.params["L"]):
                maxuvsweeps+=uvsweeps
                actualuvsweeps+=uvsweeps
                for suv in range(uvsweeps):
                    W=-np.kron(self.DenergyA(superlayer,2*i,ghost=True),self.ancilla).reshape(4,4)+(2**suv)*lam*self.UTurnOnWithHA["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i+1)].data.reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    #U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    #self.setUA(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=True)
                    self.setUA(np.dot(U,V).reshape(2,2,2,2),superlayer,2*i,ghost=True)

                    W=-np.kron(self.DenergyA(superlayer,2*i,ghost=False).conjugate(),self.ancilla).reshape(4,4)+(2**suv)*lam*self.UTurnOnWithHA["ghosts"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i+1)].data.reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    #U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    #self.setUA(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=False)
                    self.setUA(np.dot(U,V).reshape(2,2,2,2),superlayer,2*i,ghost=False)

                    UU=self.UTurnOnWithHA["s"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i+1)].data.reshape(4,4)
                    VV=self.UTurnOnWithHA["ghosts"+str(superlayer)+"_"+str(2*i)+"_"+str(2*i+1)].data.reshape(4,4).conjugate().T
                    if np.abs(4-np.trace(np.dot(UU,VV)))<tol:
                        actualuvsweeps-=(uvsweeps-suv-1)
                        break
                    #print(np.abs(4-np.trace(np.dot(UU,VV))))
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                U_upd=np.dot(UU,U_n)
                self.setUA(U_upd.reshape(2,2,2,2),superlayer,2*i,True)
                self.setUA(U_upd.reshape(2,2,2,2),superlayer,2*i,False)
                #self.setUA(UU.reshape(2,2,2,2),superlayer,2*i,True)
                #print("")
            #print("A:")
            E=self.calcenergy(superlayer)
            if not suppress:
                print(E)
            Es.append(E)
            noise*=decay
            #print(self.UVerror(superlayer))
        print("Actual sweeps: " + str(uvsweeps*(1.*actualuvsweeps)/maxuvsweeps))
        for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
            self.fixUD(superlayer,4*i)
        for i in range(1,(2**(superlayer-1))*self.params["L"]):
            self.fixUC(superlayer,2*i)
        for i in range(((2**(superlayer-1))*self.params["L"])/2):
            self.fixUB(superlayer,4*i)
        for i in range((2**(superlayer-1))*self.params["L"]):
            self.fixUA(superlayer,2*i)
        self.reg=False
        return Es;

    def optimizeUGradTI(self,superlayer,sweeps,noise=0.0,decay=0.0,suppress=False,rand=False):
        if str(superlayer) not in self.init:
            if superlayer==1:
                #self.setcoeffsrandom(.001,superlayer)
                if rand:
                    self.setcoeffsrandom(noise,superlayer)
                else:
                    self.setcoeffsanalytic(superlayer)
            else:
                self.setcoeffsprevlayer(superlayer)
        #print(self.calcenergy(superlayer))
        Es=[]
        for s in range(sweeps):
            #print("sweep "+str(s)+":")
            W=self.DenergyD(superlayer,((2**(superlayer-1))*self.params["L"]),ghost=False).reshape(4,4).conjugate()
            U,S,V=np.linalg.svd(W)
            for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUD(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=False)
            #print("D:")
            #print(self.calcenergy(superlayer))

            W=self.DenergyC(superlayer,((2**(superlayer-1))*self.params["L"]),ghost=False).reshape(4,4).conjugate()
            U,S,V=np.linalg.svd(W)
            for i in range(1,(2**(superlayer-1))*self.params["L"]):
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUC(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=False)
            #print("C:")
            #print(self.calcenergy(superlayer))
            W=self.DenergyB(superlayer,((2**(superlayer-1))*self.params["L"]),ghost=False).reshape(4,4).conjugate()
            U,S,V=np.linalg.svd(W)
            for i in range(((2**(superlayer-1))*self.params["L"])/2):
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUB(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i,ghost=False)
            #print("B:")
            #print(self.calcenergy(superlayer))
            W=np.kron(self.DenergyA(superlayer,((2**(superlayer-1))*self.params["L"]),ghost=False),self.ancilla).reshape(4,4).conjugate()
            U,S,V=np.linalg.svd(W)
            for i in range((2**(superlayer-1))*self.params["L"]):
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUA(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i,ghost=False)
            #print("A:")
            E=self.calcenergy(superlayer)
            if not suppress:
                print(E)
            Es.append(E)
            noise*=decay
        for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
            self.fixUD(superlayer,4*i)
        for i in range(1,(2**(superlayer-1))*self.params["L"]):
            self.fixUC(superlayer,2*i)
        for i in range(((2**(superlayer-1))*self.params["L"])/2):
            self.fixUB(superlayer,4*i)
        for i in range((2**(superlayer-1))*self.params["L"]):
            self.fixUA(superlayer,2*i)
        return Es;

    def optimizeUBH(self,superlayer,hops,hsweeps,fsweeps,noise=0.0,decay=0.0,suppress=False,rand=False,derrank=False,thresh=1e-6):
        if str(superlayer) not in self.init:
            if superlayer==1:
                #self.setcoeffsrandom(.001,superlayer)
                #self.setcoeffs(guess,superlayer)
                if rand:
                    self.setcoeffsrandom(noise,superlayer)
                else:
                    self.setcoeffsanalytic(superlayer)
            else:
                self.setcoeffsprevlayer(superlayer)
        snoise=noise
        Es=[]
        for hp in range(hops):
            noise=snoise
            for s in range(hsweeps):
                for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
                    W=self.DenergyD(superlayer,4*i).reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    self.setUD(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i)
                for i in range(1,(2**(superlayer-1))*self.params["L"]):
                    W=self.DenergyC(superlayer,2*i).reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    self.setUC(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i)
                for i in range(((2**(superlayer-1))*self.params["L"])/2):
                    W=self.DenergyB(superlayer,4*i).reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    self.setUB(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i)
                for i in range((2**(superlayer-1))*self.params["L"]):
                    W=np.kron(self.DenergyA(superlayer,2*i),self.ancilla).reshape(4,4)
                    U,S,V=np.linalg.svd(W)
                    U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                    self.setUA(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i)
                E=self.calcenergy(superlayer)
                if not suppress:
                    print(E)
                Es.append(E)
                if derrank:
                    if s>0:
                        if E<prevE and np.abs(1-prevE/E)<thresh:
                            remsweeps=hsweeps-s-1
                            break
                    prevE=E
                    if s+1==hsweeps:
                        remsweeps=0
                noise*=decay
            if hp==0:
                bestE=E
                tempcsA=copy.deepcopy(self.coeffsOnA)
                tempUsA=copy.deepcopy(self.UTurnOnWithHA)
                tempIsoA=copy.deepcopy(self.IsoTurnOnWithH)
                tempcsB=copy.deepcopy(self.coeffsOffB)
                tempUsB=copy.deepcopy(self.UTurnOffB)
                tempcsC=copy.deepcopy(self.coeffsOnC)
                tempUsC=copy.deepcopy(self.UTurnOnWithHC)
                tempcsD=copy.deepcopy(self.coeffsOffD)
                tempUsD=copy.deepcopy(self.UTurnOffD)
            else:
                if E<bestE:
                    bestE=E
                    tempcsA=copy.deepcopy(self.coeffsOnA)
                    tempUsA=copy.deepcopy(self.UTurnOnWithHA)
                    tempIsoA=copy.deepcopy(self.IsoTurnOnWithH)
                    tempcsB=copy.deepcopy(self.coeffsOffB)
                    tempUsB=copy.deepcopy(self.UTurnOffB)
                    tempcsC=copy.deepcopy(self.coeffsOnC)
                    tempUsC=copy.deepcopy(self.UTurnOnWithHC)
                    tempcsD=copy.deepcopy(self.coeffsOffD)
                    tempUsD=copy.deepcopy(self.UTurnOffD)
        self.coeffsOnA=copy.deepcopy(tempcsA)
        self.UTurnOnWithHA=copy.deepcopy(tempUsA)
        self.IsoTurnOnWithH=copy.deepcopy(tempIsoA)
        self.coeffsOffB=copy.deepcopy(tempcsB)
        self.UTurnOffB=copy.deepcopy(tempUsB)
        self.coeffsOnC=copy.deepcopy(tempcsC)
        self.UTurnOnWithHC=copy.deepcopy(tempUsC)
        self.coeffsOffD=copy.deepcopy(tempcsD)
        self.UTurnOffD=copy.deepcopy(tempUsD)

        if not derrank:
            remsweeps=0
        for s in range(fsweeps+remsweeps):
            for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
                W=self.DenergyD(superlayer,4*i).reshape(4,4)
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUD(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i)
            for i in range(1,(2**(superlayer-1))*self.params["L"]):
                W=self.DenergyC(superlayer,2*i).reshape(4,4)
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUC(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i)
            for i in range(((2**(superlayer-1))*self.params["L"])/2):
                W=self.DenergyB(superlayer,4*i).reshape(4,4)
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUB(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,4*i)
            for i in range((2**(superlayer-1))*self.params["L"]):
                W=np.kron(self.DenergyA(superlayer,2*i),self.ancilla).reshape(4,4)
                U,S,V=np.linalg.svd(W)
                U_n=self.Uparam(np.random.normal(0,noise,15)).reshape(4,4)
                self.setUA(np.dot(np.dot(U,V),U_n).reshape(2,2,2,2),superlayer,2*i)
            E=self.calcenergy(superlayer)
            if not suppress:
                print(E)
            Es.append(E)
            noise*=decay
        for i in range(1,((2**(superlayer-1))*self.params["L"])/2):
            self.fixUD(superlayer,4*i)
        for i in range(1,(2**(superlayer-1))*self.params["L"]):
            self.fixUC(superlayer,2*i)
        for i in range(((2**(superlayer-1))*self.params["L"])/2):
            self.fixUB(superlayer,4*i)
        for i in range((2**(superlayer-1))*self.params["L"]):
            self.fixUA(superlayer,2*i)
        return Es;

    def infidelityLto2L(self,coeffs,full=True):
        PBC=self.params["PBC"]
        coeffsA=coeffs[0:15]
        coeffsB=coeffs[15:30]
        if full:
            coeffsC=coeffs[30:45]
            coeffsD=coeffs[45:60]
        else:
            coeffsC=coeffs[0:15]
            coeffsD=coeffs[15:30]
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        self.UTurnOnWithHA = ten.Tensor(self.Uparam(coeffsA),bonds)
        self.UTurnOnWithHC = ten.Tensor(self.Uparam(coeffsC),bonds)
        self.UTurnOffB= ten.Tensor(self.Uparam(coeffsB),bonds)
        self.UTurnOffD= ten.Tensor(self.Uparam(coeffsD),bonds)
        bondspin=[ten.Bond("spin_in_right",2)]
        upspin_right=ten.Tensor(self.ancilla,bondspin)
        self.IsoTurnOnWithH=ten.contract(self.UTurnOnWithHA,upspin_right)

        return 1-self.fidelity(self.optL,1,PBC,True,self.MPSL,True,self.MPS2L)

    def localErrorLto2L(self,coeffs,full=True):
        L0=self.optL
        PBC=self.params["PBC"]
        coeffsA=coeffs[0:15]
        coeffsB=coeffs[15:30]
        if full:
            coeffsC=coeffs[30:45]
            coeffsD=coeffs[45:60]
        else:
            coeffsC=coeffs[0:15]
            coeffsD=coeffs[15:30]
        bonds=[ten.Bond("spin_out_left",2),ten.Bond("spin_out_right",2),ten.Bond("spin_in_left",2),ten.Bond("spin_in_right",2)]
        self.UTurnOnWithHA = ten.Tensor(self.Uparam(coeffsA),bonds)
        self.UTurnOnWithHC = ten.Tensor(self.Uparam(coeffsC),bonds)
        self.UTurnOffB= ten.Tensor(self.Uparam(coeffsB),bonds)
        self.UTurnOffD= ten.Tensor(self.Uparam(coeffsD),bonds)
        bondspin=[ten.Bond("spin_in_right",2)]
        upspin_right=ten.Tensor(self.ancilla,bondspin)
        self.IsoTurnOnWithH=ten.contract(self.UTurnOnWithHA,upspin_right)

        delta2=0.
        for j in range(2*L0):
            delta2+=(self.RenormEV(XIII,j,4,L0,1,self.MPSL)-self.Xtar)**2
            delta2+=(self.RenormEV(YIII,j,4,L0,1,self.MPSL)-self.Ytar)**2
            delta2+=(self.RenormEV(ZIII,j,4,L0,1,self.MPSL)-self.Ztar)**2
        for j in range(2*L0):
            delta2+=(self.RenormEV(XXII,j,4,L0,1,self.MPSL)-self.XXtar)**2
            delta2+=(self.RenormEV(YYII,j,4,L0,1,self.MPSL)-self.YYtar)**2
            delta2+=(self.RenormEV(ZZII,j,4,L0,1,self.MPSL)-self.ZZtar)**2
            delta2+=(self.RenormEV(XYII,j,4,L0,1,self.MPSL)-self.XYtar)**2
            delta2+=(self.RenormEV(YZII,j,4,L0,1,self.MPSL)-self.YZtar)**2
            delta2+=(self.RenormEV(ZXII,j,4,L0,1,self.MPSL)-self.ZXtar)**2
            delta2+=(self.RenormEV(XZII,j,4,L0,1,self.MPSL)-self.XZtar)**2
            delta2+=(self.RenormEV(YXII,j,4,L0,1,self.MPSL)-self.YXtar)**2
            delta2+=(self.RenormEV(ZYII,j,4,L0,1,self.MPSL)-self.ZYtar)**2

        return delta2

    def UVerror(self,layer):
        count=0
        err=0
        for i in range((2**(layer-1))*self.params["L"]):
            count+=1
            Vdag=copy.deepcopy(self.UTurnOnWithHA["ghosts"+str(layer)+"_"+str(2*i)+"_"+str(2*i+1)])
            Vdag.data=Vdag.data.conjugate()
            err+=4-np.complex_(ten.contract(Vdag,self.UTurnOnWithHA["s"+str(layer)+"_"+str(2*i)+"_"+str(2*i+1)]).data)
        for i in range(((2**(layer-1))*self.params["L"])/2):
            count+=1
            Vdag=copy.deepcopy(self.UTurnOffB["ghosts"+str(layer)+"_"+str(4*i)+"_"+str(4*i+2)])
            Vdag.data=Vdag.data.conjugate()
            err+=4-np.complex_(ten.contract(Vdag,self.UTurnOffB["s"+str(layer)+"_"+str(4*i)+"_"+str(4*i+2)]).data)
        for i in range(1,(2**(layer-1))*self.params["L"]):
            count+=1
            Vdag=copy.deepcopy(self.UTurnOnWithHC["ghosts"+str(layer)+"_"+str(2*i)+"_"+str(2*i-1)])
            Vdag.data=Vdag.data.conjugate()
            err+=4-np.complex_(ten.contract(Vdag,self.UTurnOnWithHC["s"+str(layer)+"_"+str(2*i)+"_"+str(2*i-1)]).data)
        for i in range(1,((2**(layer-1))*self.params["L"])/2):
            count+=1
            Vdag=copy.deepcopy(self.UTurnOffD["ghosts"+str(layer)+"_"+str(4*i)+"_"+str(4*i-2)])
            Vdag.data=Vdag.data.conjugate()
            err+=4-np.complex_(ten.contract(Vdag,self.UTurnOffD["s"+str(layer)+"_"+str(4*i)+"_"+str(4*i-2)]).data)
        return err/count

    def energy(self,coeffs,superlayer):#modifies unitaries and then calculates the energy
        self.setcoeffs(coeffs,superlayer)
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            OpA=-h*(XIIIII+IXIIII+IIXIII+IIIXII)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI)
            OpB=-h*(XIII+IXII+IIXI+IIIX)-J*(ZZII+IZZI+IIZZ)
            en=0
            for j in range(L*(2**superlayer)/4):
                if (not self.params["PBC"]) and j+1==L*(2**superlayer)/4:
                    en+=self.RenormEV(OpB,4*j,4,L,superlayer,self.MPS)
                else:
                    en+=self.RenormEV(OpA,4*j,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIXII)-hz*(ZIIIII+IZIIII+IIZIII+IIIZII)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI)
            OpB=-hx*(XIII+IXII+IIXI+IIIX)-hz*(ZIII+IZII+IIZI+IIIZ)-Jx*(XXII+IXXI+IIXX)-Jz*(ZZII+IZZI+IIZZ)
            en=0
            for j in range(L*(2**superlayer)/4):
                if (not self.params["PBC"]) and j+1==L*(2**superlayer)/4:
                    en+=self.RenormEV(OpB,4*j,4,L,superlayer,self.MPS)
                else:
                    en+=self.RenormEV(OpA,4*j,6,L,superlayer,self.MPS)
        return en;

    def calcenergy(self,superlayer):#Just calculate the energy without changing any unitaries
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            OpA=-h*(XIIIII+IXIIII+IIXIII+IIIXII)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI)
            OpB=-h*(XIII+IXII+IIXI+IIIX)-J*(ZZII+IZZI+IIZZ)
            en=0
            for j in range(L*(2**superlayer)/4):
                if (not self.params["PBC"]) and j+1==L*(2**superlayer)/4:
                    en+=self.RenormEV(OpB,4*j,4,L,superlayer,self.MPS)
                else:
                    en+=self.RenormEV(OpA,4*j,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIXII)-hz*(ZIIIII+IZIIII+IIZIII+IIIZII)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI)
            OpB=-hx*(XIII+IXII+IIXI+IIIX)-hz*(ZIII+IZII+IIZI+IIIZ)-Jx*(XXII+IXXI+IIXX)-Jz*(ZZII+IZZI+IIZZ)
            en=0
            for j in range(L*(2**superlayer)/4):
                if (not self.params["PBC"]) and j+1==L*(2**superlayer)/4:
                    en+=self.RenormEV(OpB,4*j,4,L,superlayer,self.MPS)
                else:
                    en+=self.RenormEV(OpA,4*j,6,L,superlayer,self.MPS)
        return en;

    def calcExpO(self,superlayer,O):#calculate the expectation value of operator O averaged over the sites for specific O
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if O=="X":
            OpA=XIIIII+IXIIII+IIXIII+IIIXII
            OpB=XIII+IXII+IIXI+IIIX
            NN=(2**superlayer)*L
        elif O=="Y":
            OpA=YIIIII+IYIIII+IIYIII+IIIYII
            OpB=YIII+IYII+IIYI+IIIY
            NN=(2**superlayer)*L
        elif O=="Z":
            OpA=ZIIIII+IZIIII+IIZIII+IIIZII
            OpB=ZIII+IZII+IIZI+IIIZ
            NN=(2**superlayer)*L
        elif O=="XX":
            OpA=XXIIII+IXXIII+IIXXII+IIIXXI
            OpB=XXII+IXXI+IIXX
            NN=(2**superlayer)*L-1
        elif O=="YY":
            OpA=YYIIII+IYYIII+IIYYII+IIIYYI
            OpB=YYII+IYYI+IIYY
            NN=(2**superlayer)*L-1
        elif O=="ZZ":
            OpA=ZZIIII+IZZIII+IIZZII+IIIZZI
            OpB=ZZII+IZZI+IIZZ
            NN=(2**superlayer)*L-1
        else:
            return None
        OO=0
        for j in range(L*(2**superlayer)/4):
            if (not self.params["PBC"]) and j+1==L*(2**superlayer)/4:
                OO+=self.RenormEV(OpB,4*j,4,L,superlayer,self.MPS)
            else:
                OO+=self.RenormEV(OpA,4*j,6,L,superlayer,self.MPS)
        return OO/NN;

    def energyD(self,superlayer,leg1):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            OpA=-h*(IIXIII+IIIIXI)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
            en=self.RenormEV(OpA,leg1-4,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            OpA=-hx*(IIXIII+IIIIXI)-hz*(IIZIII+IIIIZI)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
            en=self.RenormEV(OpA,leg1-4,6,L,superlayer,self.MPS)
        return en;

    def energyC(self,superlayer,leg1):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            if leg1+2==L*(2**superlayer):
                OpA=-h*(IXII+IIXI)-J*(ZZII+IZZI+IIZZ)
                en=self.RenormEV(OpA,leg1-2,4,L,superlayer,self.MPS)
            else:
                OpA=-h*(IXIIII+IIXIII+IIIXII+IIIIXI)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if leg1%4==0:
                    en=self.RenormEV(OpA,leg1-4,6,L,superlayer,self.MPS)
                elif leg1%4==2:
                    en=self.RenormEV(OpA,leg1-2,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            if leg1+2==L*(2**superlayer):
                OpA=-hx*(IXII+IIXI)-hz*(IZII+IIZI)-Jx*(XXII+IXXI+IIXX)-Jz*(ZZII+IZZI+IIZZ)
                en=self.RenormEV(OpA,leg1-2,4,L,superlayer,self.MPS)
            else:
                OpA=-hx*(IXIIII+IIXIII+IIIXII+IIIIXI)-hz*(IZIIII+IIZIII+IIIZII+IIIIZI)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if leg1%4==0:
                    en=self.RenormEV(OpA,leg1-4,6,L,superlayer,self.MPS)
                elif leg1%4==2:
                    en=self.RenormEV(OpA,leg1-2,6,L,superlayer,self.MPS)
        return en;

    def energyB(self,superlayer,leg1):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            if leg1==0:
                OpA=-h*(XIIIII+IXIIII+IIXIII+IIIIXI)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                en=self.RenormEV(OpA,leg1,6,L,superlayer,self.MPS)
            elif leg1+4==L*(2**superlayer):
                OpA=-h*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                OpB=-J*(IZZI)
                en=self.RenormEV(OpA,leg1-2,6,L,superlayer,self.MPS)
                en+=self.RenormEV(OpB,leg1-4,4,L,superlayer,self.MPS)
            else:
                OpA=-h*(IIXIII+IIIXII+IIIIXI+IIIIIX)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                OpB=-h*(IIXIII+IIIXII+IIIIXI)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                en=self.RenormEV(OpA,leg1-4,6,L,superlayer,self.MPS)
                en+=self.RenormEV(OpB,leg1,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            if leg1==0:
                OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIIXI)-hz*(ZIIIII+IZIIII+IIZIII+IIIIZI)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                en=self.RenormEV(OpA,leg1,6,L,superlayer,self.MPS)
            elif leg1+4==L*(2**superlayer):
                OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI)-hz*(ZIIIII+IZIIII+IIZIII+IIIZII+IIIIZI)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                OpB=-Jx*(IXXI)-Jz*(IZZI)
                en=self.RenormEV(OpA,leg1-2,6,L,superlayer,self.MPS)
                en+=self.RenormEV(OpB,leg1-4,4,L,superlayer,self.MPS)
            else:
                OpA=-hx*(IIXIII+IIIXII+IIIIXI+IIIIIX)-hz*(IIZIII+IIIZII+IIIIZI+IIIIIZ)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                OpB=-hx*(IIXIII+IIIXII+IIIIXI)-hz*(IIZIII+IIIZII+IIIIZI)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                en=self.RenormEV(OpA,leg1-4,6,L,superlayer,self.MPS)
                en+=self.RenormEV(OpB,leg1,6,L,superlayer,self.MPS)
        return en;

    def energyA(self,superlayer,leg1):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            if leg1==0 or leg1==2:
                OpA=-h*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                en=self.RenormEV(OpA,0,6,L,superlayer,self.MPS)
            elif leg1+4==L*(2**superlayer) or leg1+2==L*(2**superlayer):
                OpA=-h*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI+IIIIIX)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                OpB=-J*(IZZI)
                en=self.RenormEV(OpA,L*(2**superlayer)-6,6,L,superlayer,self.MPS)
                en+=self.RenormEV(OpB,L*(2**superlayer)-8,4,L,superlayer,self.MPS)
            else:
                OpA=-h*(IIXIII+IIIXII+IIIIXI+IIIIIX)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                OpB=-h*(IIXIII+IIIXII+IIIIXI)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if leg1%4==0:
                    en=self.RenormEV(OpA,leg1-4,6,L,superlayer,self.MPS)
                    en+=self.RenormEV(OpB,leg1,6,L,superlayer,self.MPS)
                elif leg1%4==2:
                    en=self.RenormEV(OpA,leg1-6,6,L,superlayer,self.MPS)
                    en+=self.RenormEV(OpB,leg1-2,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            if leg1==0 or leg1==2:
                OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI)-hz*(ZIIIII+IZIIII+IIZIII+IIIZII+IIIIZI)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                en=self.RenormEV(OpA,0,6,L,superlayer,self.MPS)
            elif leg1+4==L*(2**superlayer) or leg1+2==L*(2**superlayer):
                OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI+IIIIIX)-hz*(ZIIIII+IZIIII+IIZIII+IIIZII+IIIIZI+IIIIIZ)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                OpB=-Jx*(IXXI)-Jz*(IZZI)
                en=self.RenormEV(OpA,L*(2**superlayer)-6,6,L,superlayer,self.MPS)
                en+=self.RenormEV(OpB,L*(2**superlayer)-8,4,L,superlayer,self.MPS)
            else:
                OpA=-hx*(IIXIII+IIIXII+IIIIXI+IIIIIX)-hz*(IIZIII+IIIZII+IIIIZI+IIIIIZ)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                OpB=-hx*(IIXIII+IIIXII+IIIIXI)-hz*(IIZIII+IIIZII+IIIIZI)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if leg1%4==0:
                    en=self.RenormEV(OpA,leg1-4,6,L,superlayer,self.MPS)
                    en+=self.RenormEV(OpB,leg1,6,L,superlayer,self.MPS)
                elif leg1%4==2:
                    en=self.RenormEV(OpA,leg1-6,6,L,superlayer,self.MPS)
                    en+=self.RenormEV(OpB,leg1-2,6,L,superlayer,self.MPS)
        return en;

    def DenergyD(self,superlayer,leg1,ghost=False):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            OpA=-h*(IIXIII+IIIIXI)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
            if self.test:
                #OpA-=(2*np.abs(h)+4*np.abs(J))*IIIIII
                lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                OpA-=lam*IIIIII
            Den=self.DRenormEV(OpA,4,leg1,leg1-4,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            OpA=-hx*(IIXIII+IIIIXI)-hz*(IIZIII+IIIIZI)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
            if self.test:
                lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                OpA-=lam*IIIIII
            Den=self.DRenormEV(OpA,4,leg1,leg1-4,6,L,superlayer,self.MPS)
        return Den;

    def DenergyC(self,superlayer,leg1,ghost=False):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            if leg1+2==L*(2**superlayer):
                OpA=-h*(IXII+IIXI)-J*(ZZII+IZZI+IIZZ)
                if self.test:
                    #OpA-=(2*np.abs(h)+3*np.abs(J))*IIII
                    lam=LA.eigh(OpA.reshape(16,16),eigvals=(15,15),eigvals_only=True)
                    OpA-=lam*IIII
                Den=self.DRenormEV(OpA,3,leg1,leg1-2,4,L,superlayer,self.MPS)
            else:
                OpA=-h*(IXIIII+IIXIII+IIIXII+IIIIXI)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    #OpA-=(4*np.abs(h)+5*np.abs(J))*IIIIII
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                if leg1%4==0:
                    Den=self.DRenormEV(OpA,3,leg1,leg1-4,6,L,superlayer,self.MPS)
                elif leg1%4==2:
                    Den=self.DRenormEV(OpA,3,leg1,leg1-2,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            if leg1+2==L*(2**superlayer):
                OpA=-hx*(IXII+IIXI)-hz*(IZII+IIZI)-Jx*(XXII+IXXI+IIXX)-Jz*(ZZII+IZZI+IIZZ)
                if self.test:
                    lam=LA.eigh(OpA.reshape(16,16),eigvals=(15,15),eigvals_only=True)
                    OpA-=lam*IIII
                Den=self.DRenormEV(OpA,3,leg1,leg1-2,4,L,superlayer,self.MPS)
            else:
                OpA=-hx*(IXIIII+IIXIII+IIIXII+IIIIXI)-hz*(IZIIII+IIZIII+IIIZII+IIIIZI)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                if leg1%4==0:
                    Den=self.DRenormEV(OpA,3,leg1,leg1-4,6,L,superlayer,self.MPS)
                elif leg1%4==2:
                    Den=self.DRenormEV(OpA,3,leg1,leg1-2,6,L,superlayer,self.MPS)
        return Den;

    def DenergyB(self,superlayer,leg1,ghost=False):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            if leg1==0:
                OpA=-h*(XIIIII+IXIIII+IIXIII+IIIIXI)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    #OpA-=(4*np.abs(h)+5*np.abs(J))*IIIIII
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                Den=self.DRenormEV(OpA,2,leg1,leg1,6,L,superlayer,self.MPS)
            elif leg1+4==L*(2**superlayer):
                OpA=-h*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    #OpA-=(5*np.abs(h)+5*np.abs(J))*IIIIII
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                OpB=-J*(IZZI)
                if self.test:
                    #OpB-=(np.abs(J))*IIII
                    lam=LA.eigh(OpB.reshape(16,16),eigvals=(15,15),eigvals_only=True)
                    OpB-=lam*IIII
                Den=self.DRenormEV(OpA,2,leg1,leg1-2,6,L,superlayer,self.MPS)
                Den+=self.DRenormEV(OpB,2,leg1,leg1-4,4,L,superlayer,self.MPS)
            else:
                OpA=-h*(IIXIII+IIIXII+IIIIXI+IIIIIX)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    #OpA-=(4*np.abs(h)+4*np.abs(J))*IIIIII
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                OpB=-h*(IIXIII+IIIXII+IIIIXI)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    #OpB-=(3*np.abs(h)+4*np.abs(J))*IIIIII
                    lam=LA.eigh(OpB.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpB-=lam*IIIIII
                Den=self.DRenormEV(OpA,2,leg1,leg1-4,6,L,superlayer,self.MPS)
                Den+=self.DRenormEV(OpB,2,leg1,leg1,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            if leg1==0:
                OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIIXI)-hz*(ZIIIII+IZIIII+IIZIII+IIIIZI)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                Den=self.DRenormEV(OpA,2,leg1,leg1,6,L,superlayer,self.MPS)
            elif leg1+4==L*(2**superlayer):
                OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI)-hz*(ZIIIII+IZIIII+IIZIII+IIIZII+IIIIZI)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                OpB=-Jx*(IXXI)-Jz*(IZZI)
                if self.test:
                    lam=LA.eigh(OpB.reshape(16,16),eigvals=(15,15),eigvals_only=True)
                    OpB-=lam*IIII
                Den=self.DRenormEV(OpA,2,leg1,leg1-2,6,L,superlayer,self.MPS)
                Den+=self.DRenormEV(OpB,2,leg1,leg1-4,4,L,superlayer,self.MPS)
            else:
                OpA=-hx*(IIXIII+IIIXII+IIIIXI+IIIIIX)-hz*(IIZIII+IIIZII+IIIIZI+IIIIIZ)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                OpB=-hx*(IIXIII+IIIXII+IIIIXI)-hz*(IIZIII+IIIZII+IIIIZI)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    lam=LA.eigh(OpB.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpB-=lam*IIIIII
                Den=self.DRenormEV(OpA,2,leg1,leg1-4,6,L,superlayer,self.MPS)
                Den+=self.DRenormEV(OpB,2,leg1,leg1,6,L,superlayer,self.MPS)
        return Den;

    def DenergyA(self,superlayer,leg1,ghost=False):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=self.params["J"]
            h=self.params["h"]
            if leg1==0 or leg1==2:
                OpA=-h*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    #OpA-=(5*np.abs(h)+5*np.abs(J))*IIIIII
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                Den=self.DRenormEV(OpA,1,leg1,0,6,L,superlayer,self.MPS)
            elif leg1+4==L*(2**superlayer) or leg1+2==L*(2**superlayer):
                OpA=-h*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI+IIIIIX)-J*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    #OpA-=(6*np.abs(h)+5*np.abs(J))*IIIIII
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                OpB=-J*(IZZI)
                if self.test:
                    #OpB-=(np.abs(J))*IIII
                    lam=LA.eigh(OpB.reshape(16,16),eigvals=(15,15),eigvals_only=True)
                    OpB-=lam*IIII
                Den=self.DRenormEV(OpA,1,leg1,L*(2**superlayer)-6,6,L,superlayer,self.MPS)
                Den+=self.DRenormEV(OpB,1,leg1,L*(2**superlayer)-8,4,L,superlayer,self.MPS)
            else:
                OpA=-h*(IIXIII+IIIXII+IIIIXI+IIIIIX)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    #OpA-=(4*np.abs(h)+4*np.abs(J))*IIIIII
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                OpB=-h*(IIXIII+IIIXII+IIIIXI)-J*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    #OpB-=(3*np.abs(h)+4*np.abs(J))*IIIIII
                    lam=LA.eigh(OpB.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpB-=lam*IIIIII
                if leg1%4==0:
                    Den=self.DRenormEV(OpA,1,leg1,leg1-4,6,L,superlayer,self.MPS)
                    Den+=self.DRenormEV(OpB,1,leg1,leg1,6,L,superlayer,self.MPS)
                elif leg1%4==2:
                    Den=self.DRenormEV(OpA,1,leg1,leg1-6,6,L,superlayer,self.MPS)
                    Den+=self.DRenormEV(OpB,1,leg1,leg1-2,6,L,superlayer,self.MPS)
        elif self.modelname=="MFieldAndCoupling":
            Jx=self.params["Jx"]
            Jz=self.params["Jz"]
            hx=self.params["hx"]
            hz=self.params["hz"]
            if leg1==0 or leg1==2:
                OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI)-hz*(ZIIIII+IZIIII+IIZIII+IIIZII+IIIIZI)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                Den=self.DRenormEV(OpA,1,leg1,0,6,L,superlayer,self.MPS)
            elif leg1+4==L*(2**superlayer) or leg1+2==L*(2**superlayer):
                OpA=-hx*(XIIIII+IXIIII+IIXIII+IIIXII+IIIIXI+IIIIIX)-hz*(ZIIIII+IZIIII+IIZIII+IIIZII+IIIIZI+IIIIIZ)-Jx*(XXIIII+IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(ZZIIII+IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                OpB=-Jx*(IXXI)-Jz*(IZZI)
                if self.test:
                    lam=LA.eigh(OpB.reshape(16,16),eigvals=(15,15),eigvals_only=True)
                    OpB-=lam*IIII
                Den=self.DRenormEV(OpA,1,leg1,L*(2**superlayer)-6,6,L,superlayer,self.MPS)
                Den+=self.DRenormEV(OpB,1,leg1,L*(2**superlayer)-8,4,L,superlayer,self.MPS)
            else:
                OpA=-hx*(IIXIII+IIIXII+IIIIXI+IIIIIX)-hz*(IIZIII+IIIZII+IIIIZI+IIIIIZ)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    lam=LA.eigh(OpA.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpA-=lam*IIIIII
                OpB=-hx*(IIXIII+IIIXII+IIIIXI)-hz*(IIZIII+IIIZII+IIIIZI)-Jx*(IXXIII+IIXXII+IIIXXI+IIIIXX)-Jz*(IZZIII+IIZZII+IIIZZI+IIIIZZ)
                if self.test:
                    lam=LA.eigh(OpB.reshape(64,64),eigvals=(63,63),eigvals_only=True)
                    OpB-=lam*IIIIII
                if leg1%4==0:
                    Den=self.DRenormEV(OpA,1,leg1,leg1-4,6,L,superlayer,self.MPS)
                    Den+=self.DRenormEV(OpB,1,leg1,leg1,6,L,superlayer,self.MPS)
                elif leg1%4==2:
                    Den=self.DRenormEV(OpA,1,leg1,leg1-6,6,L,superlayer,self.MPS)
                    Den+=self.DRenormEV(OpB,1,leg1,leg1-2,6,L,superlayer,self.MPS)
        return Den;


    def tolD(self,superlayer,leg1,unittol):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=np.abs(self.params["J"])
            h=np.abs(self.params["h"])
            q=2*h+4*J
        elif self.modelname=="MFieldAndCoupling":
            Jx=np.abs(self.params["Jx"])
            Jz=np.abs(self.params["Jz"])
            hx=np.abs(self.params["hx"])
            hz=np.abs(self.params["hz"])
            q=2*(hx+hz)+4*(Jx+Jz)
        return q*unittol;

    def tolC(self,superlayer,leg1,unittol):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=np.abs(self.params["J"])
            h=np.abs(self.params["h"])
            if leg1+2==L*(2**superlayer):
                q=2*h+3*J
            else:
                if leg1%4==0:
                    q=3*h+4*J
                elif leg1%4==2:
                    q=3*h+5*J
        elif self.modelname=="MFieldAndCoupling":
            Jx=np.abs(self.params["Jx"])
            Jz=np.abs(self.params["Jz"])
            hx=np.abs(self.params["hx"])
            hz=np.abs(self.params["hz"])
            if leg1+2==L*(2**superlayer):
                q=2*(hx+hz)+3*(Jx+Jz)
            else:
                if leg1%4==0:
                    q=3*(hx+hz)+4*(Jx+Jz)
                elif leg1%4==2:
                    q=3*(hx+hz)+5*(Jx+Jz)
        return q*unittol;

    def tolB(self,superlayer,leg1,unittol):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=np.abs(self.params["J"])
            h=np.abs(self.params["h"])
            if leg1==0:
                q=4*h+5*J
            elif leg1+4==L*(2**superlayer):
                q=5*h+6*J
            else:
                q=6*h+8*J
        elif self.modelname=="MFieldAndCoupling":
            Jx=np.abs(self.params["Jx"])
            Jz=np.abs(self.params["Jz"])
            hx=np.abs(self.params["hx"])
            hz=np.abs(self.params["hz"])
            if leg1==0:
                q=4*(hx+hz)+5*(Jx+Jz)
            elif leg1+4==L*(2**superlayer):
                q=5*(hx+hz)+6*(Jx+Jz)
            else:
                q=6*(hx+hz)+8*(Jx+Jz)
        return q*unittol;

    def tolA(self,superlayer,leg1,unittol):
        L=self.params["L"]
        if L%4!=0:#Blocking of H assumes this; I think other code would break anyway, but I'm putting this here to be safe.
            return None
        if self.modelname=="TFieldIsing":
            J=np.abs(self.params["J"])
            h=np.abs(self.params["h"])
            if leg1==0:
                q=4*h+5*J
            elif leg1==2:
                q=5*h+5*J
            elif leg1+4==L*(2**superlayer):
                q=5*h+6*J
            elif leg1+2==L*(2**superlayer):
                q=6*h+6*J
            else:
                if leg1%4==0:
                    q=6*h+8*J
                elif leg1%4==2:
                    q=7*h+8*J
        elif self.modelname=="MFieldAndCoupling":
            Jx=np.abs(self.params["Jx"])
            Jz=np.abs(self.params["Jz"])
            hx=np.abs(self.params["hx"])
            hz=np.abs(self.params["hz"])
            if leg1==0:
                q=4*(hx+hz)+5*(Jx+Jz)
            elif leg1==2:
                q=5*(hx+hz)+5*(Jx+Jz)
            elif leg1+4==L*(2**superlayer):
                q=5*(hx+hz)+6*(Jx+Jz)
            elif leg1+2==L*(2**superlayer):
                q=6*(hx+hz)+6*(Jx+Jz)
            else:
                if leg1%4==0:
                    q=6*(hx+hz)+8*(Jx+Jz)
                elif leg1%4==2:
                    q=7*(hx+hz)+8*(Jx+Jz)
        return q*unittol;

    def fidelity(self,L,pow,PBC=True,haveinitial=False, initial=None, havetarget=False,target=None):
        #PBC=self.params["PBC"]

        if haveinitial:
            MPSinit=copy.deepcopy(initial)
        else:
            paramsMPSinit={"L":L,"D":self.params["D"],"d":2,"DMPO":3,"h":self.params["h"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
            mpssiminit=mini.minimizeMPO(paramsMPSinit,PBC)
            mpssiminit.initializeMinimization(PBC)
            einit=mpssiminit.sweep(PBC)
            MPSinit=mpssiminit.mps
        if havetarget:
            MPStar=copy.deepcopy(target)
        else:
            paramsMPStar={"L":(L*(2**pow)),"D":self.params["D"],"d":2,"DMPO":3,"h":self.params["h"],"J":self.params["J"],"noise":self.noise,"sweeps":self.params["sweeps"]}
            mpssimtar=mini.minimizeMPO(paramsMPStar,PBC)
            mpssimtar.initializeMinimization(PBC)
            etar=mpssimtar.sweep(PBC)
            MPStar=mpssimtar.mps
        for i in range(L*(2**pow)):
            MPStar.MPS[i].changeBondName("spin_f"+str(i),"spin"+str(i))
        self.MPSCircuitA=[]
        self.MPSCircuitB=[]
        self.MPSCircuitC=[]
        self.MPSCircuitD=[]
        if PBC:
            for p in range(pow):
                m=(L*(2**(p+1)))
                s=2**(pow-p-1)
                layer=[]
                for i in range(L*(2**p)):
                    OnA=self.buildTurnOnIso(4*p+1,2*i*s,2*i*s,(2*i+1)*s)
                    if p==0:
                        OnA.changeBondName("spin"+str(i),"l0_"+str(2*i*s))
                    elif i%2==1:
                        OnA.changeBondName("l"+str(4*p-1)+"_"+str(2*i*s),"l"+str(4*p)+"_"+str(2*i*s))
                    layer.append(OnA)
                self.MPSCircuitA.append(layer)
                layer=[]
                for i in range(L*(2**p)/2):
                    OffB=self.buildTurnOff(4*p+2,4*i*s,(4*i+2)*s,4*i*s,(4*i+2)*s)
                    layer.append(OffB)
                self.MPSCircuitB.append(layer)
                layer=[]
                for i in range(L*(2**p)):
                    OnC=self.buildTurnOn(4*p+3,2*i*s,((2*i-1)%m)*s,2*i*s,((2*i-1)%m)*s)
                    OnC.changeBondName("l"+str(4*p+1)+"_"+str(((2*i-1)%m)*s),"l"+str(4*p+2)+"_"+str(((2*i-1)%m)*s))
                    if p==pow-1:
                        OnC.changeBondName("spin_f"+str(((2*i-1)%m)*s),"l"+str(4*p+3)+"_"+str(((2*i-1)%m)*s))
                    layer.append(OnC)
                self.MPSCircuitC.append(layer)
                layer=[]
                for i in range(L*(2**p)/2):
                    OnD=self.buildTurnOff(4*p+4,4*i*s,((4*i-2)%m)*s,4*i*s,((4*i-2)%m)*s)
                    if p==pow-1:
                        OnD.changeBondName("spin_f"+str(4*i),"l"+str(4*p+4)+"_"+str(4*i))
                        OnD.changeBondName("spin_f"+str((4*i-2)%m),"l"+str(4*p+4)+"_"+str((4*i-2)%m))
                    layer.append(OnD)
                self.MPSCircuitD.append(layer)
        else:
            for p in range(pow):
                m=(L*(2**(p+1)))
                s=2**(pow-p-1)
                layer=[]
                for i in range(L*(2**p)):
                    OnA=self.buildTurnOnIso(4*p+1,2*i*s,2*i*s,(2*i+1)*s)
                    if p==0:
                        OnA.changeBondName("spin"+str(i),"l0_"+str(2*i*s))
                    elif i%2==1:
                        if i==(L*(2**p)-1):
                            OnA.changeBondName("l"+str(4*p-3)+"_"+str(2*i*s),"l"+str(4*p)+"_"+str(2*i*s))
                        else:
                            OnA.changeBondName("l"+str(4*p-1)+"_"+str(2*i*s),"l"+str(4*p)+"_"+str(2*i*s))
                    elif i==0:
                        OnA.changeBondName("l"+str(4*p-2)+"_"+str(0),"l"+str(4*p)+"_"+str(0))
                    elif i==(L*(2**p)-2):
                        OnA.changeBondName("l"+str(4*p-1)+"_"+str(2*i*s),"l"+str(4*p)+"_"+str(2*i*s))
                    if p==pow-1 and i==(L*(2**p)-1):
                        OnA.changeBondName("spin_f"+str(2*i+1),"l"+str(4*p+1)+"_"+str(2*i+1))
                    layer.append(OnA)
                self.MPSCircuitA.append(layer)
                layer=[]
                for i in range(L*(2**p)/2):
                    OffB=self.buildTurnOff(4*p+2,4*i*s,(4*i+2)*s,4*i*s,(4*i+2)*s)
                    layer.append(OffB)
                    if i==0 and p==pow-1:
                        OffB.changeBondName("spin_f0","l"+str(4*p+2)+"_0")
                self.MPSCircuitB.append(layer)
                layer=[]
                for i in range(L*(2**p)):
                    if i>0:
                        OnC=self.buildTurnOn(4*p+3,2*i*s,((2*i-1)%m)*s,2*i*s,((2*i-1)%m)*s)
                        OnC.changeBondName("l"+str(4*p+1)+"_"+str(((2*i-1)%m)*s),"l"+str(4*p+2)+"_"+str(((2*i-1)%m)*s))
                        if p==pow-1:
                            OnC.changeBondName("spin_f"+str(((2*i-1)%m)*s),"l"+str(4*p+3)+"_"+str(((2*i-1)%m)*s))
                            if i==(L*(2**p)-1):
                                OnC.changeBondName("spin_f"+str(2*i*s),"l"+str(4*p+3)+"_"+str(2*i*s))
                        layer.append(OnC)
                self.MPSCircuitC.append(layer)
                layer=[]
                for i in range(L*(2**p)/2):
                    if i>0:
                        OnD=self.buildTurnOff(4*p+4,4*i*s,((4*i-2)%m)*s,4*i*s,((4*i-2)%m)*s)
                        if p==pow-1:
                            OnD.changeBondName("spin_f"+str(4*i),"l"+str(4*p+4)+"_"+str(4*i))
                            OnD.changeBondName("spin_f"+str((4*i-2)%m),"l"+str(4*p+4)+"_"+str((4*i-2)%m))
                        layer.append(OnD)
                self.MPSCircuitD.append(layer)

        first=True
        flag=True
        for i in range(L*(2**(pow-1))):
            #print("dags, i="+str(i))
            bonds=[ten.Bond("spin_f"+str(2*i),2),ten.Bond("spin_f"+str(2*i),2,True)]
            unprimer=ten.Tensor(np.array([[1,0],[0,1]]),bonds)
            Adagn=ten.contract(MPStar.getMPSnDagger(2*i),unprimer)

            #print(i)
            #Adagn.printBonds()
            if(first):
                F=Adagn
                first=False
            else:
                F=ten.contract(F,Adagn)
            #print(F.data.shape)
            bonds=[ten.Bond("spin_f"+str(2*i+1),2),ten.Bond("spin_f"+str(2*i+1),2,True)]
            unprimer=ten.Tensor(np.array([[1,0],[0,1]]),bonds)
            Adagn=ten.contract(MPStar.getMPSnDagger(2*i+1),unprimer)
            #Adagn.printBonds()
            F=ten.contract(F,Adagn)
            #print(F.data.shape)
            for p in reversed(range(pow)):
                if (i%(2**(pow-p-1)))==0:
                    if PBC:
                        if((i%(2**(pow-p)))==0):
                            F=ten.contract(F,self.MPSCircuitD[p][i/(2**(pow-p))])
                            #print("D, p="+str(p))
                            #print(F.data.shape)
                        #print("C, p="+str(p))
                        F=ten.contract(F,self.MPSCircuitC[p][i/(2**(pow-p-1))])
                    elif i>0:
                        if((i%(2**(pow-p)))==0):
                            F=ten.contract(F,self.MPSCircuitD[p][i/(2**(pow-p))-1])
                            #print("D, p="+str(p))
                            #print(F.data.shape)
                        #print("C, p="+str(p))
                        F=ten.contract(F,self.MPSCircuitC[p][i/(2**(pow-p-1))-1])
                    #print(F.data.shape)
                    if((i%(2**(pow-p)))==0):
                        #print("B, p="+str(p))
                        F=ten.contract(F,self.MPSCircuitB[p][i/(2**(pow-p))])
                        #print(F.data.shape)
                    #print("A, p="+str(p))
                    F=ten.contract(F,self.MPSCircuitA[p][i/(2**(pow-p-1))])
                    #print(F.data.shape)
            #print("i="+str(i))
            if (i%(2**(pow-1)))==0:
                F=ten.contract(F,MPSinit.getMPSn(i/(2**(pow-1))))
                #print(F.data.shape)
        #F.printBonds()

        return abs(np.complex_(F.data))**2



    def buildTurnOn(self,layer,left_input,right_input,left_output,right_output,superlayer,prime=False):
        """
        Build turn on tensor from the "left" leg number.
        :param layer: left layer
        :param left_input: input left leg
        :param left_output: output left leg
        :param prime: is prime?
        :return:
            TO turn off tensor
        """
        lowerlayer=layer-1
        bonds=[ten.Bond("l"+str(layer)+"_"+str(left_output),2),
               ten.Bond("l"+str(layer)+"_"+str(right_output),2),
               ten.Bond("l"+str(lowerlayer)+"_"+str(left_input),2),
               ten.Bond("l"+str(lowerlayer)+"_"+str(right_input),2)]
        if prime:
            for b in bonds:
                b.prime=True
        key="s"+str(superlayer)+"_"+str(left_output)+"_"+str(right_output)
        if layer%4==1:
            if key in self.UTurnOnWithHA:
                if (not self.reg) or (self.reglayer!=superlayer):
                    TO=ten.Tensor(self.UTurnOnWithHA[key].data.conjugate() if prime else self.UTurnOnWithHA[key].data,bonds)
                else:
                    TO=ten.Tensor(self.UTurnOnWithHA["ghost"+key].data.conjugate() if prime else self.UTurnOnWithHA[key].data,bonds)
            else:
                TO=ten.Tensor(UId,bonds)
                #print("A: !!!! "+str(key))
        elif layer%4==3:
            if key in self.UTurnOnWithHC:
                if (not self.reg) or (self.reglayer!=superlayer):
                    TO=ten.Tensor(self.UTurnOnWithHC[key].data.conjugate() if prime else self.UTurnOnWithHC[key].data,bonds)
                else:
                    TO=ten.Tensor(self.UTurnOnWithHC["ghost"+key].data.conjugate() if prime else self.UTurnOnWithHC[key].data,bonds)
            else:
                TO=ten.Tensor(UId,bonds)
                #print("C: !!!! "+str(key))
        else:
            print("layer number mismatch; should be odd for ON unitaries")
        return TO
    def buildTurnOnIso(self,layer,left_input,left_output,right_output,superlayer,prime=False):
        """
        Build turn on tensor from the "left" leg number.
        :param layer: left layer
        :param left_input: input left leg
        :param left_output: output left leg
        :param prime: is prime?
        :return:
            TO turn off tensor
        """
        lowerlayer=layer-1
        bonds=[ten.Bond("l"+str(layer)+"_"+str(left_output),2),
               ten.Bond("l"+str(layer)+"_"+str(right_output),2),
               ten.Bond("l"+str(lowerlayer)+"_"+str(left_input),2)]
        if prime:
            for b in bonds:
                b.prime=True
        key="s"+str(superlayer)+"_"+str(left_output)+"_"+str(right_output)
        if key in self.IsoTurnOnWithH:
            if (not self.reg) or (self.reglayer!=superlayer):
                TO=ten.Tensor(self.IsoTurnOnWithH[key].data.conjugate() if prime else self.IsoTurnOnWithH[key].data,bonds)
            else:
                TO=ten.Tensor(self.IsoTurnOnWithH["ghost"+key].data.conjugate() if prime else self.IsoTurnOnWithH[key].data,bonds)
        else:
            TO=ten.Tensor(UId,bonds)
            #print("A_iso: !!!! "+str(key))
        return TO
    def buildTurnOff(self,layer,left_input,right_input,left_output,right_output,superlayer,prime=False):
        """
        Build turn turn tensor from the "left" leg number. Note that we assume the we skip one leg
        :param layer: left layer
        :param left_input: input left leg
        :param left_output: output left leg
        :param prime: is prime?
        :return:
            TO turn off tensor
        """
        lowerlayer=layer-1
        bonds=[ten.Bond("l"+str(layer)+"_"+str(left_output),2),
               ten.Bond("l"+str(layer)+"_"+str(right_output),2),
               ten.Bond("l"+str(lowerlayer)+"_"+str(left_input),2),
               ten.Bond("l"+str(lowerlayer)+"_"+str(right_input),2)]
        if prime:
            for b in bonds:
                b.prime=True
        key="s"+str(superlayer)+"_"+str(left_output)+"_"+str(right_output)
        if layer%4==2:
            if key in self.UTurnOffB:
                if (not self.reg) or (self.reglayer!=superlayer):
                    TO=ten.Tensor(self.UTurnOffB[key].data.conjugate() if prime else self.UTurnOffB[key].data,bonds)
                else:
                    TO=ten.Tensor(self.UTurnOffB["ghost"+key].data.conjugate() if prime else self.UTurnOffB[key].data,bonds)
            else:
                TO=ten.Tensor(UId,bonds)
                #print("B: !!!! "+str(key))
        elif layer%4==0:
            if key in self.UTurnOffD:
                if (not self.reg) or (self.reglayer!=superlayer):
                    TO=ten.Tensor(self.UTurnOffD[key].data.conjugate() if prime else self.UTurnOffD[key].data,bonds)
                else:
                    TO=ten.Tensor(self.UTurnOffD["ghost"+key].data.conjugate() if prime else self.UTurnOffD[key].data,bonds)
            else:
                TO=ten.Tensor(UId,bonds)
                #print("D: !!!! "+str(key))
        else:
            print("layer number mismatch; should be even for OFF unitaries")
        return TO

    def Uparam(self, coeffs):
        """
        Gives the 2x2x2x2 unitary corresponding to a given parameterization of an arbitrary unitary (up to a
        phase). coeffs should be a 15 element array of coefficients of the basis {sigma_i} tensor {sigma_j}
        - I_4. For the order see the definition of 'basis' above.
        """
        H=np.zeros((4,4),dtype='complex128')
        assert len(basis)==len(coeffs)
        for i in range(len(coeffs)):
            H+=coeffs[i]*basis[i]
        return LA.expm(-1j*H).reshape(2,2,2,2)



    def RGStep4site(self,O,left,L,superlayer,OisTensor=False):
        """
        :param O:
        Assumed to be a 2,2,2,2*2,2,2,2 local 4-site operator on sites left through (left+3)%L
        L is the length of the system (assumes periodic BCs). Must be a multiple of 4 and >=8.
        :return:
        O after single RG step
        """
        if left!=(left%L) or left+3>=L:
            print("Operator to renormalize is out of bounds! (size 4)")
            return None

        # O Tensor
        if not OisTensor:
            bondsO_layer_1=[ten.Bond("l4_"+str(left+(i%4)),2,False if i<4 else True) for i in range(8)]
            O_ten=ten.Tensor(O,bondsO_layer_1)
        else:
            O_ten=O

        if (left%4)==0:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            for i in range(2):
                c_ten=self.buildTurnOff(4,(left+4*i),(left+4*i-2),(left+4*i),(left+4*i-2),superlayer)
                if i==0:
                    c_ten.setBondPrime("l4_"+str((left-2)),True)
                if i==1:
                    c_ten.setBondPrime("l4_"+str((left+4)),True)
                if left==0 and i==0:
                    O_ten.changeBondName("l2_0","l4_0")
                if left+4>=L and i==1:
                    O_ten.changeBondName("l3_"+str(left+2),"l4_"+str(left+2))
                    O_ten.changeBondName("l1_"+str(left+3),"l4_"+str(left+3))
                if not ((left==0 and i==0) or (left+4>=L and i==1)):
                    turnOffTenA.append(c_ten)
                    turnOffTenA.append(self.buildTurnOff(4,(left+4*i),(left+4*i-2),(left+4*i),(left+4*i-2),superlayer,True))
                else:
                    turnOffTenA.append(None)
                    turnOffTenA.append(None)
            O_L1=O_ten
            O_L1.changeBondName("l3_"+str((left+1)),"l4_"+str((left+1)))
            O_L1.changeBondName("l3_"+str((left+3)),"l4_"+str((left+3)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(4):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i-2),(left+2*i-3),(left+2*i-2),(left+2*i-3),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-3)),True)
                if i==1:
                    c_ten.setBondPrime("l3_"+str((left-1)),True)
                if left+4>=L and i==3:
                    O_L1.changeBondName("l2_"+str(left+3),"l3_"+str(left+3))
                if not ((left==0 and i<2) or (left+4>=L and i==3)):
                    turnOnTenB.append(c_ten)
                    turnOnTenB.append(self.buildTurnOn(level,(left+2*i-2),(left+2*i-3),(left+2*i-2),(left+2*i-3),superlayer,True))
                else:
                    turnOnTenB.append(None)
                    turnOnTenB.append(None)
            O_L2=O_L1
            #for tenon in turnOnTen:
                #O_L2=ten.contract(O_L2,tenon)
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            #if L==8:
                #for i in range(2):
                    #level=2
                    #turnOffTen.append(self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,False))
                    #turnOffTen.append(self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,True))
            #else:
            for i in range(3):
                level=2
                c_ten=self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-4)),True)
                if i==2:
                    c_ten.setBondPrime("l2_"+str((left+6)),True)
                if not ((left==0 and i==0) or (left+4>=L and i==2)):
                    turnOffTenC.append(c_ten)
                    turnOffTenC.append(self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,True))
                else:
                    turnOffTenC.append(None)
                    turnOffTenC.append(None)
            O_L3=O_L2
            #for tenoff in turnOffTen:
                #O_L3=ten.contract(O_L3,tenoff)
            # upgrade to level 3
            for tenon in turnOnTenB:
                if tenon!=None:
                    tenon.changeBondName("l1_"+str((left-3)),"l2_"+str((left-3)))
                    tenon.changeBondName("l1_"+str((left-1)),"l2_"+str((left-1)))
                    tenon.changeBondName("l1_"+str((left+1)),"l2_"+str((left+1)))
                    tenon.changeBondName("l1_"+str((left+3)),"l2_"+str((left+3)))
                    tenon.changeBondName("l1_"+str((left+5)),"l2_"+str((left+5)))
            ### Layer 4
            turnOnIsoTenD=[]
            #if L==8:
                #for i in range(4):
                    #level=1
                    #turnOnIsoTen.append(self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,False))
                    #turnOnIsoTen.append(self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,True))
            #else:
            for i in range(6):
                level=1
                c_ten=self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==4:
                    c_ten.setBondPrime("l1_"+str((left+5)),True)
                if i==5:
                    c_ten.setBondPrime("l1_"+str((left+7)),True)
                if not ((left==0 and i<2) or (left+4>=L and i>3)):
                    turnOnIsoTenD.append(c_ten)
                    turnOnIsoTenD.append(self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,True))
                else:
                    turnOnIsoTenD.append(None)
                    turnOnIsoTenD.append(None)
            O_L4=O_L3
            #for tenonIso in turnOnIsoTen:
                #O_L4=ten.contract(O_L4,tenonIso)

            if left+4<L:
                O_L4=ten.contract(O_L4,turnOffTenA[2])
                O_L4=ten.contract(O_L4,turnOnTenB[6])
            O_L4=ten.contract(O_L4,turnOnTenB[4])
            if left>0:
                O_L4=ten.contract(O_L4,turnOffTenA[0])
                O_L4=ten.contract(O_L4,turnOnTenB[2])
            O_L4=ten.contract(O_L4,turnOffTenC[2])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[4])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[6])
            if left+4<L:
                O_L4=ten.contract(O_L4,turnOffTenA[3])
                O_L4=ten.contract(O_L4,turnOnTenB[7])
            O_L4=ten.contract(O_L4,turnOnTenB[5])
            if left>0:
                O_L4=ten.contract(O_L4,turnOffTenA[1])
                O_L4=ten.contract(O_L4,turnOnTenB[3])
            O_L4=ten.contract(O_L4,turnOffTenC[3])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[5])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[7])
            if left+4<L:
                O_L4=ten.contract(O_L4,turnOffTenC[4])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[8])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[10])
                O_L4=ten.contract(O_L4,turnOffTenC[5])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[9])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[11])
            if left>0:
                O_L4=ten.contract(O_L4,turnOnTenB[0])
                O_L4=ten.contract(O_L4,turnOffTenC[0])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[0])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[2])
                O_L4=ten.contract(O_L4,turnOnTenB[1])
                O_L4=ten.contract(O_L4,turnOffTenC[1])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[1])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[3])

            if not OisTensor:
                aa=np.max((left/2-2,0))
                bb=np.min((left/2+3,L/2-1))
                for i in range(aa,bb+1):
                    O_L4.changeBondName("spin"+str(i),"l0_"+str(i))
                O_L4.reorderBonds([ten.Bond("spin"+str(aa+i%(bb-aa+1)),2,False if i<(bb-aa+1) else True) for i in range(2*(bb-aa+1))])

            #if L==8:
                #for i in range(4):
                    #O_L4.changeBondName("spin"+str(i),"l0_"+str(i))
                #O_L4.reorderBonds([ten.Bond("spin"+str(i%4),2,False if i<4 else True) for i in range(8)])
            #else:
                #O_L4.changeBondName("spin"+str((left/2-2)%(L/2)),"l0_"+str((left/2-2)%(L/2)))
                #O_L4.changeBondName("spin"+str((left/2-1)%(L/2)),"l0_"+str((left/2-1)%(L/2)))
                #O_L4.changeBondName("spin"+str((left/2)%(L/2)),"l0_"+str((left/2)%(L/2)))
                #O_L4.changeBondName("spin"+str((left/2+1)%(L/2)),"l0_"+str((left/2+1)%(L/2)))
                #O_L4.changeBondName("spin"+str((left/2+2)%(L/2)),"l0_"+str((left/2+2)%(L/2)))
                #O_L4.changeBondName("spin"+str((left/2+3)%(L/2)),"l0_"+str((left/2+3)%(L/2)))
                #O_L4.reorderBonds([ten.Bond("spin"+str((left/2-2+i%6)%(L/2)),2,False if i<6 else True) for i in range(12)])

        elif (left%4)==1:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            turnOffTenA.append(self.buildTurnOff(4,(left+3),(left+1),(left+3),(left+1),superlayer))
            turnOffTenA.append(self.buildTurnOff(4,(left+3),(left+1),(left+3),(left+1),superlayer,True))
            O_L1=O_ten
            # upgrade to level 1
            O_L1.changeBondName("l3_"+str(left),"l4_"+str(left))
            O_L1.changeBondName("l3_"+str((left+2)),"l4_"+str((left+2)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(2):
                level=3
                turnOnTenB.append(self.buildTurnOn(level,(left+1+2*i),(left+2*i),(left+1+2*i),(left+2*i),superlayer,False))
                turnOnTenB.append(self.buildTurnOn(level,(left+1+2*i),(left+2*i),(left+1+2*i),(left+2*i),superlayer,True))
            O_L2=O_L1
            for tenon in turnOnTenB:
                tenon.changeBondName("l1_"+str((left)),"l2_"+str((left)))
                tenon.changeBondName("l1_"+str((left+2)),"l2_"+str((left+2)))
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            for i in range(2):
                level=2
                c_ten=self.buildTurnOff(level,(left-1+4*i),(left+1+4*i),(left-1+4*i),(left+1+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-1)),True)
                if i==1:
                    c_ten.setBondPrime("l2_"+str((left+5)),True)
                turnOffTenC.append(c_ten)
                turnOffTenC.append(self.buildTurnOff(level,(left-1+4*i),(left+1+4*i),(left-1+4*i),(left+1+4*i),superlayer,True))
            O_L3=O_L2
            ### Layer 4
            turnOnIsoTenD=[]
            for i in range(4):
                level=1
                c_ten=self.buildTurnOnIso(level,((left-1)/2+i),(left-1+2*i),(left+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==2:
                    c_ten.setBondPrime("l1_"+str((left+4)),True)
                if i==3:
                    c_ten.setBondPrime("l1_"+str((left+6)),True)
                turnOnIsoTenD.append(c_ten)
                turnOnIsoTenD.append(self.buildTurnOnIso(level,((left-1)/2+i),(left-1+2*i),(left+2*i),superlayer,True))
            O_L4=O_L3


            O_L4=ten.contract(O_L4,turnOffTenA[0])
            O_L4=ten.contract(O_L4,turnOnTenB[0])
            O_L4=ten.contract(O_L4,turnOnTenB[2])
            O_L4=ten.contract(O_L4,turnOffTenA[1])
            O_L4=ten.contract(O_L4,turnOnTenB[1])
            O_L4=ten.contract(O_L4,turnOnTenB[3])
            O_L4=ten.contract(O_L4,turnOffTenC[0])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[0])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[2])
            O_L4=ten.contract(O_L4,turnOffTenC[1])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[1])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[3])
            O_L4=ten.contract(O_L4,turnOffTenC[2])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[4])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[6])
            O_L4=ten.contract(O_L4,turnOffTenC[3])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[5])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[7])

            if not OisTensor:
                aa=np.max(((left-1)/2,0))
                bb=np.min(((left-1)/2+3,L/2-1))
                for i in range(aa,bb+1):
                    O_L4.changeBondName("spin"+str(i),"l0_"+str(i))
                O_L4.reorderBonds([ten.Bond("spin"+str(aa+i%(bb-aa+1)),2,False if i<(bb-aa+1) else True) for i in range(2*(bb-aa+1))])
            #O_L4.changeBondName("spin"+str(((left-1)/2)%(L/2)),"l0_"+str(((left-1)/2)%(L/2)))
            #O_L4.changeBondName("spin"+str(((left-1)/2+1)%(L/2)),"l0_"+str(((left-1)/2+1)%(L/2)))
            #O_L4.changeBondName("spin"+str(((left-1)/2+2)%(L/2)),"l0_"+str(((left-1)/2+2)%(L/2)))
            #O_L4.changeBondName("spin"+str(((left-1)/2+3)%(L/2)),"l0_"+str(((left-1)/2+3)%(L/2)))
            #O_L4.reorderBonds([ten.Bond("spin"+str(((left-1)/2+i%4)%(L/2)),2,False if i<4 else True) for i in range(8)])

        elif (left%4)==2:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            turnOffTenA.append(self.buildTurnOff(4,(left+2),left,(left+2),left,superlayer))
            turnOffTenA.append(self.buildTurnOff(4,(left+2),left,(left+2),left,superlayer,True))
            O_L1=O_ten
            # upgrade to level 1
            O_L1.changeBondName("l3_"+str((left+1)),"l4_"+str((left+1)))
            O_L1.changeBondName("l3_"+str((left+3)),"l4_"+str((left+3)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(3):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i),(left+2*i-1),(left+2*i),(left+2*i-1),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-1)),True)
                if i==2:
                    c_ten.setBondPrime("l3_"+str((left+4)),True)
                turnOnTenB.append(c_ten)
                turnOnTenB.append(self.buildTurnOn(level,(left+2*i),(left+2*i-1),(left+2*i),(left+2*i-1),superlayer,True))
            O_L2=O_L1
            for tenon in turnOnTenB:
                tenon.changeBondName("l1_"+str((left-1)),"l2_"+str((left-1)))
                tenon.changeBondName("l1_"+str((left+1)),"l2_"+str((left+1)))
                tenon.changeBondName("l1_"+str((left+3)),"l2_"+str((left+3)))
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            for i in range(2):
                level=2
                c_ten=self.buildTurnOff(level,(left-2+4*i),(left+4*i),(left-2+4*i),(left+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-2)),True)
                turnOffTenC.append(c_ten)
                turnOffTenC.append(self.buildTurnOff(level,(left-2+4*i),(left+4*i),(left-2+4*i),(left+4*i),superlayer,True))
            O_L3=O_L2
            ### Layer 4
            turnOnIsoTenD=[]
            for i in range(4):
                level=1
                c_ten=self.buildTurnOnIso(level,(left/2-1+i),(left-2+2*i),(left-1+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==3:
                    c_ten.setBondPrime("l1_"+str((left+5)),True)
                turnOnIsoTenD.append(c_ten)
                turnOnIsoTenD.append(self.buildTurnOnIso(level,(left/2-1+i),(left-2+2*i),(left-1+2*i),superlayer,True))
            O_L4=O_L3

            O_L4=ten.contract(O_L4,turnOffTenA[0])
            O_L4=ten.contract(O_L4,turnOnTenB[2])
            O_L4=ten.contract(O_L4,turnOffTenA[1])
            O_L4=ten.contract(O_L4,turnOnTenB[3])
            O_L4=ten.contract(O_L4,turnOnTenB[4])
            O_L4=ten.contract(O_L4,turnOffTenC[2])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[4])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[6])
            O_L4=ten.contract(O_L4,turnOnTenB[5])
            O_L4=ten.contract(O_L4,turnOffTenC[3])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[5])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[7])
            O_L4=ten.contract(O_L4,turnOnTenB[0])
            O_L4=ten.contract(O_L4,turnOffTenC[0])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[0])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[2])
            O_L4=ten.contract(O_L4,turnOnTenB[1])
            O_L4=ten.contract(O_L4,turnOffTenC[1])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[1])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[3])

            if not OisTensor:
                aa=np.max((left/2-1,0))
                bb=np.min((left/2+2,L/2-1))
                for i in range(aa,bb+1):
                    O_L4.changeBondName("spin"+str(i),"l0_"+str(i))
                O_L4.reorderBonds([ten.Bond("spin"+str(aa+i%(bb-aa+1)),2,False if i<(bb-aa+1) else True) for i in range(2*(bb-aa+1))])
            #O_L4.changeBondName("spin"+str((left/2-1)%(L/2)),"l0_"+str((left/2-1)%(L/2)))
            #O_L4.changeBondName("spin"+str((left/2)%(L/2)),"l0_"+str((left/2)%(L/2)))
            #O_L4.changeBondName("spin"+str((left/2+1)%(L/2)),"l0_"+str((left/2+1)%(L/2)))
            #O_L4.changeBondName("spin"+str((left/2+2)%(L/2)),"l0_"+str((left/2+2)%(L/2)))
            #O_L4.reorderBonds([ten.Bond("spin"+str((left/2-1+i%4)%(L/2)),2,False if i<4 else True) for i in range(8)])



        elif (left%4)==3:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            for i in range(2):
                c_ten=self.buildTurnOff(4,(left+1+4*i),(left-1+4*i),(left+1+4*i),(left-1+4*i),superlayer)
                if i==0:
                    c_ten.setBondPrime("l4_"+str((left-1)),True)
                if i==1:
                    c_ten.setBondPrime("l4_"+str((left+5)),True)
                if left+5>=L and i==1:
                    O_ten.changeBondName("l3_"+str(left+3),"l4_"+str(left+3))
                    turnOffTenA.append(None)
                    turnOffTenA.append(None)
                else:
                    turnOffTenA.append(c_ten)
                    turnOffTenA.append(self.buildTurnOff(4,(left+1+4*i),(left-1+4*i),(left+1+4*i),(left-1+4*i),superlayer,True))
            O_L1=O_ten
            # upgrade to level 1
            O_L1.changeBondName("l3_"+str(left),"l4_"+str(left))
            O_L1.changeBondName("l3_"+str((left+2)),"l4_"+str((left+2)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(4):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i-1),(left+2*i-2),(left+2*i-1),(left+2*i-2),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-2)),True)
                if i==3:
                    c_ten.setBondPrime("l3_"+str((left+4)),True)
                if left+5>=L and i==3:
                    O_L1.changeBondName("l2_"+str(left+4),"l3_"+str(left+4))
                    turnOnTenB.append(None)
                    turnOnTenB.append(None)
                else:
                    turnOnTenB.append(c_ten)
                    turnOnTenB.append(self.buildTurnOn(level,(left+2*i-1),(left+2*i-2),(left+2*i-1),(left+2*i-2),superlayer,True))
            O_L2=O_L1
            for tenon in turnOnTenB:
                if tenon!=None:
                    tenon.changeBondName("l1_"+str((left-2)),"l2_"+str((left-2)))
                    tenon.changeBondName("l1_"+str((left)),"l2_"+str((left)))
                    tenon.changeBondName("l1_"+str((left+2)),"l2_"+str((left+2)))
                    tenon.changeBondName("l1_"+str((left+4)),"l2_"+str((left+4)))
                    tenon.changeBondName("l1_"+str((left+6)),"l2_"+str((left+6)))
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            #if L==8:
                #for i in range(2):
                    #level=2
                    #turnOffTen.append(self.buildTurnOff(level,(left-3+4*i)%L,(left-1+4*i)%L,(left-3+4*i)%L,(left-1+4*i)%L,superlayer,False))
                    #turnOffTen.append(self.buildTurnOff(level,(left-3+4*i)%L,(left-1+4*i)%L,(left-3+4*i)%L,(left-1+4*i)%L,superlayer,True))
            #else:
            for i in range(3):
                level=2
                c_ten=self.buildTurnOff(level,(left-3+4*i),(left-1+4*i),(left-3+4*i),(left-1+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-3)),True)
                if i==2:
                    c_ten.setBondPrime("l2_"+str((left+7)),True)
                if not (left+5>=L and i==2):
                    turnOffTenC.append(c_ten)
                    turnOffTenC.append(self.buildTurnOff(level,(left-3+4*i),(left-1+4*i),(left-3+4*i),(left-1+4*i),superlayer,True))
                else:
                    turnOffTenC.append(None)
                    turnOffTenC.append(None)
            O_L3=O_L2
            ### Layer 4
            turnOnIsoTenD=[]
            #if L==8:
                #for i in range(4):
                    #level=1
                    #turnOnIsoTen.append(self.buildTurnOnIso(level,((left-3)/2+i)%(L/2),(left-3+2*i)%L,(left-2+2*i)%L,superlayer,False))
                    #turnOnIsoTen.append(self.buildTurnOnIso(level,((left-3)/2+i)%(L/2),(left-3+2*i)%L,(left-2+2*i)%L,superlayer,True))
            #else:
            for i in range(6):
                level=1
                c_ten=self.buildTurnOnIso(level,((left-3)/2+i),(left-3+2*i),(left-2+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==4:
                    c_ten.setBondPrime("l1_"+str((left+6)),True)
                if i==5:
                    c_ten.setBondPrime("l1_"+str((left+8)),True)
                if left+5>=L and i==3:
                    c_ten.setBondPrime("l1_"+str((left+4)),True)
                if not (left+5>=L and i>3):
                    turnOnIsoTenD.append(c_ten)
                    turnOnIsoTenD.append(self.buildTurnOnIso(level,((left-3)/2+i),(left-3+2*i),(left-2+2*i),superlayer,True))
                else:
                    turnOnIsoTenD.append(None)
                    turnOnIsoTenD.append(None)
            O_L4=O_L3


            if left+5<L:
                O_L4=ten.contract(O_L4,turnOffTenA[2])
            O_L4=ten.contract(O_L4,turnOnTenB[4])
            O_L4=ten.contract(O_L4,turnOffTenA[0])
            O_L4=ten.contract(O_L4,turnOnTenB[2])
            O_L4=ten.contract(O_L4,turnOffTenC[2])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[4])
            if left+5<L:
                O_L4=ten.contract(O_L4,turnOnTenB[6])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[6])
            if left+5<L:
                O_L4=ten.contract(O_L4,turnOffTenA[3])
            O_L4=ten.contract(O_L4,turnOnTenB[5])
            O_L4=ten.contract(O_L4,turnOffTenA[1])
            O_L4=ten.contract(O_L4,turnOnTenB[3])
            O_L4=ten.contract(O_L4,turnOffTenC[3])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[5])
            if left+5<L:
                O_L4=ten.contract(O_L4,turnOnTenB[7])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[7])
            if left+5<L:
                O_L4=ten.contract(O_L4,turnOffTenC[4])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[8])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[10])
                O_L4=ten.contract(O_L4,turnOffTenC[5])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[9])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[11])
            O_L4=ten.contract(O_L4,turnOnTenB[0])
            O_L4=ten.contract(O_L4,turnOffTenC[0])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[0])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[2])
            O_L4=ten.contract(O_L4,turnOnTenB[1])
            O_L4=ten.contract(O_L4,turnOffTenC[1])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[1])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[3])

            if not OisTensor:
                aa=np.max(((left-3)/2,0))
                bb=np.min(((left-3)/2+5,L/2-1))
                for i in range(aa,bb+1):
                    O_L4.changeBondName("spin"+str(i),"l0_"+str(i))
                O_L4.reorderBonds([ten.Bond("spin"+str(aa+i%(bb-aa+1)),2,False if i<(bb-aa+1) else True) for i in range(2*(bb-aa+1))])
            #if L==8:
                #for i in range(4):
                    #O_L4.changeBondName("spin"+str(i),"l0_"+str(i))
                #O_L4.reorderBonds([ten.Bond("spin"+str(i%4),2,False if i<4 else True) for i in range(8)])
            #else:
                #for i in range(6):
                    #O_L4.changeBondName("spin"+str(((left-3)/2+i)%(L/2)),"l0_"+str(((left-3)/2+i)%(L/2)))
                #O_L4.reorderBonds([ten.Bond("spin"+str(((left-3)/2+i%6)%(L/2)),2,False if i<6 else True) for i in range(12)])


        return O_L4


    def DRGStep4site(self,O,layer,leg1,left,L,superlayer,ghost=False):
        """
        :param O:
        Assumed to be a 2,2,2,2*2,2,2,2 local 4-site operator on sites left through (left+3)%L
        L is the length of the system (assumes periodic BCs). Must be a multiple of 4 and >=8.
        :return:
        O after single RG step
        """
        if left!=(left%L) or left+3>=L:
            print("Operator to renormalize is out of bounds! (size 4)")
            return None

        if layer==4 or layer==2:
            if leg1%4!=0:
                print("Invalid tensor specification! Pass the leg divisible by 4.")
                return None
        elif layer==3 or layer==1:
            if leg1%2!=0:
                print("Invalid tensor specification! Pass the even leg.")
                return None
        else:
            print("Invalid layer!")
            return None

        # O Tensor
        bondsO_layer_1=[ten.Bond("l4_"+str(left+(i%4)),2,False if i<4 else True) for i in range(8)]
        O_ten=ten.Tensor(O,bondsO_layer_1)

        if (left%4)==0:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            for i in range(2):
                c_ten=self.buildTurnOff(4,(left+4*i),(left+4*i-2),(left+4*i),(left+4*i-2),superlayer)
                if i==0:
                    c_ten.setBondPrime("l4_"+str((left-2)),True)
                if i==1:
                    c_ten.setBondPrime("l4_"+str((left+4)),True)
                if left==0 and i==0:
                    O_ten.changeBondName("l2_0","l4_0")
                if left+4>=L and i==1:
                    O_ten.changeBondName("l3_"+str(left+2),"l4_"+str(left+2))
                    O_ten.changeBondName("l1_"+str(left+3),"l4_"+str(left+3))
                if not ((left==0 and i==0) or (left+4>=L and i==1)):
                    turnOffTenA.append(c_ten)
                    turnOffTenA.append(self.buildTurnOff(4,(left+4*i),(left+4*i-2),(left+4*i),(left+4*i-2),superlayer,True))
                else:
                    turnOffTenA.append(None)
                    turnOffTenA.append(None)
            O_L1=O_ten
            O_L1.changeBondName("l3_"+str((left+1)),"l4_"+str((left+1)))
            O_L1.changeBondName("l3_"+str((left+3)),"l4_"+str((left+3)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(4):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i-2),(left+2*i-3),(left+2*i-2),(left+2*i-3),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-3)),True)
                if i==1:
                    c_ten.setBondPrime("l3_"+str((left-1)),True)
                if left+4>=L and i==3:
                    O_L1.changeBondName("l2_"+str(left+3),"l3_"+str(left+3))
                if not ((left==0 and i<2) or (left+4>=L and i==3)):
                    turnOnTenB.append(c_ten)
                    turnOnTenB.append(self.buildTurnOn(level,(left+2*i-2),(left+2*i-3),(left+2*i-2),(left+2*i-3),superlayer,True))
                else:
                    turnOnTenB.append(None)
                    turnOnTenB.append(None)
            O_L2=O_L1
            #for tenon in turnOnTen:
                #O_L2=ten.contract(O_L2,tenon)
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            #if L==8:
                #for i in range(2):
                    #level=2
                    #turnOffTen.append(self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,False))
                    #turnOffTen.append(self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,True))
            #else:
            for i in range(3):
                level=2
                c_ten=self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-4)),True)
                if i==2:
                        c_ten.setBondPrime("l2_"+str((left+6)),True)
                if not ((left==0 and i==0) or (left+4>=L and i==2)):
                    turnOffTenC.append(c_ten)
                    turnOffTenC.append(self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,True))
                else:
                    turnOffTenC.append(None)
                    turnOffTenC.append(None)
            O_L3=O_L2
            #for tenoff in turnOffTen:
                #O_L3=ten.contract(O_L3,tenoff)
            # upgrade to level 3
            for tenon in turnOnTenB:
                if tenon!=None:
                    tenon.changeBondName("l1_"+str((left-3)),"l2_"+str((left-3)))
                    tenon.changeBondName("l1_"+str((left-1)),"l2_"+str((left-1)))
                    tenon.changeBondName("l1_"+str((left+1)),"l2_"+str((left+1)))
                    tenon.changeBondName("l1_"+str((left+3)),"l2_"+str((left+3)))
                    tenon.changeBondName("l1_"+str((left+5)),"l2_"+str((left+5)))
            ### Layer 4
            turnOnIsoTenD=[]
            #if L==8:
                #for i in range(4):
                    #level=1
                    #turnOnIsoTen.append(self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,False))
                    #turnOnIsoTen.append(self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,True))
            #else:
            for i in range(6):
                level=1
                c_ten=self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==4:
                    c_ten.setBondPrime("l1_"+str((left+5)),True)
                if i==5:
                    c_ten.setBondPrime("l1_"+str((left+7)),True)
                if not ((left==0 and i<2) or (left+4>=L and i>3)):
                    turnOnIsoTenD.append(c_ten)
                    turnOnIsoTenD.append(self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,True))
                else:
                    turnOnIsoTenD.append(None)
                    turnOnIsoTenD.append(None)
            O_L4=O_L3
            #for tenonIso in turnOnIsoTen:
                #O_L4=ten.contract(O_L4,tenonIso)

            if layer==4:
                #legs=["l4_"+str(leg1),"l4_"+str(leg1-2),"l3_"+str(leg1),"l3_"+str(leg1-2)]
                for i in range(len(turnOffTenA)/2):
                    if turnOffTenA[2*i+1]!=None:
                        if turnOffTenA[2*i+1].checkIfBondExist(ten.Bond("l4_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenA[2*i+1].bonds]
                                turnOffTenA[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenA[2*i].bonds]
                                turnOffTenA[2*i]=None
            elif layer==3:
                #legs=["l3_"+str(leg1),"l3_"+str(leg1-1),"l2_"+str(leg1),"l2_"+str(leg1-1)]
                for i in range(len(turnOnTenB)/2):
                    if turnOnTenB[2*i+1]!=None:
                        if turnOnTenB[2*i+1].checkIfBondExist(ten.Bond("l3_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnTenB[2*i+1].bonds]
                                turnOnTenB[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnTenB[2*i].bonds]
                                turnOnTenB[2*i]=None
            elif layer==2:
                #legs=["l2_"+str(leg1),"l2_"+str(leg1+2),"l1_"+str(leg1),"l1_"+str(leg1+2)]
                for i in range(len(turnOffTenC)/2):
                    if turnOffTenC[2*i+1]!=None:
                        if turnOffTenC[2*i+1].checkIfBondExist(ten.Bond("l2_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenC[2*i+1].bonds]
                                turnOffTenC[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenC[2*i].bonds]
                                turnOffTenC[2*i]=None
            elif layer==1:
                #legs=["l1_"+str(leg1),"l1_"+str(leg1+1)]
                for i in range(len(turnOnIsoTenD)/2):
                    if turnOnIsoTenD[2*i+1]!=None:
                        if turnOnIsoTenD[2*i+1].checkIfBondExist(ten.Bond("l1_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnIsoTenD[2*i+1].bonds]
                                turnOnIsoTenD[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnIsoTenD[2*i].bonds]
                                turnOnIsoTenD[2*i]=None
            tens=[]
            if ghost:
                tens.append(turnOffTenA[2])
                tens.append(turnOnTenB[6])
                tens.append(turnOnTenB[4])
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOffTenA[3])
                tens.append(turnOnTenB[7])
                tens.append(turnOnTenB[5])
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOffTenC[4])
                tens.append(turnOnIsoTenD[8])
                tens.append(turnOnIsoTenD[10])
                tens.append(turnOffTenC[5])
                tens.append(turnOnIsoTenD[9])
                tens.append(turnOnIsoTenD[11])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
            else:
                tens.append(turnOffTenA[3])
                tens.append(turnOnTenB[7])
                tens.append(turnOnTenB[5])
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOffTenA[2])
                tens.append(turnOnTenB[6])
                tens.append(turnOnTenB[4])
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOffTenC[5])
                tens.append(turnOnIsoTenD[9])
                tens.append(turnOnIsoTenD[11])
                tens.append(turnOffTenC[4])
                tens.append(turnOnIsoTenD[8])
                tens.append(turnOnIsoTenD[10])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])
            for tns in tens:
                if tns!=None:
                    O_L4=ten.contract(O_L4,tns)
            legsNew=["a","b","c","d"]
            if layer==1:
                bnds=[ten.Bond("l0_"+str(leg1/2),2,ghost),ten.Bond("c",2,ghost)]
                O_q=ten.Tensor(s0,bnds)
                O_L4=ten.contract(O_L4,O_q)#fixes naming of this bond when it's reintroduced in the next layer
                for i in range(2):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)
            else:
                for i in range(4):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)

        elif (left%4)==1:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            turnOffTenA.append(self.buildTurnOff(4,(left+3),(left+1),(left+3),(left+1),superlayer))
            turnOffTenA.append(self.buildTurnOff(4,(left+3),(left+1),(left+3),(left+1),superlayer,True))
            O_L1=O_ten
            # upgrade to level 1
            O_L1.changeBondName("l3_"+str(left),"l4_"+str(left))
            O_L1.changeBondName("l3_"+str((left+2)),"l4_"+str((left+2)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(2):
                level=3
                turnOnTenB.append(self.buildTurnOn(level,(left+1+2*i),(left+2*i),(left+1+2*i),(left+2*i),superlayer,False))
                turnOnTenB.append(self.buildTurnOn(level,(left+1+2*i),(left+2*i),(left+1+2*i),(left+2*i),superlayer,True))
            O_L2=O_L1
            for tenon in turnOnTenB:
                tenon.changeBondName("l1_"+str((left)),"l2_"+str((left)))
                tenon.changeBondName("l1_"+str((left+2)),"l2_"+str((left+2)))
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            for i in range(2):
                level=2
                c_ten=self.buildTurnOff(level,(left-1+4*i),(left+1+4*i),(left-1+4*i),(left+1+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-1)),True)
                if i==1:
                    c_ten.setBondPrime("l2_"+str((left+5)),True)
                turnOffTenC.append(c_ten)
                turnOffTenC.append(self.buildTurnOff(level,(left-1+4*i),(left+1+4*i),(left-1+4*i),(left+1+4*i),superlayer,True))
            O_L3=O_L2
            ### Layer 4
            turnOnIsoTenD=[]
            for i in range(4):
                level=1
                c_ten=self.buildTurnOnIso(level,((left-1)/2+i),(left-1+2*i),(left+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==2:
                    c_ten.setBondPrime("l1_"+str((left+4)),True)
                if i==3:
                    c_ten.setBondPrime("l1_"+str((left+6)),True)
                turnOnIsoTenD.append(c_ten)
                turnOnIsoTenD.append(self.buildTurnOnIso(level,((left-1)/2+i),(left-1+2*i),(left+2*i),superlayer,True))
            O_L4=O_L3


            if layer==4:
                #legs=["l4_"+str(leg1),"l4_"+str(leg1-2),"l3_"+str(leg1),"l3_"+str(leg1-2)]
                for i in range(len(turnOffTenA)/2):
                    if turnOffTenA[2*i+1]!=None:
                        if turnOffTenA[2*i+1].checkIfBondExist(ten.Bond("l4_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenA[2*i+1].bonds]
                                turnOffTenA[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenA[2*i].bonds]
                                turnOffTenA[2*i]=None
            elif layer==3:
                #legs=["l3_"+str(leg1),"l3_"+str(leg1-1),"l2_"+str(leg1),"l2_"+str(leg1-1)]
                for i in range(len(turnOnTenB)/2):
                    if turnOnTenB[2*i+1]!=None:
                        if turnOnTenB[2*i+1].checkIfBondExist(ten.Bond("l3_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnTenB[2*i+1].bonds]
                                turnOnTenB[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnTenB[2*i].bonds]
                                turnOnTenB[2*i]=None
            elif layer==2:
                #legs=["l2_"+str(leg1),"l2_"+str(leg1+2),"l1_"+str(leg1),"l1_"+str(leg1+2)]
                for i in range(len(turnOffTenC)/2):
                    if turnOffTenC[2*i+1]!=None:
                        if turnOffTenC[2*i+1].checkIfBondExist(ten.Bond("l2_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenC[2*i+1].bonds]
                                turnOffTenC[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenC[2*i].bonds]
                                turnOffTenC[2*i]=None
            elif layer==1:
                #legs=["l1_"+str(leg1),"l1_"+str(leg1+1)]
                for i in range(len(turnOnIsoTenD)/2):
                    if turnOnIsoTenD[2*i+1]!=None:
                        if turnOnIsoTenD[2*i+1].checkIfBondExist(ten.Bond("l1_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnIsoTenD[2*i+1].bonds]
                                turnOnIsoTenD[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnIsoTenD[2*i].bonds]
                                turnOnIsoTenD[2*i]=None
            tens=[]
            if ghost:
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnIsoTenD[7])
            else:
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnIsoTenD[6])

            for tns in tens:
                if tns!=None:
                    O_L4=ten.contract(O_L4,tns)
            legsNew=["a","b","c","d"]
            if layer==1:
                bnds=[ten.Bond("l0_"+str(leg1/2),2,ghost),ten.Bond("c",2,ghost)]
                O_q=ten.Tensor(s0,bnds)
                O_L4=ten.contract(O_L4,O_q)#fixes naming of this bond when it's reintroduced in the next layer
                for i in range(2):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)
            else:
                for i in range(4):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)

        elif (left%4)==2:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            turnOffTenA.append(self.buildTurnOff(4,(left+2),left,(left+2),left,superlayer))
            turnOffTenA.append(self.buildTurnOff(4,(left+2),left,(left+2),left,superlayer,True))
            O_L1=O_ten
            # upgrade to level 1
            O_L1.changeBondName("l3_"+str((left+1)),"l4_"+str((left+1)))
            O_L1.changeBondName("l3_"+str((left+3)),"l4_"+str((left+3)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(3):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i),(left+2*i-1),(left+2*i),(left+2*i-1),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-1)),True)
                if i==2:
                    c_ten.setBondPrime("l3_"+str((left+4)),True)
                turnOnTenB.append(c_ten)
                turnOnTenB.append(self.buildTurnOn(level,(left+2*i),(left+2*i-1),(left+2*i),(left+2*i-1),superlayer,True))
            O_L2=O_L1
            for tenon in turnOnTenB:
                tenon.changeBondName("l1_"+str((left-1)),"l2_"+str((left-1)))
                tenon.changeBondName("l1_"+str((left+1)),"l2_"+str((left+1)))
                tenon.changeBondName("l1_"+str((left+3)),"l2_"+str((left+3)))
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            for i in range(2):
                level=2
                c_ten=self.buildTurnOff(level,(left-2+4*i),(left+4*i),(left-2+4*i),(left+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-2)),True)
                turnOffTenC.append(c_ten)
                turnOffTenC.append(self.buildTurnOff(level,(left-2+4*i),(left+4*i),(left-2+4*i),(left+4*i),superlayer,True))
            O_L3=O_L2
            ### Layer 4
            turnOnIsoTenD=[]
            for i in range(4):
                level=1
                c_ten=self.buildTurnOnIso(level,(left/2-1+i),(left-2+2*i),(left-1+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==3:
                    c_ten.setBondPrime("l1_"+str((left+5)),True)
                turnOnIsoTenD.append(c_ten)
                turnOnIsoTenD.append(self.buildTurnOnIso(level,(left/2-1+i),(left-2+2*i),(left-1+2*i),superlayer,True))
            O_L4=O_L3

            if layer==4:
                #legs=["l4_"+str(leg1),"l4_"+str(leg1-2),"l3_"+str(leg1),"l3_"+str(leg1-2)]
                for i in range(len(turnOffTenA)/2):
                    if turnOffTenA[2*i+1]!=None:
                        if turnOffTenA[2*i+1].checkIfBondExist(ten.Bond("l4_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenA[2*i+1].bonds]
                                turnOffTenA[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenA[2*i].bonds]
                                turnOffTenA[2*i]=None
            elif layer==3:
                #legs=["l3_"+str(leg1),"l3_"+str(leg1-1),"l2_"+str(leg1),"l2_"+str(leg1-1)]
                for i in range(len(turnOnTenB)/2):
                    if turnOnTenB[2*i+1]!=None:
                        if turnOnTenB[2*i+1].checkIfBondExist(ten.Bond("l3_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnTenB[2*i+1].bonds]
                                turnOnTenB[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnTenB[2*i].bonds]
                                turnOnTenB[2*i]=None
            elif layer==2:
                #legs=["l2_"+str(leg1),"l2_"+str(leg1+2),"l1_"+str(leg1),"l1_"+str(leg1+2)]
                for i in range(len(turnOffTenC)/2):
                    if turnOffTenC[2*i+1]!=None:
                        if turnOffTenC[2*i+1].checkIfBondExist(ten.Bond("l2_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenC[2*i+1].bonds]
                                turnOffTenC[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenC[2*i].bonds]
                                turnOffTenC[2*i]=None
            elif layer==1:
                #legs=["l1_"+str(leg1),"l1_"+str(leg1+1)]
                for i in range(len(turnOnIsoTenD)/2):
                    if turnOnIsoTenD[2*i+1]!=None:
                        if turnOnIsoTenD[2*i+1].checkIfBondExist(ten.Bond("l1_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnIsoTenD[2*i+1].bonds]
                                turnOnIsoTenD[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnIsoTenD[2*i].bonds]
                                turnOnIsoTenD[2*i]=None
            tens=[]
            if ghost:
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOnTenB[4])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOnTenB[5])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
            else:
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOnTenB[5])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOnTenB[4])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])

            for tns in tens:
                if tns!=None:
                    O_L4=ten.contract(O_L4,tns)
            legsNew=["a","b","c","d"]
            if layer==1:
                bnds=[ten.Bond("l0_"+str(leg1/2),2,ghost),ten.Bond("c",2,ghost)]
                O_q=ten.Tensor(s0,bnds)
                O_L4=ten.contract(O_L4,O_q)#fixes naming of this bond when it's reintroduced in the next layer
                for i in range(2):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)
            else:
                for i in range(4):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)



        elif (left%4)==3:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            for i in range(2):
                c_ten=self.buildTurnOff(4,(left+1+4*i),(left-1+4*i),(left+1+4*i),(left-1+4*i),superlayer)
                if i==0:
                    c_ten.setBondPrime("l4_"+str((left-1)),True)
                if i==1:
                    c_ten.setBondPrime("l4_"+str((left+5)),True)
                if left+5>=L and i==1:
                    O_ten.changeBondName("l3_"+str(left+3),"l4_"+str(left+3))
                    turnOffTenA.append(None)
                    turnOffTenA.append(None)
                else:
                    turnOffTenA.append(c_ten)
                    turnOffTenA.append(self.buildTurnOff(4,(left+1+4*i),(left-1+4*i),(left+1+4*i),(left-1+4*i),superlayer,True))
            O_L1=O_ten
            # upgrade to level 1
            O_L1.changeBondName("l3_"+str(left),"l4_"+str(left))
            O_L1.changeBondName("l3_"+str((left+2)),"l4_"+str((left+2)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(4):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i-1),(left+2*i-2),(left+2*i-1),(left+2*i-2),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-2)),True)
                if i==3:
                    c_ten.setBondPrime("l3_"+str((left+4)),True)
                if left+5>=L and i==3:
                    O_L1.changeBondName("l2_"+str(left+4),"l3_"+str(left+4))
                    turnOnTenB.append(None)
                    turnOnTenB.append(None)
                else:
                    turnOnTenB.append(c_ten)
                    turnOnTenB.append(self.buildTurnOn(level,(left+2*i-1),(left+2*i-2),(left+2*i-1),(left+2*i-2),superlayer,True))
            O_L2=O_L1
            for tenon in turnOnTenB:
                if tenon!=None:
                    tenon.changeBondName("l1_"+str((left-2)),"l2_"+str((left-2)))
                    tenon.changeBondName("l1_"+str((left)),"l2_"+str((left)))
                    tenon.changeBondName("l1_"+str((left+2)),"l2_"+str((left+2)))
                    tenon.changeBondName("l1_"+str((left+4)),"l2_"+str((left+4)))
                    tenon.changeBondName("l1_"+str((left+6)),"l2_"+str((left+6)))
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            #if L==8:
                #for i in range(2):
                    #level=2
                    #turnOffTen.append(self.buildTurnOff(level,(left-3+4*i)%L,(left-1+4*i)%L,(left-3+4*i)%L,(left-1+4*i)%L,superlayer,False))
                    #turnOffTen.append(self.buildTurnOff(level,(left-3+4*i)%L,(left-1+4*i)%L,(left-3+4*i)%L,(left-1+4*i)%L,superlayer,True))
            #else:
            for i in range(3):
                level=2
                c_ten=self.buildTurnOff(level,(left-3+4*i),(left-1+4*i),(left-3+4*i),(left-1+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-3)),True)
                if i==2:
                    c_ten.setBondPrime("l2_"+str((left+7)),True)
                if not (left+5>=L and i==2):
                    turnOffTenC.append(c_ten)
                    turnOffTenC.append(self.buildTurnOff(level,(left-3+4*i),(left-1+4*i),(left-3+4*i),(left-1+4*i),superlayer,True))
                else:
                    turnOffTenC.append(None)
                    turnOffTenC.append(None)
            O_L3=O_L2
            ### Layer 4
            turnOnIsoTenD=[]
            #if L==8:
                #for i in range(4):
                    #level=1
                    #turnOnIsoTen.append(self.buildTurnOnIso(level,((left-3)/2+i)%(L/2),(left-3+2*i)%L,(left-2+2*i)%L,superlayer,False))
                    #turnOnIsoTen.append(self.buildTurnOnIso(level,((left-3)/2+i)%(L/2),(left-3+2*i)%L,(left-2+2*i)%L,superlayer,True))
            #else:
            for i in range(6):
                level=1
                c_ten=self.buildTurnOnIso(level,((left-3)/2+i),(left-3+2*i),(left-2+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==4:
                    c_ten.setBondPrime("l1_"+str((left+6)),True)
                if i==5:
                    c_ten.setBondPrime("l1_"+str((left+8)),True)
                if left+5>=L and i==3:
                    c_ten.setBondPrime("l1_"+str((left+4)),True)
                if not (left+5>=L and i>3):
                    turnOnIsoTenD.append(c_ten)
                    turnOnIsoTenD.append(self.buildTurnOnIso(level,((left-3)/2+i),(left-3+2*i),(left-2+2*i),superlayer,True))
                else:
                    turnOnIsoTenD.append(None)
                    turnOnIsoTenD.append(None)
            O_L4=O_L3


            if layer==4:
                #legs=["l4_"+str(leg1),"l4_"+str(leg1-2),"l3_"+str(leg1),"l3_"+str(leg1-2)]
                for i in range(len(turnOffTenA)/2):
                    if turnOffTenA[2*i+1]!=None:
                        if turnOffTenA[2*i+1].checkIfBondExist(ten.Bond("l4_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenA[2*i+1].bonds]
                                turnOffTenA[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenA[2*i].bonds]
                                turnOffTenA[2*i]=None
            elif layer==3:
                #legs=["l3_"+str(leg1),"l3_"+str(leg1-1),"l2_"+str(leg1),"l2_"+str(leg1-1)]
                for i in range(len(turnOnTenB)/2):
                    if turnOnTenB[2*i+1]!=None:
                        if turnOnTenB[2*i+1].checkIfBondExist(ten.Bond("l3_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnTenB[2*i+1].bonds]
                                turnOnTenB[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnTenB[2*i].bonds]
                                turnOnTenB[2*i]=None
            elif layer==2:
                #legs=["l2_"+str(leg1),"l2_"+str(leg1+2),"l1_"+str(leg1),"l1_"+str(leg1+2)]
                for i in range(len(turnOffTenC)/2):
                    if turnOffTenC[2*i+1]!=None:
                        if turnOffTenC[2*i+1].checkIfBondExist(ten.Bond("l2_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenC[2*i+1].bonds]
                                turnOffTenC[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenC[2*i].bonds]
                                turnOffTenC[2*i]=None
            elif layer==1:
                #legs=["l1_"+str(leg1),"l1_"+str(leg1+1)]
                for i in range(len(turnOnIsoTenD)/2):
                    if turnOnIsoTenD[2*i+1]!=None:
                        if turnOnIsoTenD[2*i+1].checkIfBondExist(ten.Bond("l1_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnIsoTenD[2*i+1].bonds]
                                turnOnIsoTenD[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnIsoTenD[2*i].bonds]
                                turnOnIsoTenD[2*i]=None
            tens=[]
            if ghost:
                tens.append(turnOffTenA[2])
                tens.append(turnOnTenB[4])
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnTenB[6])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOffTenA[3])
                tens.append(turnOnTenB[5])
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnTenB[7])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOffTenC[4])
                tens.append(turnOnIsoTenD[8])
                tens.append(turnOnIsoTenD[10])
                tens.append(turnOffTenC[5])
                tens.append(turnOnIsoTenD[9])
                tens.append(turnOnIsoTenD[11])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
            else:
                tens.append(turnOffTenA[3])
                tens.append(turnOnTenB[5])
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnTenB[7])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOffTenA[2])
                tens.append(turnOnTenB[4])
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnTenB[6])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOffTenC[5])
                tens.append(turnOnIsoTenD[9])
                tens.append(turnOnIsoTenD[11])
                tens.append(turnOffTenC[4])
                tens.append(turnOnIsoTenD[8])
                tens.append(turnOnIsoTenD[10])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])

            for tns in tens:
                if tns!=None:
                    O_L4=ten.contract(O_L4,tns)
            legsNew=["a","b","c","d"]
            if layer==1:
                bnds=[ten.Bond("l0_"+str(leg1/2),2,ghost),ten.Bond("c",2,ghost)]
                O_q=ten.Tensor(s0,bnds)
                O_L4=ten.contract(O_L4,O_q)#fixes naming of this bond when it's reintroduced in the next layer
                for i in range(2):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)
            else:
                for i in range(4):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)

        return O_L4



    def RGStep6site(self,O,left,L,superlayer,OisTensor=False):
        """
            :param O:
            Assumed to be a 2,2,2,2,2,2*2,2,2,2,2,2 local 6-site operator on sites left through (left+5)%L
            with left even
            L is the length of the system (assumes periodic BCs). Must be a multiple of 4 and >=8.
            :return:
            O after single RG step
            """
        if left!=(left%L) or left+5>=L:
            print("Operator to renormalize is out of bounds! (size 6)")
            return None

        # O Tensor
        if not OisTensor:
            bondsO_layer_1=[ten.Bond("l4_"+str((left+(i%6))),2,False if i<6 else True) for i in range(12)]
            O_ten=ten.Tensor(O,bondsO_layer_1)
        else:
            O_ten=O

        if (left%4)==0:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            for i in range(2):
                c_ten=self.buildTurnOff(4,(left+4*i),(left+4*i-2),(left+4*i),(left+4*i-2),superlayer)
                if i==0:
                    c_ten.setBondPrime("l4_"+str((left-2)),True)
                if left==0 and i==0:
                    O_ten.changeBondName("l2_0","l4_0")
                    turnOffTenA.append(None)
                    turnOffTenA.append(None)
                else:
                    turnOffTenA.append(c_ten)
                    turnOffTenA.append(self.buildTurnOff(4,(left+4*i),(left+4*i-2),(left+4*i),(left+4*i-2),superlayer,True))
            O_L1=O_ten
            #for tenoff in turnOffTenA:
                #O_L1=ten.contract(O_L1,tenoff)
            #rename bonds
            O_L1.changeBondName("l3_"+str((left+1)),"l4_"+str((left+1)))
            O_L1.changeBondName("l3_"+str((left+3)),"l4_"+str((left+3)))
            O_L1.changeBondName("l3_"+str((left+5)),"l4_"+str((left+5)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(5):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i-2),(left+2*i-3),(left+2*i-2),(left+2*i-3),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-3)),True)
                if i==1:
                    c_ten.setBondPrime("l3_"+str((left-1)),True)
                if i==4:
                    c_ten.setBondPrime("l3_"+str((left+6)),True)
                if left==0 and i<2:
                    turnOnTenB.append(None)
                    turnOnTenB.append(None)
                else:
                    turnOnTenB.append(c_ten)
                    turnOnTenB.append(self.buildTurnOn(level,(left+2*i-2),(left+2*i-3),(left+2*i-2),(left+2*i-3),superlayer,True))
            O_L2=O_L1
            #for tenon in turnOnTenB:
                #O_L2=ten.contract(O_L2,tenon)
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            for i in range(3):
                level=2
                c_ten=self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-4)),True)
                if left==0 and i==0:
                    turnOffTenC.append(None)
                    turnOffTenC.append(None)
                else:
                    turnOffTenC.append(c_ten)
                    turnOffTenC.append(self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,True))
            O_L3=O_L2
            #for tenoff in turnOffTenC:
                #O_L3=ten.contract(O_L3,tenoff)
            #change names
            for tenon in turnOnTenB:
                if tenon!=None:
                    tenon.changeBondName("l1_"+str((left-3)),"l2_"+str((left-3)))
                    tenon.changeBondName("l1_"+str((left-1)),"l2_"+str((left-1)))
                    tenon.changeBondName("l1_"+str((left+1)),"l2_"+str((left+1)))
                    tenon.changeBondName("l1_"+str((left+3)),"l2_"+str((left+3)))
                    tenon.changeBondName("l1_"+str((left+5)),"l2_"+str((left+5)))
            ### Layer 4
            turnOnIsoTenD=[]
            for i in range(6):
                level=1
                c_ten=self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==5:
                    c_ten.setBondPrime("l1_"+str((left+7)),True)
                if left==0 and i<2:
                    turnOnIsoTenD.append(None)
                    turnOnIsoTenD.append(None)
                else:
                    turnOnIsoTenD.append(c_ten)
                    turnOnIsoTenD.append(self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,True))
            O_L4=O_L3
            #for tenonIso in turnOnIsoTenD:
                #O_L4=ten.contract(O_L4,tenonIso)
            O_L4=ten.contract(O_L4,turnOffTenA[2])
            O_L4=ten.contract(O_L4,turnOnTenB[4])
            O_L4=ten.contract(O_L4,turnOnTenB[6])
            O_L4=ten.contract(O_L4,turnOffTenA[3])
            O_L4=ten.contract(O_L4,turnOnTenB[5])
            O_L4=ten.contract(O_L4,turnOnTenB[7])
            if left>0:
                O_L4=ten.contract(O_L4,turnOffTenA[0])
                O_L4=ten.contract(O_L4,turnOnTenB[2])
            O_L4=ten.contract(O_L4,turnOffTenC[2])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[4])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[6])
            if left>0:
                O_L4=ten.contract(O_L4,turnOffTenA[1])
                O_L4=ten.contract(O_L4,turnOnTenB[3])
            O_L4=ten.contract(O_L4,turnOffTenC[3])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[5])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[7])
            O_L4=ten.contract(O_L4,turnOnTenB[8])
            O_L4=ten.contract(O_L4,turnOffTenC[4])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[8])
            O_L4=ten.contract(O_L4,turnOnTenB[9])
            O_L4=ten.contract(O_L4,turnOffTenC[5])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[9])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[10])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[11])
            if left>0:
                O_L4=ten.contract(O_L4,turnOnTenB[0])
                O_L4=ten.contract(O_L4,turnOffTenC[0])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[0])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[2])
                O_L4=ten.contract(O_L4,turnOnTenB[1])
                O_L4=ten.contract(O_L4,turnOffTenC[1])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[1])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[3])

            if not OisTensor:
                aa=np.max((left/2-2,0))
                bb=np.min((left/2+3,L/2-1))
                for i in range(aa,bb+1):
                    O_L4.changeBondName("spin"+str(i),"l0_"+str(i))
                O_L4.reorderBonds([ten.Bond("spin"+str(aa+i%(bb-aa+1)),2,False if i<(bb-aa+1) else True) for i in range(2*(bb-aa+1))])

        elif (left%4)==2:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            for i in range(2):
                c_ten=self.buildTurnOff(4,(left+2+4*i),(left+4*i),(left+2+4*i),(left+4*i),superlayer)
                if i==1:
                    c_ten.setBondPrime("l4_"+str((left+6)),True)
                if left+6>=L and i==1:
                    O_ten.changeBondName("l3_"+str(left+4),"l4_"+str(left+4))
                else:
                    turnOffTenA.append(c_ten)
                    turnOffTenA.append(self.buildTurnOff(4,(left+2+4*i),(left+4*i),(left+2+4*i),(left+4*i),superlayer,True))
            if left+6>=L:
                O_ten.changeBondName("l1_"+str(left+5),"l4_"+str(left+5))
            O_L1=O_ten
            #for tenoff in turnOffTenA:
                #O_L1=ten.contract(O_L1,tenoff)
            # upgrade to level 1
            O_L1.changeBondName("l3_"+str((left+1)),"l4_"+str((left+1)))
            O_L1.changeBondName("l3_"+str((left+3)),"l4_"+str((left+3)))
            O_L1.changeBondName("l3_"+str((left+5)),"l4_"+str((left+5)))

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(4):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i),(left+2*i-1),(left+2*i),(left+2*i-1),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-1)),True)
                if left+6>=L and i==3:
                    O_L1.changeBondName("l1_"+str(left+5),"l4_"+str(left+5))
                else:
                    turnOnTenB.append(c_ten)
                    turnOnTenB.append(self.buildTurnOn(level,(left+2*i),(left+2*i-1),(left+2*i),(left+2*i-1),superlayer,True))
            O_L2=O_L1
            #for tenon in turnOnTenB:
                #O_L2=ten.contract(O_L2,tenon)
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            for i in range(3):
                level=2
                c_ten=self.buildTurnOff(level,(left-2+4*i),(left+4*i),(left-2+4*i),(left+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-2)),True)
                if i==2:
                    c_ten.setBondPrime("l2_"+str((left+8)),True)
                if not (left+6>=L and i==2):
                    turnOffTenC.append(c_ten)
                    turnOffTenC.append(self.buildTurnOff(level,(left-2+4*i),(left+4*i),(left-2+4*i),(left+4*i),superlayer,True))
            O_L3=O_L2
            #for tenoff in turnOffTenC:
                #O_L3=ten.contract(O_L3,tenoff)
            #change names
            for tenon in turnOnTenB:
                tenon.changeBondName("l1_"+str((left-1)),"l2_"+str((left-1)))
                tenon.changeBondName("l1_"+str((left+1)),"l2_"+str((left+1)))
                tenon.changeBondName("l1_"+str((left+3)),"l2_"+str((left+3)))
                tenon.changeBondName("l1_"+str((left+5)),"l2_"+str((left+5)))
                tenon.changeBondName("l1_"+str((left+7)),"l2_"+str((left+7)))
            ### Layer 4
            turnOnIsoTenD=[]
            for i in range(6):
                level=1
                c_ten=self.buildTurnOnIso(level,(left/2-1+i)%(L/2),(left-2+2*i)%L,(left-1+2*i)%L,superlayer,False)
                # "self" contract tensors - make unprime
                if i==4:
                    c_ten.setBondPrime("l1_"+str((left+7)%L),True)
                if i==5:
                    c_ten.setBondPrime("l1_"+str((left+9)%L),True)
                if not (left+6>=L and i>3):
                    turnOnIsoTenD.append(c_ten)
                    turnOnIsoTenD.append(self.buildTurnOnIso(level,(left/2-1+i)%(L/2),(left-2+2*i)%L,(left-1+2*i)%L,superlayer,True))
            O_L4=O_L3
            #for tenonIso in turnOnIsoTenD:
                #O_L4=ten.contract(O_L4,tenonIso)
            O_L4=ten.contract(O_L4,turnOffTenA[0])
            O_L4=ten.contract(O_L4,turnOnTenB[2])
            if left+6<L:
                O_L4=ten.contract(O_L4,turnOffTenA[2])
            O_L4=ten.contract(O_L4,turnOnTenB[4])
            O_L4=ten.contract(O_L4,turnOffTenC[2])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[4])
            if left+6<L:
                O_L4=ten.contract(O_L4,turnOnTenB[6])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[6])
            O_L4=ten.contract(O_L4,turnOffTenA[1])
            O_L4=ten.contract(O_L4,turnOnTenB[3])
            if left+6<L:
                O_L4=ten.contract(O_L4,turnOffTenA[3])
            O_L4=ten.contract(O_L4,turnOnTenB[5])
            O_L4=ten.contract(O_L4,turnOffTenC[3])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[5])
            if left+6<L:
                O_L4=ten.contract(O_L4,turnOnTenB[7])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[7])
            if left+6<L:
                O_L4=ten.contract(O_L4,turnOffTenC[4])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[8])
                O_L4=ten.contract(O_L4,turnOffTenC[5])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[9])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[10])
                O_L4=ten.contract(O_L4,turnOnIsoTenD[11])
            O_L4=ten.contract(O_L4,turnOnTenB[0])
            O_L4=ten.contract(O_L4,turnOffTenC[0])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[0])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[2])
            O_L4=ten.contract(O_L4,turnOnTenB[1])
            O_L4=ten.contract(O_L4,turnOffTenC[1])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[1])
            O_L4=ten.contract(O_L4,turnOnIsoTenD[3])

            if not OisTensor:
                aa=np.max((left/2-1,0))
                bb=np.min((left/2+4,L/2-1))
                for i in range(aa,bb+1):
                    O_L4.changeBondName("spin"+str(i),"l0_"+str(i))
                O_L4.reorderBonds([ten.Bond("spin"+str(aa+i%(bb-aa+1)),2,False if i<(bb-aa+1) else True) for i in range(2*(bb-aa+1))])
        else:
            print("Blocks of 6 starting on an odd site are not currently supported!")
            return None

        return O_L4

    def DRGStep6site(self,O,layer,leg1,left,L,superlayer,ghost=False):
        """
            :param O:
            Assumed to be a 2,2,2,2,2,2*2,2,2,2,2,2 local 6-site operator on sites left through (left+5)%L
            with left even
            L is the length of the system (assumes periodic BCs). Must be a multiple of 4 and >=8.
            :return:
            O after single RG step
            """
        if left!=(left%L) or left+5>=L:
            print("Operator to renormalize is out of bounds! (size 6)")
            return None
        if layer==4 or layer==2:
            if leg1%4!=0:
                print("Invalid tensor specification! Pass the leg divisible by 4.")
                return None
        elif layer==3 or layer==1:
            if leg1%2!=0:
                print("Invalid tensor specification! Pass the even leg.")
                return None
        else:
            print("Invalid layer!")
            return None

        # O Tensor
        bondsO_layer_1=[ten.Bond("l4_"+str((left+(i%6))),2,False if i<6 else True) for i in range(12)]
        O_ten=ten.Tensor(O,bondsO_layer_1)

        if (left%4)==0:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            for i in range(2):
                c_ten=self.buildTurnOff(4,(left+4*i),(left+4*i-2),(left+4*i),(left+4*i-2),superlayer)
                if i==0:
                    c_ten.setBondPrime("l4_"+str((left-2)),True)
                if left==0 and i==0:
                    O_ten.changeBondName("l2_0","l4_0")
                    turnOffTenA.append(None)
                    turnOffTenA.append(None)
                else:
                    turnOffTenA.append(c_ten)
                    turnOffTenA.append(self.buildTurnOff(4,(left+4*i),(left+4*i-2),(left+4*i),(left+4*i-2),superlayer,True))
            O_L1=O_ten
            #for tenoff in turnOffTenA:
                #O_L1=ten.contract(O_L1,tenoff)
            #rename bonds
            O_L1.changeBondName("l3_"+str((left+1)),"l4_"+str((left+1)))
            O_L1.changeBondName("l3_"+str((left+3)),"l4_"+str((left+3)))
            O_L1.changeBondName("l3_"+str((left+5)),"l4_"+str((left+5)))
            #O_L1.printBonds()
            #O_L1.reorderBonds([ten.Bond("l3_"+str((i%4)+1),2,False if i<4 else True) for i in range(8)])

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(5):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i-2),(left+2*i-3),(left+2*i-2),(left+2*i-3),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-3)),True)
                if i==1:
                    c_ten.setBondPrime("l3_"+str((left-1)),True)
                if i==4:
                    c_ten.setBondPrime("l3_"+str((left+6)),True)
                if left==0 and i<2:
                    turnOnTenB.append(None)
                    turnOnTenB.append(None)
                else:
                    turnOnTenB.append(c_ten)
                    turnOnTenB.append(self.buildTurnOn(level,(left+2*i-2),(left+2*i-3),(left+2*i-2),(left+2*i-3),superlayer,True))
            O_L2=O_L1
            #for tenon in turnOnTenB:
                #O_L2=ten.contract(O_L2,tenon)
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            for i in range(3):
                level=2
                c_ten=self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-4)),True)
                if left==0 and i==0:
                    turnOffTenC.append(None)
                    turnOffTenC.append(None)
                else:
                    turnOffTenC.append(c_ten)
                    turnOffTenC.append(self.buildTurnOff(level,(left-4+4*i),(left-2+4*i),(left-4+4*i),(left-2+4*i),superlayer,True))
            O_L3=O_L2
            #for tenoff in turnOffTenC:
                #O_L3=ten.contract(O_L3,tenoff)
            #change names
            for tenon in turnOnTenB:
                if tenon!=None:
                    tenon.changeBondName("l1_"+str((left-3)),"l2_"+str((left-3)))
                    tenon.changeBondName("l1_"+str((left-1)),"l2_"+str((left-1)))
                    tenon.changeBondName("l1_"+str((left+1)),"l2_"+str((left+1)))
                    tenon.changeBondName("l1_"+str((left+3)),"l2_"+str((left+3)))
                    tenon.changeBondName("l1_"+str((left+5)),"l2_"+str((left+5)))
            ### Layer 4
            turnOnIsoTenD=[]
            for i in range(6):
                level=1
                c_ten=self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==5:
                    c_ten.setBondPrime("l1_"+str((left+7)),True)
                if left==0 and i<2:
                    turnOnIsoTenD.append(None)
                    turnOnIsoTenD.append(None)
                else:
                    turnOnIsoTenD.append(c_ten)
                    turnOnIsoTenD.append(self.buildTurnOnIso(level,(left/2-2+i),(left-4+2*i),(left-3+2*i),superlayer,True))
            O_L4=O_L3

            if layer==4:
                #legs=["l4_"+str(leg1),"l4_"+str(leg1-2),"l3_"+str(leg1),"l3_"+str(leg1-2)]
                for i in range(len(turnOffTenA)/2):
                    if turnOffTenA[2*i+1]!=None:
                        if turnOffTenA[2*i+1].checkIfBondExist(ten.Bond("l4_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenA[2*i+1].bonds]
                                turnOffTenA[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenA[2*i].bonds]
                                turnOffTenA[2*i]=None
            elif layer==3:
                #legs=["l3_"+str(leg1),"l3_"+str(leg1-1),"l2_"+str(leg1),"l2_"+str(leg1-1)]
                for i in range(len(turnOnTenB)/2):
                    if turnOnTenB[2*i+1]!=None:
                        if turnOnTenB[2*i+1].checkIfBondExist(ten.Bond("l3_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnTenB[2*i+1].bonds]
                                turnOnTenB[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnTenB[2*i].bonds]
                                turnOnTenB[2*i]=None
            elif layer==2:
                #legs=["l2_"+str(leg1),"l2_"+str(leg1+2),"l1_"+str(leg1),"l1_"+str(leg1+2)]
                for i in range(len(turnOffTenC)/2):
                    if turnOffTenC[2*i+1]!=None:
                        if turnOffTenC[2*i+1].checkIfBondExist(ten.Bond("l2_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenC[2*i+1].bonds]
                                turnOffTenC[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenC[2*i].bonds]
                                turnOffTenC[2*i]=None
            elif layer==1:
                #legs=["l1_"+str(leg1),"l1_"+str(leg1+1)]
                for i in range(len(turnOnIsoTenD)/2):
                    if turnOnIsoTenD[2*i+1]!=None:
                        if turnOnIsoTenD[2*i+1].checkIfBondExist(ten.Bond("l1_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnIsoTenD[2*i+1].bonds]
                                turnOnIsoTenD[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnIsoTenD[2*i].bonds]
                                turnOnIsoTenD[2*i]=None
            tens=[]
            if ghost:
                tens.append(turnOffTenA[2])
                tens.append(turnOnTenB[4])
                tens.append(turnOnTenB[6])
                tens.append(turnOffTenA[3])
                tens.append(turnOnTenB[5])
                tens.append(turnOnTenB[7])
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOnTenB[8])
                tens.append(turnOffTenC[4])
                tens.append(turnOnIsoTenD[8])
                tens.append(turnOnTenB[9])
                tens.append(turnOffTenC[5])
                tens.append(turnOnIsoTenD[9])
                tens.append(turnOnIsoTenD[10])
                tens.append(turnOnIsoTenD[11])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
            else:
                tens.append(turnOffTenA[3])
                tens.append(turnOnTenB[5])
                tens.append(turnOnTenB[7])
                tens.append(turnOffTenA[2])
                tens.append(turnOnTenB[4])
                tens.append(turnOnTenB[6])
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOnTenB[9])
                tens.append(turnOffTenC[5])
                tens.append(turnOnIsoTenD[9])
                tens.append(turnOnTenB[8])
                tens.append(turnOffTenC[4])
                tens.append(turnOnIsoTenD[8])
                tens.append(turnOnIsoTenD[11])
                tens.append(turnOnIsoTenD[10])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])

            for tns in tens:
                if tns!=None:
                    O_L4=ten.contract(O_L4,tns)
            legsNew=["a","b","c","d"]
            if layer==1:
                bnds=[ten.Bond("l0_"+str(leg1/2),2,ghost),ten.Bond("c",2,ghost)]
                O_q=ten.Tensor(s0,bnds)
                O_L4=ten.contract(O_L4,O_q)#fixes naming of this bond when it's reintroduced in the next layer
                for i in range(2):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)
            else:
                for i in range(4):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)

        elif (left%4)==2:
            ### Layer 1
            # Turn "off" tensors
            turnOffTenA=[]
            for i in range(2):
                c_ten=self.buildTurnOff(4,(left+2+4*i),(left+4*i),(left+2+4*i),(left+4*i),superlayer)
                if i==1:
                    c_ten.setBondPrime("l4_"+str((left+6)),True)
                if left+6>=L and i==1:
                    O_ten.changeBondName("l3_"+str(left+4),"l4_"+str(left+4))
                    turnOffTenA.append(None)
                    turnOffTenA.append(None)
                else:
                    turnOffTenA.append(c_ten)
                    turnOffTenA.append(self.buildTurnOff(4,(left+2+4*i),(left+4*i),(left+2+4*i),(left+4*i),superlayer,True))
            if left+6>=L:
                O_ten.changeBondName("l1_"+str(left+5),"l4_"+str(left+5))
            O_L1=O_ten
            #for tenoff in turnOffTenA:
                #O_L1=ten.contract(O_L1,tenoff)
            # upgrade to level 1
            O_L1.changeBondName("l3_"+str((left+1)),"l4_"+str((left+1)))
            O_L1.changeBondName("l3_"+str((left+3)),"l4_"+str((left+3)))
            O_L1.changeBondName("l3_"+str((left+5)),"l4_"+str((left+5)))

            ### Layer 2
            # turn "on" tensors
            turnOnTenB=[]
            for i in range(4):
                level=3
                c_ten=self.buildTurnOn(level,(left+2*i),(left+2*i-1),(left+2*i),(left+2*i-1),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_"+str((left-1)),True)
                if left+6>=L and i==3:
                    O_L1.changeBondName("l1_"+str(left+5),"l4_"+str(left+5))
                    turnOnTenB.append(None)
                    turnOnTenB.append(None)
                else:
                    turnOnTenB.append(c_ten)
                    turnOnTenB.append(self.buildTurnOn(level,(left+2*i),(left+2*i-1),(left+2*i),(left+2*i-1),superlayer,True))
            O_L2=O_L1
            #for tenon in turnOnTenB:
                #O_L2=ten.contract(O_L2,tenon)
            ### Layer 3
            # turn "off" tensors
            turnOffTenC=[]
            for i in range(3):
                level=2
                c_ten=self.buildTurnOff(level,(left-2+4*i),(left+4*i),(left-2+4*i),(left+4*i),superlayer,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_"+str((left-2)),True)
                if i==2:
                    c_ten.setBondPrime("l2_"+str((left+8)),True)
                if not (left+6>=L and i==2):
                    turnOffTenC.append(c_ten)
                    turnOffTenC.append(self.buildTurnOff(level,(left-2+4*i),(left+4*i),(left-2+4*i),(left+4*i),superlayer,True))
                else:
                    turnOffTenC.append(None)
                    turnOffTenC.append(None)
            O_L3=O_L2
            #for tenoff in turnOffTenC:
                #O_L3=ten.contract(O_L3,tenoff)
            #change names
            for tenon in turnOnTenB:
                if tenon!=None:
                    tenon.changeBondName("l1_"+str((left-1)),"l2_"+str((left-1)))
                    tenon.changeBondName("l1_"+str((left+1)),"l2_"+str((left+1)))
                    tenon.changeBondName("l1_"+str((left+3)),"l2_"+str((left+3)))
                    tenon.changeBondName("l1_"+str((left+5)),"l2_"+str((left+5)))
                    tenon.changeBondName("l1_"+str((left+7)),"l2_"+str((left+7)))
            ### Layer 4
            turnOnIsoTenD=[]
            for i in range(6):
                level=1
                c_ten=self.buildTurnOnIso(level,(left/2-1+i)%(L/2),(left-2+2*i)%L,(left-1+2*i)%L,superlayer,False)
                # "self" contract tensors - make unprime
                if i==4:
                    c_ten.setBondPrime("l1_"+str((left+7)%L),True)
                if i==5:
                    c_ten.setBondPrime("l1_"+str((left+9)%L),True)
                if not (left+6>=L and i>3):
                    turnOnIsoTenD.append(c_ten)
                    turnOnIsoTenD.append(self.buildTurnOnIso(level,(left/2-1+i)%(L/2),(left-2+2*i)%L,(left-1+2*i)%L,superlayer,True))
                else:
                    turnOnIsoTenD.append(None)
                    turnOnIsoTenD.append(None)
            O_L4=O_L3

            if layer==4:
                #legs=["l4_"+str(leg1),"l4_"+str(leg1-2),"l3_"+str(leg1),"l3_"+str(leg1-2)]
                for i in range(len(turnOffTenA)/2):
                    if turnOffTenA[2*i+1]!=None:
                        if turnOffTenA[2*i+1].checkIfBondExist(ten.Bond("l4_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenA[2*i+1].bonds]
                                turnOffTenA[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenA[2*i].bonds]
                                turnOffTenA[2*i]=None
            elif layer==3:
                #legs=["l3_"+str(leg1),"l3_"+str(leg1-1),"l2_"+str(leg1),"l2_"+str(leg1-1)]
                for i in range(len(turnOnTenB)/2):
                    if turnOnTenB[2*i+1]!=None:
                        if turnOnTenB[2*i+1].checkIfBondExist(ten.Bond("l3_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnTenB[2*i+1].bonds]
                                turnOnTenB[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnTenB[2*i].bonds]
                                turnOnTenB[2*i]=None
            elif layer==2:
                #legs=["l2_"+str(leg1),"l2_"+str(leg1+2),"l1_"+str(leg1),"l1_"+str(leg1+2)]
                for i in range(len(turnOffTenC)/2):
                    if turnOffTenC[2*i+1]!=None:
                        if turnOffTenC[2*i+1].checkIfBondExist(ten.Bond("l2_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOffTenC[2*i+1].bonds]
                                turnOffTenC[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOffTenC[2*i].bonds]
                                turnOffTenC[2*i]=None
            elif layer==1:
                #legs=["l1_"+str(leg1),"l1_"+str(leg1+1)]
                for i in range(len(turnOnIsoTenD)/2):
                    if turnOnIsoTenD[2*i+1]!=None:
                        if turnOnIsoTenD[2*i+1].checkIfBondExist(ten.Bond("l1_"+str(leg1),2,True)):
                            if ghost:
                                legs=[x.name for x in turnOnIsoTenD[2*i+1].bonds]
                                turnOnIsoTenD[2*i+1]=None
                            else:
                                legs=[x.name for x in turnOnIsoTenD[2*i].bonds]
                                turnOnIsoTenD[2*i]=None
            tens=[]
            if ghost:
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenA[2])
                tens.append(turnOnTenB[4])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnTenB[6])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenA[3])
                tens.append(turnOnTenB[5])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnTenB[7])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOffTenC[4])
                tens.append(turnOnIsoTenD[8])
                tens.append(turnOffTenC[5])
                tens.append(turnOnIsoTenD[9])
                tens.append(turnOnIsoTenD[10])
                tens.append(turnOnIsoTenD[11])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
            else:
                tens.append(turnOffTenA[1])
                tens.append(turnOnTenB[3])
                tens.append(turnOffTenA[3])
                tens.append(turnOnTenB[5])
                tens.append(turnOffTenC[3])
                tens.append(turnOnIsoTenD[5])
                tens.append(turnOnTenB[7])
                tens.append(turnOnIsoTenD[7])
                tens.append(turnOffTenA[0])
                tens.append(turnOnTenB[2])
                tens.append(turnOffTenA[2])
                tens.append(turnOnTenB[4])
                tens.append(turnOffTenC[2])
                tens.append(turnOnIsoTenD[4])
                tens.append(turnOnTenB[6])
                tens.append(turnOnIsoTenD[6])
                tens.append(turnOffTenC[5])
                tens.append(turnOnIsoTenD[9])
                tens.append(turnOffTenC[4])
                tens.append(turnOnIsoTenD[8])
                tens.append(turnOnIsoTenD[11])
                tens.append(turnOnIsoTenD[10])
                tens.append(turnOnTenB[1])
                tens.append(turnOffTenC[1])
                tens.append(turnOnIsoTenD[1])
                tens.append(turnOnIsoTenD[3])
                tens.append(turnOnTenB[0])
                tens.append(turnOffTenC[0])
                tens.append(turnOnIsoTenD[0])
                tens.append(turnOnIsoTenD[2])

            for tns in tens:
                if tns!=None:
                    O_L4=ten.contract(O_L4,tns)
            legsNew=["a","b","c","d"]
            if layer==1:
                bnds=[ten.Bond("l0_"+str(leg1/2),2,ghost),ten.Bond("c",2,ghost)]
                O_q=ten.Tensor(s0,bnds)
                O_L4=ten.contract(O_L4,O_q)#fixes naming of this bond when it's reintroduced in the next layer
                for i in range(2):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)
            else:
                for i in range(4):
                    O_L4.changeBondName(legsNew[i],legs[i])
                    O_L4.setBondPrime(legsNew[i],ghost)

        else:
            print("Blocks of 6 starting on an odd site are not currently supported!")
            return None

        return O_L4


    def findleft(self,left,L):
        if L==8:
            return 0
        elif left%4==0:
            return ((left-4)/2)
        elif left%4==1:
            return ((left-1)/2)
        elif left%4==2:
            return ((left-2)/2)
        elif left%4==3:
            return ((left-3)/2)

    def findright(self,right,L):
        if L==8:
            return 3
        elif right%4==0:
            return ((right+2)/2)
        elif right%4==1:
            return ((right+1)/2)
        elif right%4==2:
            return ((right+4)/2)
        elif right%4==3:
            return ((right+3)/2)


    def MPS1Point(self,i,O1,L,MPS):
        Q=copy.deepcopy(MPS.getMPSn(0))
        if(i==0):
            bonds=[ten.Bond("spin"+str(0),2),ten.Bond("spin"+str(0),2,True)]
            A=ten.Tensor(O1,bonds)
            Q=ten.contract(Q,A)
        else:
            Q.setBondPrime("spin"+str(0),True)
        Q=ten.contract(Q,MPS.getMPSnDagger(0))

        for k in range(L-1):
            Q=ten.contract(Q,copy.deepcopy(MPS.getMPSn(k+1)))
            if(k+1==i):
                bonds=[ten.Bond("spin"+str(k+1),2),ten.Bond("spin"+str(k+1),2,True)]
                A=ten.Tensor(O1,bonds)
                Q=ten.contract(Q,A)
            else:
                Q.setBondPrime("spin"+str(k+1),True)
            Q=ten.contract(Q,MPS.getMPSnDagger(k+1))
        #Q.printBonds()
        return float(Q.data.real)

    def MPS2Point(self,i,j,O1,O2,L,MPS):
        Q=copy.deepcopy(MPS.getMPSn(0))
        if(i==0):
            bonds=[ten.Bond("spin"+str(0),2),ten.Bond("spin"+str(0),2,True)]
            A=ten.Tensor(O1,bonds)
            Q=ten.contract(Q,A)
        elif(j==0):
            bonds=[ten.Bond("spin"+str(0),2),ten.Bond("spin"+str(0),2,True)]
            A=ten.Tensor(O2,bonds)
            Q=ten.contract(Q,A)
        else:
            Q.setBondPrime("spin"+str(0),True)
        Q=ten.contract(Q,MPS.getMPSnDagger(0))

        for k in range(L-1):
            Q=ten.contract(Q,copy.deepcopy(MPS.getMPSn(k+1)))
            if(k+1==i):
                bonds=[ten.Bond("spin"+str(k+1),2),ten.Bond("spin"+str(k+1),2,True)]
                A=ten.Tensor(O1,bonds)
                Q=ten.contract(Q,A)
            elif(k+1==j):
                bonds=[ten.Bond("spin"+str(k+1),2),ten.Bond("spin"+str(k+1),2,True)]
                A=ten.Tensor(O2,bonds)
                Q=ten.contract(Q,A)
            else:
                Q.setBondPrime("spin"+str(k+1),True)
            Q=ten.contract(Q,MPS.getMPSnDagger(k+1))
        #Q.printBonds()
        return float(Q.data.real)


    def RenormEV(self,O,left,size,L0,pow,MPS):
        #left is the leftmost site in O
        #size is the number of sites on which O is defined; assumed to be 4 or 6
        #MPS should be of size L0
        #the system is renormalized pow times
        s=size
        l=left
        r=(left+s-1)
        Op=copy.deepcopy(O)
        for k in range(pow):
            if k>0:
                Op=Op.data
            if s==4:
                Op=self.RGStep4site(Op,l,L0*(2**(pow-k)),pow-k)
                l=np.max((self.findleft(l,L0*(2**(pow-k))),0))
                r=np.min((self.findright(r,L0*(2**(pow-k))),(L0*(2**(pow-k-1)))-1))
                s=r-l+1
            elif s==6:
                Op=self.RGStep6site(Op,l,L0*(2**(pow-k)),pow-k)
                l=np.max((self.findleft(l,L0*(2**(pow-k))),0))
                r=np.min((self.findright(r,L0*(2**(pow-k))),(L0*(2**(pow-k-1)))-1))
                s=r-l+1
            else:
                print("Error! s is not 4 or 6!")
        Q=copy.deepcopy(Op)
        for k in range(s):
            Q=ten.contract(Q,copy.deepcopy(MPS.getMPSn((l+k)%L0)))
        for k in range(s):
            Q=ten.contract(Q,MPS.getMPSnDagger((l+k)%L0))
        for k in range(L0-s):
            Q=ten.contract(Q,copy.deepcopy(MPS.getMPSn((l+s+k)%L0)))
            Q.setBondPrime("spin"+str((l+s+k)%L0),True)
            Q=ten.contract(Q,MPS.getMPSnDagger((l+s+k)%L0))
        #Q.printBonds()
        #return Q
        #return float(Q.data.real)
        return complex(Q.data)  # HACK np.complex_(Q.data)

    def DRenormEV(self,O,layer,leg1,left,size,L0,pow,MPS,ghost=False):
        #Computes the derivative of RenormEV wrt U^dagger for U in the top superlayer of the circuit
        #And with U specified by layer layer and first leg leg1
        #left is the leftmost site in O
        #size is the number of sites on which O is defined; assumed to be 4 or 6
        #MPS should be of size L0
        #the system is renormalized pow times
        s=size
        l=left
        r=(left+s-1)
        Op=copy.deepcopy(O)
        for k in range(pow):
            if k==0:
                if s==4:
                    Op=self.DRGStep4site(Op,layer,leg1,l,L0*(2**(pow-k)),pow-k,ghost)
                    l=np.max((self.findleft(l,L0*(2**(pow-k))),0))
                    r=np.min((self.findright(r,L0*(2**(pow-k))),(L0*(2**(pow-k-1)))-1))
                    s=r-l+1
                elif s==6:
                    Op=self.DRGStep6site(Op,layer,leg1,l,L0*(2**(pow-k)),pow-k,ghost)
                    l=np.max((self.findleft(l,L0*(2**(pow-k))),0))
                    r=np.min((self.findright(r,L0*(2**(pow-k))),(L0*(2**(pow-k-1)))-1))
                    s=r-l+1
                else:
                    print("Error! s is not 4 or 6!")
            else:
                if s==4:
                    Op=self.RGStep4site(Op,l,L0*(2**(pow-k)),pow-k,OisTensor=True)
                    l=np.max((self.findleft(l,L0*(2**(pow-k))),0))
                    r=np.min((self.findright(r,L0*(2**(pow-k))),(L0*(2**(pow-k-1)))-1))
                    s=r-l+1
                elif s==6:
                    Op=self.RGStep6site(Op,l,L0*(2**(pow-k)),pow-k,OisTensor=True)
                    l=np.max((self.findleft(l,L0*(2**(pow-k))),0))
                    r=np.min((self.findright(r,L0*(2**(pow-k))),(L0*(2**(pow-k-1)))-1))
                    s=r-l+1
                else:
                    print("Error! s is not 4 or 6!")
            aa=np.max((l,0))
            bb=np.min((r,L0*(2**(pow-k-1))-1))
            for i in range(aa,bb+1):
                Op.changeBondName("l4_"+str(i),"l0_"+str(i))
        aa=np.max((l,0))
        bb=np.min((r,L0-1))
        for i in range(aa,bb+1):
            Op.changeBondName("spin"+str(i),"l4_"+str(i))
        Q=copy.deepcopy(Op)
        for k in range(s):
            Q=ten.contract(Q,copy.deepcopy(MPS.getMPSn((l+k)%L0)))
        for k in range(s):
            Q=ten.contract(Q,MPS.getMPSnDagger((l+k)%L0))
        for k in range(L0-s):
            Q=ten.contract(Q,copy.deepcopy(MPS.getMPSn((l+s+k)%L0)))
            Q.setBondPrime("spin"+str((l+s+k)%L0),True)
            Q=ten.contract(Q,MPS.getMPSnDagger((l+s+k)%L0))
        #Q.printBonds()
        ten1=ten.Tensor([1.],[ten.Bond("mps_-1_0",1,False)])
        ten2=ten.Tensor([1.],[ten.Bond("mps_-1_0",1,True)])
        ten3=ten.Tensor([1.],[ten.Bond("mps_"+str(L0-1)+"_"+str(L0),1,False)])
        ten4=ten.Tensor([1.],[ten.Bond("mps_"+str(L0-1)+"_"+str(L0),1,True)])
        Q=ten.contract(ten.contract(ten.contract(ten.contract(Q,ten1),ten2),ten3),ten4)
        if layer==1:
            Q.reorderBonds([ten.Bond("a",2,ghost),ten.Bond("b",2,ghost),ten.Bond("c",2,ghost)])
        else:
            Q.reorderBonds([ten.Bond("a",2,ghost),ten.Bond("b",2,ghost),ten.Bond("c",2,ghost),ten.Bond("d",2,ghost)])
        return Q.data

    def RG2PointLimited(self,O1,O2,n):
        """
        O1 and O2 are assumed to be local 4 site 2x2x2x2x2x2x2x2 tensors separated by
        4*(2^n-1) sites on each side in a 2^(n+3) particle system with PBCs.
        """
        Temp1=copy.deepcopy(O1)
        Temp2=copy.deepcopy(O2)
        #renormalize O1 and O2 n times until the blocks are adjacent
        for i in range(n):
            Temp1=self.RGStep(Temp1).data
            Temp2=self.RGStep(Temp2).data


        bondsA_layer_1=[ten.Bond("l4_"+str(i%4),2,False if i<4 else True) for i in range(8)]
        A_ten=ten.Tensor(Temp1,bondsA_layer_1)
        bondsB_layer_1=[ten.Bond("l4_"+str(4+i%4),2,False if i<4 else True) for i in range(8)]
        B_ten=ten.Tensor(Temp2,bondsB_layer_1)

        ### Layer 1
        # upgrade to level 1
        #TO=self.buildTurnOff(4,0,2,0,2)
        #TO_p=self.buildTurnOff(4,0,2,0,2,True)
        TO=self.buildTurnOff(4,2,0,2,0)
        TO_p=self.buildTurnOff(4,2,0,2,0,True)
        A_L1=ten.contract(A_ten,TO)
        A_L1=ten.contract(A_L1,TO_p)

        A_L1.changeBondName("l3_1","l4_1")
        A_L1.changeBondName("l3_3","l4_3")
        A_L1.changeBondName("l3_1","l4_1")
        A_L1.changeBondName("l3_3","l4_3")

        #TO=self.buildTurnOff(4,4,6,4,6)
        #TO_p=self.buildTurnOff(4,4,6,4,6,True)
        TO=self.buildTurnOff(4,6,4,6,4)
        TO_p=self.buildTurnOff(4,6,4,6,4,True)
        B_L1=ten.contract(B_ten,TO)
        B_L1=ten.contract(B_L1,TO_p)

        B_L1.changeBondName("l3_5","l4_5")
        B_L1.changeBondName("l3_7","l4_7")
        B_L1.changeBondName("l3_5","l4_5")
        B_L1.changeBondName("l3_7","l4_7")

        ### Layer 2
        # turn "on" tensors
        turnOnTen=[]
        for i in range(4):
            ll=2*i+1
            level=3
            #c_ten=self.buildTurnOn(level,ll,(ll+1)%8,ll,(ll+1)%8,False)
            c_ten=self.buildTurnOn(level,(ll+1)%8,ll,(ll+1)%8,ll,False)
            ############Stuff made irrelevant by PBCs; keeping this here for safety of future edits
            # "self" contract tensors - make unprime
            #if i==0:
            #c_ten.setBondPrime("l3_0",True)
            #if i==2:
            #c_ten.setBondPrime("l3_5",True)
            turnOnTen.append(c_ten)
            #turnOnTen.append(self.buildTurnOn(level,ll,(ll+1)%8,ll,(ll+1)%8,True))
            turnOnTen.append(self.buildTurnOn(level,(ll+1)%8,ll,(ll+1)%8,ll,True))
        A_L2=A_L1
        B_L2=B_L1

        A_L2=ten.contract(A_L2,turnOnTen[0])
        A_L2=ten.contract(A_L2,turnOnTen[1])
        B_L2=ten.contract(B_L2,turnOnTen[4])
        B_L2=ten.contract(B_L2,turnOnTen[5])
        C_L2=ten.contract(A_L2,turnOnTen[2])
        C_L2=ten.contract(C_L2,B_L2)
        C_L2=ten.contract(C_L2,turnOnTen[3])
        C_L2=ten.contract(C_L2,turnOnTen[6])
        C_L2=ten.contract(C_L2,turnOnTen[7])

        ### Layer 3
        # turn "off" tensors
        turnOffTen=[]
        for i in range(2):
            ll=4*i
            level=2
            c_ten=self.buildTurnOff(level,(ll-2)%8,ll,(ll-2)%8,ll,False)
            ########################ignore due to PBCs
            # "self" contract tensors - make unprime
            #if i==0:
            #c_ten.setBondPrime("l2_0",True)
            turnOffTen.append(c_ten)
            turnOffTen.append(self.buildTurnOff(level,(ll-2)%8,ll,(ll-2)%8,ll,True))
        C_L3=C_L2
        for tenoff in turnOffTen:
            C_L3=ten.contract(C_L3,tenoff)
        # upgrade to level 3
        C_L3.changeBondName("l1_1","l2_1")
        C_L3.changeBondName("l1_3","l2_3")
        C_L3.changeBondName("l1_5","l2_5")
        C_L3.changeBondName("l1_7","l2_7")

        ### Layer 4
        turnOnIsoTen=[]
        for i in range(4):
            ll=i
            level=1
            c_ten=self.buildTurnOnIso(level,ll,2*ll,2*ll+1,False)
            ########################ignore due to PBCs
            # "self" contract tensors - make unprime
            #if i==3:
            #   c_ten.setBondPrime("l1_7",True)
            turnOnIsoTen.append(c_ten)
            turnOnIsoTen.append(self.buildTurnOnIso(level,ll,2*ll,2*ll+1,True))
        C_L4=C_L3
        for tenonIso in turnOnIsoTen:
            C_L4=ten.contract(C_L4,tenonIso)

        #C_L4.changeBondName("spin0","l0_0")
        #C_L4.changeBondName("spin1","l0_1")
        #C_L4.changeBondName("spin2","l0_2")
        #C_L4.changeBondName("spin3","l0_3")
        C_L4.changeBondName("spin1","l0_0")
        C_L4.changeBondName("spin2","l0_1")
        C_L4.changeBondName("spin3","l0_2")
        C_L4.changeBondName("spin0","l0_3")

        C_L4.reorderBonds([ten.Bond("spin"+str(i%4),2,False if i<4 else True) for i in range(8)])
        return C_L4




    def RG2Point(self,O1,O2,n,m):
        """
        O1 and O2 are assumed to be local 4 site 2x2x2x2x2x2x2x2 tensors separated by
        4*(2^n-1) sites on each side in a 2^(n+3) particle system with PBCs.
        """
        Temp1=copy.deepcopy(O1)
        Temp2=copy.deepcopy(O2)
        #renormalize O1 and O2 n times until the blocks are adjacent
        for i in range(n):
            Temp1=self.RGStep(Temp1).data
            Temp2=self.RGStep(Temp2).data

        if m==0:
            bondsA_layer_1=[ten.Bond("l4_"+str(i%4),2,False if i<4 else True) for i in range(8)]
            A_ten=ten.Tensor(Temp1,bondsA_layer_1)
            bondsB_layer_1=[ten.Bond("l4_"+str(4+i%4),2,False if i<4 else True) for i in range(8)]
            B_ten=ten.Tensor(Temp2,bondsB_layer_1)

            ### Layer 1
            # upgrade to level 1
            #TO=self.buildTurnOff(4,0,2,0,2)
            #TO_p=self.buildTurnOff(4,0,2,0,2,True)
            TO=self.buildTurnOff(4,2,0,2,0)
            TO_p=self.buildTurnOff(4,2,0,2,0,True)
            A_L1=ten.contract(A_ten,TO)
            A_L1=ten.contract(A_L1,TO_p)

            A_L1.changeBondName("l3_1","l4_1")
            A_L1.changeBondName("l3_3","l4_3")

            #TO=self.buildTurnOff(4,4,6,4,6)
            #TO_p=self.buildTurnOff(4,4,6,4,6,True)
            TO=self.buildTurnOff(4,6,4,6,4)
            TO_p=self.buildTurnOff(4,6,4,6,4,True)
            B_L1=ten.contract(B_ten,TO)
            B_L1=ten.contract(B_L1,TO_p)

            B_L1.changeBondName("l3_5","l4_5")
            B_L1.changeBondName("l3_7","l4_7")

            ### Layer 2
            # turn "on" tensors
            turnOnTen=[]
            for i in range(4):
                ll=2*i+1
                level=3
                #c_ten=self.buildTurnOn(level,ll,(ll+1)%8,ll,(ll+1)%8,False)
                c_ten=self.buildTurnOn(level,(ll+1)%8,ll,(ll+1)%8,ll,False)
                ############Stuff made irrelevant by PBCs; keeping this here for safety of future edits
                # "self" contract tensors - make unprime
                #if i==0:
                #c_ten.setBondPrime("l3_0",True)
                #if i==2:
                #c_ten.setBondPrime("l3_5",True)
                turnOnTen.append(c_ten)
                #turnOnTen.append(self.buildTurnOn(level,ll,(ll+1)%8,ll,(ll+1)%8,True))
                turnOnTen.append(self.buildTurnOn(level,(ll+1)%8,ll,(ll+1)%8,ll,True))
            A_L2=A_L1
            B_L2=B_L1

            A_L2=ten.contract(A_L2,turnOnTen[0])
            A_L2=ten.contract(A_L2,turnOnTen[1])
            B_L2=ten.contract(B_L2,turnOnTen[4])
            B_L2=ten.contract(B_L2,turnOnTen[5])
            C_L2=ten.contract(A_L2,turnOnTen[2])
            C_L2=ten.contract(C_L2,B_L2)
            C_L2=ten.contract(C_L2,turnOnTen[3])
            C_L2=ten.contract(C_L2,turnOnTen[6])
            C_L2=ten.contract(C_L2,turnOnTen[7])

            ### Layer 3
            # turn "off" tensors
            turnOffTen=[]
            for i in range(2):
                ll=4*i
                level=2
                c_ten=self.buildTurnOff(level,(ll-2)%8,ll,(ll-2)%8,ll,False)
                ########################ignore due to PBCs
                # "self" contract tensors - make unprime
                #if i==0:
                #c_ten.setBondPrime("l2_0",True)
                turnOffTen.append(c_ten)
                turnOffTen.append(self.buildTurnOff(level,(ll-2)%8,ll,(ll-2)%8,ll,True))
            C_L3=C_L2
            for tenoff in turnOffTen:
                C_L3=ten.contract(C_L3,tenoff)
            # upgrade to level 3
            C_L3.changeBondName("l1_1","l2_1")
            C_L3.changeBondName("l1_3","l2_3")
            C_L3.changeBondName("l1_5","l2_5")
            C_L3.changeBondName("l1_7","l2_7")

            ### Layer 4
            turnOnIsoTen=[]
            for i in range(4):
                ll=i
                level=1
                c_ten=self.buildTurnOnIso(level,ll,2*ll,2*ll+1,False)
                ########################ignore due to PBCs
                # "self" contract tensors - make unprime
                #if i==3:
                #   c_ten.setBondPrime("l1_7",True)
                turnOnIsoTen.append(c_ten)
                turnOnIsoTen.append(self.buildTurnOnIso(level,ll,2*ll,2*ll+1,True))
            C_L4=C_L3
            for tenonIso in turnOnIsoTen:
                C_L4=ten.contract(C_L4,tenonIso)

        else:
            bondsA_layer_1=[ten.Bond("l4_"+str(i%4),2,False if i<4 else True) for i in range(8)]
            A_ten=ten.Tensor(Temp1,bondsA_layer_1)
            bondsB_layer_1=[ten.Bond("l4_"+str(4+i%4),2,False if i<4 else True) for i in range(8)]
            B_ten=ten.Tensor(Temp2,bondsB_layer_1)

            ### Layer 1
            # upgrade to level 1
            #TO=self.buildTurnOff(4,0,2,0,2)
            #TO_p=self.buildTurnOff(4,0,2,0,2,True)
            TO=self.buildTurnOff(4,2,0,2,0)
            TO_p=self.buildTurnOff(4,2,0,2,0,True)
            A_L1=ten.contract(A_ten,TO)
            A_L1=ten.contract(A_L1,TO_p)

            A_L1.changeBondName("l3_1","l4_1")
            A_L1.changeBondName("l3_3","l4_3")

            #TO=self.buildTurnOff(4,4,6,4,6)
            #TO_p=self.buildTurnOff(4,4,6,4,6,True)
            TO=self.buildTurnOff(4,6,4,6,4)
            TO_p=self.buildTurnOff(4,6,4,6,4,True)
            B_L1=ten.contract(B_ten,TO)
            B_L1=ten.contract(B_L1,TO_p)

            B_L1.changeBondName("l3_5","l4_5")
            B_L1.changeBondName("l3_7","l4_7")

            ### Layer 2
            # turn "on" tensors
            turnOnTen=[]
            for i in range(5):
                ll=2*i-1
                level=3
                #c_ten=self.buildTurnOn(level,ll,(ll+1)%8,ll,(ll+1)%8,False)
                c_ten=self.buildTurnOn(level,(ll+1),ll,(ll+1),ll,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l3_-1",True)
                if i==4:
                    c_ten.setBondPrime("l3_8",True)
                turnOnTen.append(c_ten)
                #turnOnTen.append(self.buildTurnOn(level,ll,(ll+1)%8,ll,(ll+1)%8,True))
                turnOnTen.append(self.buildTurnOn(level,(ll+1),ll,(ll+1),ll,True))
            A_L2=A_L1
            B_L2=B_L1

            A_L2=ten.contract(A_L2,turnOnTen[0])
            A_L2=ten.contract(A_L2,turnOnTen[1])
            A_L2=ten.contract(A_L2,turnOnTen[2])
            A_L2=ten.contract(A_L2,turnOnTen[3])
            B_L2=ten.contract(B_L2,turnOnTen[6])
            B_L2=ten.contract(B_L2,turnOnTen[7])
            B_L2=ten.contract(B_L2,turnOnTen[8])
            B_L2=ten.contract(B_L2,turnOnTen[9])
            C_L2=ten.contract(A_L2,turnOnTen[4])
            C_L2=ten.contract(C_L2,B_L2)
            C_L2=ten.contract(C_L2,turnOnTen[5])

            ### Layer 3
            # turn "off" tensors
            turnOffTen=[]
            for i in range(3):
                ll=4*i
                level=2
                c_ten=self.buildTurnOff(level,(ll-2),ll,(ll-2),ll,False)
                # "self" contract tensors - make unprime
                if i==0:
                    c_ten.setBondPrime("l2_-2",True)
                turnOffTen.append(c_ten)
                turnOffTen.append(self.buildTurnOff(level,(ll-2),ll,(ll-2),ll,True))
            C_L3=C_L2
            for tenoff in turnOffTen:
                C_L3=ten.contract(C_L3,tenoff)
                # upgrade to level 3
                C_L3.changeBondName("l1_-1","l2_-1")
                C_L3.changeBondName("l1_1","l2_1")
                C_L3.changeBondName("l1_3","l2_3")
                C_L3.changeBondName("l1_5","l2_5")
                C_L3.changeBondName("l1_7","l2_7")

            ### Layer 4
            turnOnIsoTen=[]
            for i in range(6):
                ll=i-1
                level=1
                c_ten=self.buildTurnOnIso(level,ll,2*ll,2*ll+1,False)
                # "self" contract tensors - make unprime
                if i==5:
                   c_ten.setBondPrime("l1_9",True)
                turnOnIsoTen.append(c_ten)
                turnOnIsoTen.append(self.buildTurnOnIso(level,ll,2*ll,2*ll+1,True))
            C_L4=C_L3
            for tenonIso in turnOnIsoTen:
                C_L4=ten.contract(C_L4,tenonIso)

            C_L4.changeBondName("l4_0","l0_-1")
            C_L4.changeBondName("l4_1","l0_0")
            C_L4.changeBondName("l4_2","l0_1")
            C_L4.changeBondName("l4_3","l0_2")
            C_L4.changeBondName("l4_4","l0_3")
            C_L4.changeBondName("l4_5","l0_4")

            for k in range(m-2):
                print("hi")



        #C_L4.changeBondName("spin0","l0_0")
        #C_L4.changeBondName("spin1","l0_1")
        #C_L4.changeBondName("spin2","l0_2")
        #C_L4.changeBondName("spin3","l0_3")
        C_L4.changeBondName("spin1","l0_0")
        C_L4.changeBondName("spin2","l0_1")
        C_L4.changeBondName("spin3","l0_2")
        C_L4.changeBondName("spin0","l0_3")

        C_L4.reorderBonds([ten.Bond("spin"+str(i%4),2,False if i<4 else True) for i in range(8)])
        return C_L4


    def genqs(self,n):
        return np.array([2*np.pi*(.5+i)/n for i in range(n/2)])

    def genws(self,h,J,qs):
        return np.array([np.sqrt(h**2+2*J*h*np.cos(q)+J**2) for q in qs])

    def genus(self,h,J,qs,ws,n):
        return [J*np.sin(qs[i])/np.sqrt(2*ws[i]*(ws[i]+h+J*np.cos(qs[i]))) for i in range(n/2)]

    def genvs(self,h,J,qs,ws,n):
        return [(ws[i]+h+J*np.cos(qs[i]))/np.sqrt(2*ws[i]*(ws[i]+h+J*np.cos(qs[i]))) for i in range(n/2)]

    def Gr(self,qs,ws,us,vs,n,r):
        return (1./n)*np.sum([2*np.cos(qs[i]*r)*(1-2*us[i]**2)-4*us[i]*vs[i]*np.sin(qs[i]*r) for i in range(n/2)])

    def XExactTFIM(self,qs,ws,us,vs,n):
        return self.Gr(qs,ws,us,vs,n,0)

    def XXExactTFIM(self,qs,ws,us,vs,n,r):
        return -self.Gr(qs,ws,us,vs,n,r)*self.Gr(qs,ws,us,vs,n,-r)+self.Gr(qs,ws,us,vs,n,0)**2

    def YYExactTFIM(self,qs,ws,us,vs,n,r):
        k=[]
        for i in range(-r,r-1):
            k.append(self.Gr(qs,ws,us,vs,n,-i)*np.ones(r-np.abs(i+1)))
        offset=range(-r+1,r)
        G=diags(k,offset).toarray()
        return LA.det(G)

    def ZZExactTFIM(self,qs,ws,us,vs,n,r):
        k=[]
        for i in range(-r,r-1):
            k.append(self.Gr(qs,ws,us,vs,n,i)*np.ones(r-np.abs(i+1)))
        offset=range(-r+1,r)
        G=diags(k,offset).toarray()
        return LA.det(G)
