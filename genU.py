# -*- coding: utf-8 -*-
"""
Created on Sat May 28 19:22:31 2016

@author: snirgaz
"""
import numpy as np
import linmax
#import odespy
import math
from scipy.integrate import ode

sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0],[0,-1]])
s0=np.array([[1,0],[0,1]])
basis=[np.kron(sx,sx),np.kron(sx,sy),np.kron(sx,sz),np.kron(sx,s0),
       np.kron(sy,sx),np.kron(sy,sy),np.kron(sy,sz),np.kron(sy,s0),
       np.kron(sz,sx),np.kron(sz,sy),np.kron(sz,sz),np.kron(sz,s0),
       np.kron(s0,sx),np.kron(s0,sy),np.kron(s0,sz)]

def H(J,hLx,hRx,hLz,hRZ):
    """ Generates the two site Hamitonian H=J sz1 sz2 + hL sx1 + hR sx2
    Parameters
    ----------
    J : float
        Value of J
    hL : float
        Value fo hL
    hR : float
        Value fo hR
    Returns
    -------
    numpy.array
        Returns the Hamitonian 4x4 matrix

    """
    return linmax.makearray(-J*np.kron(sz,sz)-hLx*np.kron(sx,s0)-hRx*np.kron(s0,sx)-hLz*np.kron(sz,s0)-hRZ*np.kron(s0,sz),'complex128')


def HTensor(J,hL,hR,hLz,hRZ):
    """ Generates the two site Hamitonian H=J sz1 sz2 + hL sx1 + hR sx2 as a 2x2x2x2 Tensor
    Parameters
    ----------
    J : float
        Value of J
    hL : float
        Value fo hL
    hR : float
        Value fo hR
    Returns
    -------
    numpy.array
        Returns the Hamitonian 2x2x2x2 tensor

    """
    return np.reshape(H(J,hL,hR,hLz,hRZ),(2,2,2,2))

def genU(J,h,left_right,M):
    """ Generates the two site unitary evolution correspoding to linear
        in time turning on the hamitonain H=J sz1 sz2 + hL sx1 + hR sx2
    Parameters
    ----------
    J : float
        Value of J
    h : float
        Value fo hL
    M : float
        number of slices in the trotter approximation
    left_right : int
        0 -- h increase on left site , 1-- h increase on right site
    Returns
    -------
    numpy.array
        Returns U  4x4 matrix

    """
    Id=np.kron(s0,s0)
    U=np.kron(s0,s0)
    eps=1./M
    hl=h if left_right else 0
    hr=h if not left_right else 0
    for i in range(M):
        htl=(hl*i)/M
        htr=(hr*i)/M
        Jt=(J*i)/M
        U=np.dot(Id-1j*eps*H(Jt,htl,htr),U)
    return U

def genURotSpinODE(J,h,T,left=False,backward=False):
    """ Generates the two site unitary evolution correspoding to linear
        in time turning on the hamitonain H=J sz1 sz2 + hL sx1 + hR sx2
        Using a ODE solution
    Parameters
    ----------
    J : float
        Value of J
    h : float
        strength of the magnetic field
    T : period overwhich the unitary evolves
    left : bool
        False -- rotate lest site , True - rotate right site
    backword : bool
        True -- propagate backward in time , False-- propagate forward in time
    Returns
    -------
    Tensor object
        Returns U  4x4 matrix

    """
    t0=0.0
    ## Set h left of right
    hl,hr = (h,h) if left else (h,h)
    def f_forward(t):
        ct=1.0#math.cos(0.5*math.pi*t/T)
        st=1.0#math.sin(0.5*math.pi*t/T)
        hLx=hl*st
        hRx=hr*st
        hLz=hl*ct
        hRZ=hr*ct
        Jt=((J*t)/T)
        return linmax.makearray(-1j*H(Jt,hLx,hRx,hLz,hRZ))
    def f_back(t):
        Jt=(J*(T-t)/T)
        return linmax.makearray(-1j*H(Jt,0,0,0,0))
    f=f_back if backward else f_forward
    U=np.zeros((4,4), dtype='complex128')
    for pos in range(4):
        y0=np.zeros(4,dtype='complex128')
        y0[pos]=1.0
        y0=linmax.makearray(y0,'complex128')
        times = linmax.maketimes(0.,T,stepsize=0.1)
        states = linmax.solveonce(f,times,y0)
        U[pos,:]=states[:,-1]
    return np.reshape(U,[2,2,2,2])




def genUODE(J,h_l,h_r,backward=False):
    """ Generates the two site unitary evolution correspoding to linear
        in time turning on the hamitonain H=J sz1 sz2 + hL sx1 + hR sx2
        Using a ODE solution
    Parameters
    ----------
    J : float
        Value of J
    h_l,h_r : float
        Value fo h left and h right
    M : float
        number of slices in the trotter approximation
    lforward_backword : int
        True -- propagate backward in time , False-- propagate forward in time
    Returns
    -------
    Tensor object
        Returns U  4x4 matrix

    """
    t0=0.0
    def f_forward(t, y, J,hl,hr):
        htl=(hl*(t-0.5)*2)
        htr=(hr*(t-0.5)*2)
        Jt=(J*t)
        return (1./1j)*np.dot(H(Jt,htl,htr),y)
    def f_backward(t, y, J,hl,hr):
        htl=(hl*(1.-t))
        htr=(hr*(1.-t))
        Jt=(J*(1.-t))
        return (1./1j)*np.dot(H(Jt,htl,htr),y)
    def jac(t,y):
        return np.zeros((4,4), dtype=np.complex)
    U=np.zeros((4,4), dtype=np.complex)
    f=f_forward if not backward else f_backward
    for pos in range(4):
        y0=np.zeros(4)
        y0[pos]=1.0
        r = ode(f,jac).set_integrator('zvode', method='bdf',atol=1E-10)
        r.set_initial_value(y0, t0).set_f_params(J,h_l,h_r)
        U[pos,:]=r.integrate(1.0)
    return np.reshape(U,(2,2,2,2))