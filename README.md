# SSource
S Source tensor network code used in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.155152

Code is unfortunately not super well documented. 

RGMPS.py contains the actual s-source tensor network. SSourceExample.ipynb does some basic stuff with it, initializing an RGMPS object for the transverse field Ising model and then calculating the energy error both as a function of optimization step and as a function of magnetic field after 100 optimization steps for each field value.

Several of the base files (e.g. tensor.py) were originally written by Snir Gazit; others files involve taking ITensor matrix product states and converting them into our mps objects were written by Maxwell Block
