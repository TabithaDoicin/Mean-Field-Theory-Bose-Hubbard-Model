# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 23:18:43 2020

@author: anton
"""

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_general # bosonic Hilbert space
from scipy.optimize import minimize_scalar #for finding psi
import numpy as np #generic numpy stuff
import matplotlib.pyplot as plt

##Model parameters##
Lx = 1
Ly = 1
N_2d = Lx*Ly
#
mu = 0 #chemical potential
U = 1 #on-site potential
z = 4 #num of nearest neighbours
t = 0 #hopping constant
m = 10 #maximum particles per site
#
##basis##
basis_2d = boson_basis_general(N_2d, sps = m+1)
#######################
##
xsteps = 50
x_min = 0
x_max = 0.05
##
ysteps = 50
y_min = 0
y_max = 3
##
ti = np.linspace(x_min,x_max,xsteps)
mui = np.linspace(y_min,y_max,ysteps)
psimat = np.zeros((ysteps,xsteps))

def construct_h(t,mu,psi):
    potential=[[-mu-U/2,i] for i in range(N_2d)]
    interaction=[[U/2,i,i] for i in range(N_2d)]
    hopping_1 = [[-z*t*psi.conjugate(),i] for i in range(N_2d)]
    hopping_2 = [[-z*t*psi,i] for i in range(N_2d)]
    extra = [[z*t*np.abs(psi)**2,i] for i in range(N_2d)]
    static=[["n",potential],["nn",interaction],["-",hopping_1],["+",hopping_2],["I",extra]]
    dynamic = []
    # build hamiltonian
    H=hamiltonian(static,dynamic,basis = basis_2d, dtype = np.complex128,check_symm=False,check_herm=False,check_pcon=False)
    return H.todense()

def gs_energy(H):
    E_GS=np.linalg.eigvalsh(H)
    #print(E_GS[0])
    return E_GS[0]

def minimize_gs(t,mu):
    def gs_min_energy(psi):
        return gs_energy(construct_h(t,mu,psi))
    return minimize_scalar(gs_min_energy).x

for y in range(xsteps):
    t = ti[y]
    print('progress: ' + str(y/xsteps * 100) + '%')
    for x in range(ysteps):
        mu = mui[x]
        psimat[x,y]=np.abs(minimize_gs(t,mu))

fig, ax = plt.subplots()
im = ax.imshow(psimat, origin = 'lower', extent=(x_min,x_max,y_min,y_max), aspect = 'auto')#, cmap='Blues')
cb = plt.colorbar(im)
cb.set_label(r'$\Psi$',size=15)
#ax.set_title('Mean Field Bose-Hubbard Model',size=20)
ax.set_xlabel(r'$t/U$',fontsize=15)
ax.set_ylabel(r'$\mu/U$',fontsize=15)
