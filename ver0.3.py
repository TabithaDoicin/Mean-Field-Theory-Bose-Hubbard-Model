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
import time
##timing
start_time = time.time()
##Model parameters##
#
L = 1 #length of 1d chain ##Dont change this##
U = 1 #on-site potential
z = 4 #num of nearest neighbours
m = 6 #maximum particles per site
#
##basis##
basis_2d = boson_basis_general(L, sps = m+1)
#######################
##x=t/U
xsteps = 10
x_min = 0
x_max = 0.05
##y=mu/U
ysteps = 10
y_min = 0
y_max = 2
##
ti = np.linspace(x_min,x_max,xsteps)
mui = np.linspace(y_min,y_max,ysteps)
psimat = np.zeros((ysteps,xsteps))

def construct_h(t,mu,psi):
    potential=[[-mu-U/2,i] for i in range(L)]
    interaction=[[U/2,i,i] for i in range(L)]
    hopping_1 = [[-z*t*psi.conjugate(),i] for i in range(L)]
    hopping_2 = [[-z*t*psi,i] for i in range(L)]
    extra = [[z*t*np.abs(psi)**2,i] for i in range(L)]
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
    print('progress: ' + str(round(y/xsteps * 100,4)) + '%')
    for x in range(ysteps):
        mu = mui[x]
        psimat[x,y]=np.abs(minimize_gs(t,mu))

elapsed_time = time.time() - start_time
print('Time elapsed: ' + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
fig, ax = plt.subplots()
im = ax.imshow(psimat, origin = 'lower', extent=(x_min,x_max,y_min,y_max), aspect = 'auto')#, cmap='Blues')
cb = plt.colorbar(im)
cb.set_label(r'$\Psi$',size=15)
#ax.set_title('Mean Field Bose-Hubbard Model',size=20)
ax.set_xlabel(r'$t/U$',fontsize=15)
ax.set_ylabel(r'$\mu/U$',fontsize=15)