# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:01:01 2020

@author: anton
"""

#!usr/bin/python

from quspin.operators import hamiltonian as h_construct# Hamiltonians and operators
from quspin.basis import boson_basis_general # bosonic Hilbert space
from scipy.optimize import minimize_scalar #for finding psi
import numpy as np #generic numpy stuff
import matplotlib.pyplot as plt#plotting stuff
import time#timer for timing

class SPSystem:   #1d only 
    
    def __init__(self, U, z, max_p, L):
        self.U = U
        self.z = z
        self.L = L
        self.basis = boson_basis_general(self.L, sps = max_p+1)
        self.local_z = [self.z for i in range(self.L)]
        #self.local_z[0] = self.local_z[0] + 1
        #self.local_z[self.L-1] = self.local_z[self.L-1] + 1
    
    def construct_h(self, t, mu, psi):
        potential = [[-mu-self.U/2, i] for i in range(self.L)]
        interaction = [[self.U/2, i, i] for i in range(self.L)]
        hopping_1 = [[-self.local_z[i]*t*psi.conjugate(), i] for i in range(self.L)]
        hopping_2 = [[-self.local_z[i]*t*psi, i] for i in range(self.L)]
        extra = [[self.local_z[i]*t*np.abs(psi)**2, i] for i in range(self.L)]
        static = [["n", potential], ["nn", interaction], ["-", hopping_1], ["+", hopping_2], ["I", extra]]
        if self.L>=1:
            hop = [[-t,i,(i+1)] for i in range(self.L-1)]
            static.append(["+-",hop])
            static.append(["-+",hop])
        dynamic = []
        H = h_construct(static, dynamic, basis = self.basis, dtype = np.complex128,
                     check_symm = False, check_herm = False, check_pcon = False)
        return H.todense()

    def gs_energy(self, H):
        E_GS=np.linalg.eigvalsh(H)
        return E_GS[0]

    def minimize_gs(self, t, mu):
        def gs_min_energy(psi):
            return self.gs_energy(self.construct_h(t, mu, psi))
        return minimize_scalar(gs_min_energy).x
    
def simulate(system, xmin, xmax, xsteps, ymin, ymax, ysteps):
    start_time = time.time()
    ti = np.linspace(xmin, xmax, xsteps)
    mui = np.linspace(ymin, ymax, ysteps)
    global psimat
    psimat = np.zeros((ysteps, xsteps))
    for x in range(xsteps):
        t = ti[x]
        print('progress: ' + str(round(x/xsteps * 100, 4)) + '%')
        for y in range(ysteps):
            mu = mui[y]
            psimat[y, x]=np.abs(system.minimize_gs(t, mu))
    elapsed_time = time.time() - start_time
    print('Time elapsed: ' + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


U = 1
z = 4
max_p = 2
L = 1
xmin = 0.01
xmax = 0.06
xsteps = 20
ymin = 0
ymax = 1
ysteps = 20

s = SPSystem(U, z, max_p, L) #initialise base system parameters/ z = number of nearest neighbours
simulate(s, xmin, xmax, xsteps, 
         ymin, ymax, ysteps) #calling simulate function, does calculations


fig, ax = plt.subplots()
im = ax.imshow(psimat, origin = 'lower', extent=(xmin,xmax,ymin,ymax), aspect = 'auto')
cb = plt.colorbar(im)
cb.set_label(r'$\Psi$', size=15)
ax.set_xlabel(r'$t/U$', fontsize=15)
ax.set_ylabel(r'$\mu/U$', fontsize=15)

fig1,ax1 = plt.subplots()
ti = np.linspace(xmin,xmax,xsteps)
ax1.scatter(ti,psimat[int(round(ysteps/2)),:])
ax1.scatter(ti,psimat[int(round(ysteps/4)),:])
ax1.scatter(ti,psimat[int(round(3*ysteps/4)),:])
