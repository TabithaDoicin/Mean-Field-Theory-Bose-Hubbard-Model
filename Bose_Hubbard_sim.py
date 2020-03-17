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

class SPSystem:
    
    def __init__(self, U, z, max_p):
        self.U = U
        self.z = z
        self.L = 1
        self.basis = boson_basis_general(self.L, sps = max_p+1)
    
    def construct_h(self, t, mu, psi):
        potential = [[-mu-self.U/2, i] for i in range(self.L)]
        interaction = [[self.U/2, i, i] for i in range(self.L)]
        hopping_1 = [[-self.z*t*psi.conjugate(), i] for i in range(self.L)]
        hopping_2 = [[-self.z*t*psi, i] for i in range(self.L)]
        extra = [[self.z*t*np.abs(psi)**2, i] for i in range(self.L)]
        static = [["n", potential], ["nn", interaction], ["-", hopping_1], ["+", hopping_2], ["I", extra]]
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
    psimat = np.zeros((ysteps, xsteps))
    for x in range(xsteps):
        t = ti[x]
        print('progress: ' + str(round(x/xsteps * 100, 4)) + '%')
        for y in range(ysteps):
            mu = mui[y]
            psimat[y, x]=np.abs(system.minimize_gs(t, mu))
    elapsed_time = time.time() - start_time
    print('Time elapsed: ' + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    fig, ax = plt.subplots()
    im = ax.imshow(psimat, origin = 'lower', extent=(xmin,xmax,ymin,ymax), aspect = 'auto')
    cb = plt.colorbar(im)
    cb.set_label(r'$\Psi$', size=15)
    ax.set_xlabel(r'$t/U$', fontsize=15)
    ax.set_ylabel(r'$\mu/U$', fontsize=15)
    
s = SPSystem(U = 1, z = 4, max_p = 10) #initialise base system parameters
simulate(system = s, xmin = 0, xmax = 0.05, xsteps = 50, 
         ymin = 0, ymax = 4, ysteps = 50) #calling simulate function, does calculations
