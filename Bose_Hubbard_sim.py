# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:01:01 2020

@author: anton
"""

#!usr/bin/python

from quspin.operators import hamiltonian as h_construct# Hamiltonians and operators
from quspin.basis import boson_basis_general # bosonic Hilbert space
from scipy.optimize import minimize_scalar #for automatic minimization
import numpy as np #generic numpy stuff
import matplotlib.pyplot as plt#plotting stuff
import time#timer for timing

class System_1D:   #1d only, creates system object
    
    def __init__(self, U, z, max_p, L): #initialises object parameters
        self.U = U
        self.z = z - 2 #effective z is different to 'number of nearest neighbours' z, depends on positioning
        self.L = L
        self.basis = boson_basis_general(self.L, sps = max_p+1) #basis generation
        self.local_z = [self.z for i in range(self.L)] #local_z list
        self.local_z[0] = self.local_z[0] + 1
        self.local_z[self.L-1] = self.local_z[self.L-1] + 1#differences in the z for the ends of the chain
    
    def construct_h(self, t, mu, psi): #function for constructing hamiltonian
        potential = [[-mu-self.U/2, i] for i in range(self.L)]
        interaction = [[self.U/2, i, i] for i in range(self.L)]
        hopping_1 = [[-self.local_z[i]*t*psi.conjugate(), i] for i in range(self.L)]
        hopping_2 = [[-self.local_z[i]*t*psi, i] for i in range(self.L)]
        extra = [[self.local_z[i]*t*np.abs(psi)**2, i] for i in range(self.L)]
        static = [["n", potential], ["nn", interaction], ["-", hopping_1], ["+", hopping_2], ["I", extra]]
        if self.L>1: #exact hopping terms that only exist for L>1
            hop = [[-t,i,(i+1)] for i in range(self.L-1)] 
            static.append(["+-",hop])
            static.append(["-+",hop])
        dynamic = []
        H = h_construct(static, dynamic, basis = self.basis, dtype = np.complex128,
                     check_symm = False, check_herm = False, check_pcon = False) #construction of H
        return H.todense() #needed to change data type to solve eigensystem properly

    def gs_energy(self, H):
        E_GS=np.linalg.eigvalsh(H) #part of minimisation process
        return E_GS[0] #ground state energy

    def minimize_gs(self, t, mu): #another part of minimisation process
        def gs_min_energy(psi):
            return self.gs_energy(self.construct_h(t, mu, psi))
        return minimize_scalar(gs_min_energy).x #americans spell minimize with a z, odd..
    
def simulate(system, xmin, xmax, xsteps, ymin, ymax, ysteps): #main function to be called when modelling a system
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
    return psimat #returns matrix of psi's as function solution
    
#This next bit can be in a seperate file, by means of importing the functional part of the file above. 
    #this next bit is just graphs and using the above code to generate things that look cool

U = 1
z = 4
max_p = 2
L=4
xmin = 0.035
xmax = 0.06
xsteps = 85
ymin = 0.42
ymax = 0.42
ysteps = 1

s1 = System_1D(U, z, max_p, 1) #initialise base system parameters/ z = number of nearest neighbours
solution1 = simulate(s1, xmin, xmax, xsteps, 
         ymin, ymax, ysteps) #calling simulate function, does calculations
s2 = System_1D(U, z, max_p, 2) #initialise base system parameters/ z = number of nearest neighbours
solution2 = simulate(s2, xmin, xmax, xsteps, ymin, ymax, ysteps) #calling simulate function, does calculations
s3 = System_1D(U, z, max_p, 4) #initialise base system parameters/ z = number of nearest neighbours
solution3 = simulate(s3, xmin, xmax, xsteps, ymin, ymax, ysteps) #calling simulate function, does calculations
s4 = System_1D(U, z, max_p, 6) #initialise base system parameters/ z = number of nearest neighbours
solution4 = simulate(s4, xmin, xmax, xsteps, ymin, ymax, ysteps) #calling simulate function, does calculations

fig, ax = plt.subplots()
im = ax.imshow(solution1, origin = 'lower', extent=(xmin,xmax,ymin,ymax), aspect = 'auto')
cb = plt.colorbar(im)
cb.set_label(r'$\Psi$', size=15)
ax.set_xlabel(r'$t/U$', fontsize=15)
ax.set_ylabel(r'$\mu/U$', fontsize=15)

fig1,ax1 = plt.subplots()
ti = np.linspace(xmin,xmax,xsteps)
ax1.scatter(ti,solution1[int(round(0)),:], marker = 'x', c = 'r', s = 2)
ax1.scatter(ti,solution2[int(round(0)),:], marker = 'x', c = 'b', s=2)
ax1.scatter(ti,solution3[int(round(0)),:], marker = 'x', c='k', s=2)
ax1.scatter(ti,solution4[int(round(0)),:], marker = 'x', c='m', s=2)
ax1.set_xlabel(r'$t/U$', fontsize=15)
ax1.set_ylabel(r'$\Psi$', fontsize=15)
