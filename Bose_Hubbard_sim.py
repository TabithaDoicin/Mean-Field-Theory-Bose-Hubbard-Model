# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:01:01 2020

@author: anton
"""

#!usr/bin/python

from quspin.operators import hamiltonian as h_construct # Hamiltonians and operators
from quspin.basis import boson_basis_general # bosonic Hilbert space
from scipy.optimize import minimize_scalar  # for finding psi
import numpy as np # generic numpy stuff
import matplotlib.pyplot as plt # plotting stuff
import time # timer for timing
import multiprocessing as mp # multiprocessing and stuff

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
 
def multiboi(xmt_min, xmt_max, system): #xmt_min, xmt_max as matrix elements
    for x in range(xmt_min, xmt_max):
        t = ti[x]
        print('progress: ' + str(round(x/(xmt_max-xmt_min) * 100, 4)) + '%')
        for y in range(y_num):
            mu = mui[y]
            psimat[y, x]=np.abs(system.minimize_gs(t, mu))

def simulate(number_of_processes, system, xmin, xmax, xsteps, ymin, ymax, ysteps):
    global ti, mui, psimat, y_num
    ti = np.linspace(xmin, xmax, xsteps)
    mui = np.linspace(ymin, ymax, ysteps)
    psimat = np.zeros((ysteps, xsteps))
    y_num = ysteps # funny pythonness requires this
    start_time = time.time()
    bpp = len(ti)//number_of_processes # base 'x' line calcs per process
    ap = len(ti)%number_of_processes # added 'x' lines
    print(bpp,ap)
    xmt_min_array = []
    xmt_max_array = []
    processes = []
    
    for i in range(number_of_processes):
        if ap >0:
            ap = ap-1
            extra = 1
        else:
            extra = 0    
        if len(xmt_min_array) == 0:
            xmt_min_array.append(0)
            xmt_max_array.append((bpp-1) + extra)
        else:
            xmt_min_array.append(xmt_max_array[i-1]+1)
            xmt_max_array.append(xmt_min_array[i] + (bpp-1) + extra)
    if __name__ == '__main__':        
        for i in range(number_of_processes):
            p = mp.Process(target = multiboi, args = (xmt_min_array[i], xmt_max_array[i], system))
            print('boop2')
            p.start()
            print('boop1')
            processes.append(p)
        for process in processes:
            print('boop')
            process.join()
        
    elapsed_time = time.time() - start_time
    fig, ax = plt.subplots()
    im = ax.imshow(psimat, origin = 'lower', extent=(xmin,xmax,ymin,ymax), aspect = 'auto')
    cb = plt.colorbar(im)
    cb.set_label(r'$\Psi$', size=15)
    ax.set_xlabel(r'$t/U$', fontsize=15)
    ax.set_ylabel(r'$\mu/U$', fontsize=15)
    print('Time elapsed: ' + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    
###USAGE###x=t/U##y=mu/U##
s = SPSystem(U = 1, z = 4, max_p = 6) #initialise base system parameters
simulate(number_of_processes = 4, system = s, xmin = 0, xmax = 0.05, xsteps = 50, 
         ymin = 0, ymax = 3, ysteps = 50) #calling simulate function, does calculations
