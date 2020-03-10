# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:36:10 2020

@author: anton
"""

import numpy as np
import matplotlib.pyplot as plt

U = 1
psi_init = 1
z = 4
n_max = 8
tolerance = 0.0001

t_min = 0
t_max = 0.05
t_res = 500
mu_min = 0
mu_max = 2
mu_res = 500

b = np.diag(np.power(range(1, n_max), 0.5), 1)
bdag = np.transpose(b)
n = np.diag(range(0, n_max))

t_i = np.ndarray.tolist(np.linspace(t_min, t_max, t_res))
mu_i = np.ndarray.tolist(np.linspace(mu_min, mu_max, mu_res))

psimat = np.zeros((len(t_i),len(mu_i)))

for mu in mu_i:
    print('progress: ' + str(round(mu/mu_max * 100, 4)) + '%')
    for t in t_i:
        psi = psi_init
        psi_prev = 1000000000000
        while np.abs(psi_prev - psi) > tolerance:
            H = (-mu-U/2)*n + U/2*np.matmul(n,n) -z*t*psi*bdag - z*t*np.conjugate(psi) + z*t*np.absolute(psi)
            E, V = np.linalg.eigh(H)
            idx = np.argsort(E)
            V = V[:,idx]
            gs = V[:,0]
            psi_prev = psi
            psi = np.abs(gs.dot(b.dot(gs)))
        psimat[mu_i.index(mu),t_i.index(t)] = psi

fig, ax = plt.subplots()
im = ax.imshow(psimat, origin = 'lower', extent=(t_min,t_max,mu_min,mu_max), aspect = 'auto')
cb = plt.colorbar(im)
cb.set_label(r'$\Psi$', size=15)
ax.set_xlabel(r'$t/U$', fontsize=15)
ax.set_ylabel(r'$\mu/U$', fontsize=15)



            
        
            