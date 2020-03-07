# Mean-Field-Theory-Bose-Hubbard-Model #
Simple Mean-field theory simulation of the Bose-Hubbard model with the aim of capturing the phase diagram.

#dependencies#

~Requires Quspin

~~(It is easiest to get Anaconda, update it thoroughly, and install Quspin and all its own dependencies automatically
using the info here: https://anaconda.org/weinbe58/quspin)~~

###USAGE###

#base system parameters#
~U = On-site potential
~z = number of nearest neighbours
~max_p = maximum number of particles per site
~L = Length of chain (currently not in use)

#initialisation of system
EXAMPLE:

s = SPSystem(U = 1, z = 4, max_p = 6, L = 1) 

's' is the system name in this case, 'SPSystem' is the name of the object

#simulation parameters, and calling the simulation method#
~system = s #name of system (a variable in the simulation method)
>x = t/U<
>y = mu/U<
~xmin = minimum value of x
~xmax = maximum value of y
~xsteps = resolution of diagram in the x direction
~ymin, ymax, ysteps = similar to x case

#simulation of system (includes mechanism for plotting phase diagram)
EXAMPLE:

simulate(system = s, xmin = 0, xmax = 0.05, xsteps = 50, 
         ymin = 0, ymax = 3, ysteps = 50) #calling simulate function, does calculations

