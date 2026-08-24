# synchrotron_cooling_fitting_functions
Python module implementing the fitting functions for local synchrotron cooling (emission and absorption coefficients) presented in Ferguson & Margalit (2026b). 

The fitting functions can be implemented using the four provided classes, corresponding to (1) pitch-angle-averaged power-law emission, (2) perpendicular-pitch-angle power-law emission, (3) pitch-angle-averaged thermal emission, and (4) perpendicular-pitch-angle power-law emission. Exact numerical integral forms are given at the top of the module.
  
Note that each class contains several methods with which the end user need not interact. J_pl(), A_pl(), J_th(), and A_th() are the only methods that need to be called on directly.
