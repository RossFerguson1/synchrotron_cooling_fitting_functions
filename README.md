# synchrotron_cooling_fitting_functions
Python module implementing the fitting functions for local synchrotron cooling (emission and absorption coefficients) presented in Ferguson & Margalit (2026b). 

The fitting functions can be implemented using the four provided classes, corresponding to (1) pitch-angle-averaged power-law emission, (2) perpendicular-pitch-angle power-law emission, (3) pitch-angle-averaged thermal emission, and (4) perpendicular-pitch-angle power-law emission. Exact numerical integral forms are given at the top of the module.

For example, the pitch-angle averaged emission and absorption coefficients may be calculated for a given p, x1, and eta (see Ferguson & Margalit (2026b) for definitions of the independent variables) as

  pl_functions = Power_Law_Cooling_Pitch_Angle_Averaged_Class(x1, eta, p)
  thermal_functions = Thermal_Cooling_Pitch_Angle_Averaged_Class(y, zmax)
  
  J_pl = pl_functions.J_pl()
  A_pl = pl_functions.A_pl()
  J_th = thermal_functions.J_th()
  A_th = thermal_functions.A_th()
  
Note that each class contains several methods with which the end user need not interact. The functions J_pl(), A_pl(), J_th(), and A_th() are the only methods that need to be called on directly.
