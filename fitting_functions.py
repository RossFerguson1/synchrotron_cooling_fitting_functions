'''Synchrotron Fitting Functions: Local Cooling

    This module calculates the local cooled synchrotron emission and absorption from two populations of electrons,
    power-law and thermal, following the formalism presented in Ferguson & Margalit (2026b; FM26b). The main functions
    are J_pl, A_pl, J_th, and A_th. The other functions are various intermediate expressions used to generate the final
    fitting functions, along with numerical calculations and other useful approximate forms.

    This file can be imported as a module and contains the following functions and classes:
        * I_prime - eq. (32), Mahadevan et al. 1996 (hereafter, Mahadevan96)
        * F_pitch_angle_averaged - Pitch-angle averaged synchrotron function (F_tilde, eqs. (B9-B13), FM26b)
        * H_pitch_angle_averaged - Pitch-angle averaged auxiliary absorption synchrotron function (H_tilde, eqs. (B17, B19), FM26b)
        * F - Normal synchrotron function (F, eq. (B14), FM26b). This form is appropriate for perendicular pitch angles (sin alpha = 1)
        * H - Non-pitch-angle averaged auxiliary absorption synchrotron function (H, eqs. (B15, B16, B18), FM26b)
        * J_pl_integral - Complete integral form used to calculate the pitch-angle-averaged power-law emissivity  (eq. (18), FM26b)
        * A_pl_integral -  Complete integral form used to calculate the pitch-angle-averaged power-law absorption  (eq. (40), FM26b)
        * J_th_integral - Complete integral form used to calculate the pitch-angle-averaged thermal emissivity  (eq. (60), FM26b)
        * A_th_integral -  Complete integral form used to calculate the pitch-angle-averaged thermal absorption  (eq. (63), FM26b)
        * J_pl_integral_perp : Complete integral form used to calculate the power-law emissivity for perpendicular pitch angles with sin alpha = 1  (see Appendix C of FM26b)
        * A_pl_integral_perp : Complete integral form used to calculate the power-law absorption for perpendicular pitch angles with sin alpha = 1  (see Appendix C of FM26b)
        * J_th_integral_perp : Complete integral form used to calculate the thermal emissivity for perpendicular pitch angles with sin alpha = 1  (see Appendix C of FM26b)
        * A_th_integral_perp : Complete integral form used to calculate the thermal absorption for perpendicular pitch angles with sin alpha = 1  (see Appendix C of FM26b)

        * Power_Law_Cooling_Pitch_Angle_Averaged_Class - Class which collects fitting functions for the pitch-angle-averaged radiation coefficients of a cooled power-law distribution (Section 3, FM26b)
        * Power_Law_Cooling_Perpendicular_Pitch_Angle_Class - Class which collects fitting functions for the perpendicular pitch-angle radiation coefficients of a cooled power-law distribution (Appendix C.1, FM26b)
        * Thermal_Cooling_Pitch_Angle_Averaged_Class - Class which collects fitting functions for the pitch-angle-averaged radiation coefficients of a cooled thermal distribution (Section 4, FM26b)
        * Thermal_Cooling_Perpendicular_Pitch_Angle_Class - Class which collects fitting functions for the perpendicular pitch-angle radiation coefficients of a cooled thermal distribution (Appendix C.2, FM26b)
    '''
import numpy as np
from scipy import special
import scipy.integrate as integrate

'''
APPROXIMATE AND EXACT FORMS
'''
def I_prime(y):
    '''Auxiliary thermal function I'(x) (eq. (32), Mahadevan96). Can be used to calculate J_th (x * I_prime(x)) and
       A_th (0.5*therm.I_prime(x)/x) in the limit z_infinity >> 1.

    Parameters
    --------
        y: array
            Nondimensional frequency (nu/nu_0)/Theta^2, where nu_0 is the characteristic frequency 3eB/4pi m_e c and 
            Theta is the temperature in units of the electron mass

    Returns
    --------
        I_prime : array
    '''
    fac1 = 4.0505/y**(1/6)
    fac2 = 1 + 0.4/y**0.25 + 0.5316/y**0.5
    fac3 = np.exp(-1.8899*y**(1/3))
    return fac1*fac2*fac3

def F_pitch_angle_averaged(x):
    '''Pitch-angle averaged synchrotron function (F_tilde, eqs. (B9-B13), FM26b). 

    Parameters
    --------
        x: array
            Nonthermal characteristic frequency (nu/nu_0)/gamma^2, where nu_0 is the characteristic frequency 3eB/4pi m_e c and 
            gamma is the electron Lorentz factor

    Returns
    --------
        F_tilde : array
    '''
    prefactor = (1.808*x**(1/3))/np.sqrt(1+3.4*x**(2/3))
    num = 1 + 2.21*x**(2/3) + 0.347*x**(4/3)
    denom = 1 + 1.353*x**(2/3) + 0.217*x**(4/3)
    return prefactor*num*np.exp(-x)/denom

def H_pitch_angle_averaged(x):
    '''Pitch-angle averaged auxiliary absorption synchrotron function (H_tilde, eqs. (B17, B19), FM26b). 

    Parameters
    --------
        x: array
            Nonthermal characteristic frequency (nu/nu_0)/gamma^2, where nu_0 is the characteristic frequency 3eB/4pi m_e c and 
            gamma is the electron Lorentz factor

    Returns
    --------
        H_tilde : array
    '''
    #Max rel. error is 3.69% in the range 10^-5 to 10^2.5
    fac1 =   1.206*(1+1.45*x**1.262)**0.53
    fac2 = (1 + -0.4615*x**(2/3) + 2.35*x**(4/3))/(1 +  -0.656*x**(2/3) + 2.192*x**(4/3))
    func = x**(-5/3)*np.exp(-x)#*(1/x)
    return fac1*fac2*func

def F(x):
    '''Normal synchrotron function (F, eq. (B14), FM26b). This form is appropriate for perendicular pitch angles (sin alpha = 1)

    Parameters
    --------
        x: array
            Nonthermal characteristic frequency (nu/nu_0)/gamma^2, where nu_0 is the characteristic frequency 3eB/4pi m_e c and 
            gamma is the electron Lorentz factor

    Returns
    --------
        F_tilde : array
    '''
    prefactor = 2.15*(x**(1/3))*((1 + 3.06*x)**(1/6))
    num = 1 + 0.884*x**(2/3) + 0.471*x**(4/3)
    denom = 1 + 1.64*x**(2/3) + 0.974*x**(4/3)
    return prefactor*num*np.exp(-x)/denom

def H(x):
    '''Non-pitch-angle averaged auxiliary absorption synchrotron function (H, eqs. (B15, B16, B18), FM26b). This form is appropriate for perendicular pitch angles (sin alpha = 1)

    Parameters
    --------
        x: array
            Nonthermal characteristic frequency (nu/nu_0)/gamma^2, where nu_0 is the characteristic frequency 3eB/4pi m_e c and 
            gamma is the electron Lorentz factor

    Returns
    --------
        H_tilde : array
    '''
    fac1 =  1.433*(1+0.764*x**0.89)**1.31
    fac2 = (1 + 0.181*x**(2/3) + 0.628*x**(4/3))/(1 + 0.33*x**(2/3) + 0.505*x**(4/3))
    func = x**(-5/3)*np.exp(-x)#*(1/x)
    return fac1*fac2*func

#Pitch-angle-averaged integrals
def J_pl_integral(x1, eta, p, res):
    '''Complete integral form used to calculate the pitch-angle-averaged power-law emissivity  (eq. (18), FM26b)

    Parameters
    --------
    x1 : float
        Ratio of observed frequency nu to the characteristic synchrotron frequency gamma1^2 nu_0 of an electron with Lorentz factor gamma1. Note that nu_0 is the electron gyrofrequency
    eta : float
        Describes the range of the cooled power-law distribution; given by eta = gamma_inf / gamma1 = (x1/x_inf)^2
    p : float
        Power-law spectral index; initial distribution goes as gamma^-p
    res: int
        Number of points with which to calculate the numerical integral
    Returns
    --------
        J_pl : float
    '''
    x2 = x1/eta**2
    xvals = np.logspace(np.log10((1+1e-15)*x2),np.log10(x1),res)
    a, y = p-2,np.sqrt(x2/xvals)
    F_twiggle = F_pitch_angle_averaged(xvals)
    return integrate.simpson(F_twiggle*xvals**((p-3)/2)*(1-y)**a,x=xvals)

def A_pl_integral(x1, eta, p, res):
    '''Complete integral form used to calculate the pitch-angle-averaged power-law absorption  (eq. (40), FM26b)

    Parameters
    --------
    x1 : float
        Ratio of observed frequency nu to the characteristic synchrotron frequency gamma1^2 nu_0 of an electron with Lorentz factor gamma1. Note that nu_0 is the electron gyrofrequency
    eta : float
        Describes the range of the cooled power-law distribution; given by eta = gamma_inf / gamma1 = (x1/x_inf)^2
    p : float
        Power-law spectral index; initial distribution goes as gamma^-p
    res: int
        Number of points with which to calculate the numerical integral
    Returns
    --------
        A_pl : float
    '''
    x2 = x1/eta**2
    xvals = np.logspace(np.log10((1+1e-15)*x2),np.log10(x1),res)
    a, y = p-2,np.sqrt(x2/xvals)
    F_twiggle = F_pitch_angle_averaged(xvals)
    alpha_deriv = np.gradient(F_twiggle/xvals, xvals)
    return integrate.simpson(-(xvals**((p+2)/2))*(alpha_deriv)*((1-y)**a), x=xvals)

def J_th_integral(y, zmax, res):
    '''Complete integral form used to calculate the pitch-angle-averaged thermal emissivity  (eq. (60), FM26b). Uses the 's' variable, which
    improves numerical accuracy compared to using the 'z' variable.

    Parameters
    --------
    y:  float
        Nondimensional frequency (nu/nu_0)/Theta^2, where nu_0 is the characteristic frequency 3eB/4pi m_e c and Theta is
        the temperature in units of the electron mass
    zmax: float
        Maximum value of z in the cooled distribution, taken to be equal to gamma_infinity/Theta
    res: int
        Number of points with which to calculate the numerical integral

    Returns
    --------
        J_th : float
    '''
    svals = np.logspace(-10, 10, res)
    F_vals = F_pitch_angle_averaged(y*(1 + svals/zmax)**2/svals**2)
    integrand =  np.exp(-svals) * svals**2 * F_vals
    return integrate.simpson(integrand, x=svals)

def A_th_integral(y, zmax, res):
    '''Complete integral form used to calculate the pitch-angle-averaged thermal absorption  (eq. (63), FM26b). Uses the 's' variable, which
    improves numerical accuracy compared to using the 'z' variable.

    Parameters
    --------
    y:  float
        Nondimensional frequency (nu/nu_0)/Theta^2, where nu_0 is the characteristic frequency 3eB/4pi m_e c and Theta is
        the temperature in units of the electron mass
    zmax: float
        Maximum value of z in the cooled distribution, taken to be equal to gamma_infinity/Theta
    res: int
        Number of points with which to calculate the numerical integral
    Returns
    --------
        A_th : float
    '''
    svals = np.logspace(-10, 10, res)
    H_vals = H_pitch_angle_averaged(y*(1 + svals/zmax)**2/svals**2)
    integrand =  np.exp(-svals) * svals**-3 * H_vals * (1 + svals/zmax)**5
    return integrate.simpson(integrand, x=svals)

#Perpendicular pitch-angle integrals
def J_pl_integral_perp(x1, eta, p, res):
    '''Complete integral form used to calculate the power-law emissivity for perpendicular pitch angles with sin alpha = 1  (see Appendix C of FM26b)

    Parameters
    --------
    x1 : float
        Ratio of observed frequency nu to the characteristic synchrotron frequency gamma1^2 nu_0 of an electron with Lorentz factor gamma1. Note that nu_0 is the electron gyrofrequency
    eta : float
        Describes the range of the cooled power-law distribution; given by eta = gamma_inf / gamma1 = (x1/x_inf)^2
    p : float
        Power-law spectral index; initial distribution goes as gamma^-p
    res: int
        Number of points with which to calculate the numerical integral
    Returns
    --------
        J_pl : float
    '''
    x2 = x1/eta**2
    xvals = np.logspace(np.log10((1+1e-15)*x2),np.log10(x1),res)
    a, y = p-2,np.sqrt(x2/xvals)
    F_twiggle = F(xvals)
    return integrate.simpson(F_twiggle*xvals**((p-3)/2)*(1-y)**a,x=xvals)

def A_pl_integral_perp(x1, eta, p, res):
    '''Complete integral form used to calculate the power-law absorption for perpendicular pitch angles with sin alpha = 1  (see Appendix C of FM26b)

    Parameters
    --------
    x1 : float
        Ratio of observed frequency nu to the characteristic synchrotron frequency gamma1^2 nu_0 of an electron with Lorentz factor gamma1. Note that nu_0 is the electron gyrofrequency
    eta : float
        Describes the range of the cooled power-law distribution; given by eta = gamma_inf / gamma1 = (x1/x_inf)^2
    p : float
        Power-law spectral index; initial distribution goes as gamma^-p
    res: int
        Number of points with which to calculate the numerical integral
    Returns
    --------
        A_pl : float
    '''
    x2 = x1/eta**2
    xvals = np.logspace(np.log10((1+1e-15)*x2),np.log10(x1),res)
    a, y = p-2,np.sqrt(x2/xvals)
    F_twiggle = F_pitch_angle_averaged(xvals)
    alpha_deriv = np.gradient(F_twiggle/xvals, xvals)
    return integrate.simpson(-(xvals**((p+2)/2))*(alpha_deriv)*((1-y)**a), x=xvals)

def J_th_integral_perp(y, zmax, res):
    '''Complete integral form used to calculate the thermal emissivity for perpendicular pitch angles with sin alpha = 1  (see Appendix C of FM26b). Uses the 's' variable, which
    improves numerical accuracy compared to using the 'z' variable.

    Parameters
    --------
    y:  float
        Nondimensional frequency (nu/nu_0)/Theta^2, where nu_0 is the characteristic frequency 3eB/4pi m_e c and Theta is
        the temperature in units of the electron mass
    zmax: float
        Maximum value of z in the cooled distribution, taken to be equal to gamma_infinity/Theta
    res: int
        Number of points with which to calculate the numerical integral

    Returns
    --------
        J_th : float
    '''
    svals = np.logspace(-10, 10, res)
    F_vals = F(y*(1 + svals/zmax)**2/svals**2)
    integrand =  np.exp(-svals) * svals**2 * F_vals
    return integrate.simpson(integrand, x=svals)

def A_th_integral_perp(y, zmax, res):
    '''Complete integral form used to calculate the thermal absorption for perpendicular pitch angles with sin alpha = 1  (see Appendix C of FM26b). Uses the 's' variable, which
    improves numerical accuracy compared to using the 'z' variable.

    Parameters
    --------
    y:  float
        Nondimensional frequency (nu/nu_0)/Theta^2, where nu_0 is the characteristic frequency 3eB/4pi m_e c and Theta is
        the temperature in units of the electron mass
    zmax: float
        Maximum value of z in the cooled distribution, taken to be equal to gamma_infinity/Theta
    res: int
        Number of points with which to calculate the numerical integral
    Returns
    --------
        A_th : float
    '''
    svals = np.logspace(-10, 10, res)
    H_vals = H(y*(1 + svals/zmax)**2/svals**2)
    integrand =  np.exp(-svals) * svals**-3 * H_vals * (1 + svals/zmax)**5
    return integrate.simpson(integrand, x=svals)


'''POWER-LAW FITTING FUNCTIONS'''

class Power_Law_Cooling_Pitch_Angle_Averaged_Class:
    '''Class which collects fitting functions for the pitch-angle-averaged radiation coefficients of a cooled power-law distribution (Section 3, FM26b). 
        The distribution is a power-law between Lorentz factors gamma_1 and gamma_inf, and zero outside

    Parameters
    --------
    x1 : array
        Ratio of observed frequency nu to the characteristic synchrotron frequency gamma1^2 nu_0 of an electron with Lorentz factor gamma1. Note that nu_0 is the electron gyrofrequency
    eta : float
        Describes the range of the cooled power-law distribution; given by eta = gamma_inf / gamma1 = (x1/x_inf)^2
    p : float
        Power-law spectral index; initial distribution goes as gamma^-p

    This class contains the following methods:
        *A1 : Intermediate-frequency power-law emissivity constant  (eq. (72), FM26b)
        *B1 : Intermediate-frequency power-law absorption constant  (eq. (46), FM26b)
        *psi_p : High-frequency power-law steepest descent approximation  (eqs. (25, 29), FM26b)
        *Omega_p : Exact low-frequency solution for power-law emissivity   (eq. (19), FM26b)
        *chi_p : Exact low-frequency solution for power-law absorption   (eq. (43), FM26b)
        *Psi_p : Fitting function joining intermediate- and high-frequency solutions for power-law emission  (eq. (30), FM26b)
        *Sigma_p : Fitting function joining intermediate- and high-frequency solutions for power-law emission  (eq. (49), FM26b)
        *J_pl : Fitting function for power-law emissivity  (eq. (33), FM26b)
        *A_pl : Fitting function for power-law absorption  (eq. (52), FM26b)
    '''
    
    def __init__(self, x1, eta, p):
        x1_arr, eta_arr = np.meshgrid(x1, eta)
        self.x1 = x1_arr
        self.eta = eta_arr
        self.p: float = p

    def A1(self):
        """Intermediate-frequency power-law emissivity constant  (eq. (72), FM26b)

        Returns
        --------
        A1: array
        """
        p = self.p
        return np.sqrt(np.pi)*2**((p+3)/2)*special.gamma(p/4 + 5/4)*special.gamma(p/4 + 19/12)*special.gamma(p/4 -1/12)/\
                    ((p+3)*(p+1)*special.gamma(p/4 + 3/4))
    
    def B1(self):
        """Intermediate-frequency power-law absorption constant  (eq. (46), FM26b)

        Returns
        --------
        B1: array
        """
        p = self.p
        return np.sqrt(np.pi)*2**((p+2)/2)*(1/(p+4))*special.gamma(p/4 + 3/2)*special.gamma(p/4 + 11/6)*special.gamma(p/4 + 1/6)/ special.gamma(p/4+1)

    def psi_p(self, a):
        """High-frequency power-law steepest descent approximation  (eqs. (25, 29), FM26b)

        Returns
        --------
        psi_p: array
        """
        p = self.p
        x2 = self.x1/self.eta**2
        return special.gamma(p-1)/((2*x2)**(p-1))*np.pi*x2**(a+1)*np.exp(-x2)

    def Omega_p(self):
        """Exact low-frequency solution for power-law emissivity   (eq. (19), FM26b)

        Returns
        --------
        Omega_p: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1

        prefactor = (6/(3*p-1))
        hyper_term1 = special.hyp2f1(-(p-2), -(p-1/3), -(p-4/3), 1/eta)
        hyper_term2 = (special.gamma(p-1)*special.gamma(-(p-4/3))/special.gamma(-2/3))*eta**(-(p-1/3))

        val = prefactor*x1**((p-1)/2)*F_pitch_angle_averaged(x1)*(hyper_term1 - hyper_term2)
        val[eta<1.0005] = (x1**((p-1)/2)*F_pitch_angle_averaged(x1)* 2**(1)*(1/(p-1))*(eta-1)**(p-1))[eta<1.0005]
        return val
        
    def chi_p(self):
        """Exact low-frequency solution for power-law absorption   (eq. (43), FM26b)

        Returns
        --------
        chi_p: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1
        x2 = x1/eta**2
        
        prefactor = (6/(3*p+2))
        hyper_term1 = special.hyp2f1(-(p-2), -(3*p+2)/3, -(3*p-1)/3, 1/eta)*eta**((3*p+2)/3)
        hyper_term2 = (special.gamma(p-1)*special.gamma(-(3*p-1)/3)/special.gamma(-5/3))
        

        val = prefactor*x2**((p+4)/2)*(hyper_term1 - hyper_term2)*H_pitch_angle_averaged(x2)
        val[eta<1.0005] = (x1**((p+4)/2)*H_pitch_angle_averaged(x1)* 2**(1)*(1/(p-1))*(eta-1)**(p-1))[eta<1.0005]
        return val

    def const(self, coeff_params):
        return np.polynomial.polynomial.Polynomial(coeff_params)(self.p)

    def Psi_p(self):
        """Fitting function joining intermediate- and high-frequency solutions for power-law emission  (eq. (30), FM26b)

        Returns
        --------
        Psi_p: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1
        x2 = x1/eta**2

        #a_i arrays (aleph_ij values)
        a1_params = np.array([ 0.50167391, -0.28651117,  0.05670197, -0.00375472])
        a2_params = np.array([ 1.11499065,  0.44114452, -0.00551598])
        a3_params = np.array([-3.26558351,  3.16921880, -0.717951408,  7.21730411e-02, -2.20057647e-03])
        a4_params = np.array([-0.22100298,  0.72074841, -0.35194957,  0.06491111, -0.00410027])

        Psi_low = self.A1()
        Psi_high = self.psi_p((p-3)/2)

        a1, a2, a3, a4 = self.const( a1_params), self.const(a2_params), self.const(a3_params), self.const(a4_params)
        delta1_psi = np.exp(-a1*x2**2 -a2*x2**(2/3))
        delta2_psi = (1-np.exp( -a4*x2))**a3
        return Psi_low*delta1_psi + Psi_high*delta2_psi

    def Sigma_p(self):
        """Fitting function joining intermediate- and high-frequency solutions for power-law emission  (eq. (49), FM26b)

        Returns
        --------
        Sigma_p: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1
        x2 = x1/eta**2

        #b_i arrays (beth_ij values)
        b1_params = np.array([ 0.16243734, -0.12370775,  0.04112642, -0.00643677,  0.00037763])
        b2_params = np.array([-9.01460783, 11.28277205, -4.19792133,  0.67281174, -0.03636567])
        b3_params = np.array([ 43.42959643, -50.35287845,  21.14604186,  -3.24801454,0.16323743])
        b4_params = np.array([-2.56584127,  6.36572335, -2.5054413 ,  0.42788566, -0.02725223])

        Sigma_low = np.sqrt(np.pi)*2**((p+2)/2)*(1/(p+4))*special.gamma(p/4 + 3/2)*special.gamma(p/4 + 11/6)*special.gamma(p/4 + 1/6)/ \
                            special.gamma(p/4+1)
        Sigma_high = self.psi_p(p/2) + self.psi_p((p-2)/2)

        b1, b2, b3, b4= self.const(b1_params), self.const(b2_params), self.const(b3_params), self.const(b4_params)
        
        delta1_sigma = np.exp(- b1*x2**(2) - b2*x2**(2/3))
        delta2_sigma = (1-np.exp(  -b4*x2**(1/3)))**b3
        return Sigma_low*delta1_sigma + Sigma_high*delta2_sigma

    #Emission and absorption coefficients
    def J_pl(self):
        """Fitting function for power-law emissivity  (eq. (33), FM26b)

        Returns
        --------
        J_pl: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1

        epsilon = eta**2 -1

        alpha1 = -3.85047272e+03 + 4.29887467e+03*p/(p-8.67876066e-04)  -4.48444088e+02 * p**3.09238160e-04
        alpha1 =  -0.0322*p**3 + 0.4484*p**2 - 2.294*p + 4.77
        alpha2  = 0.62210982   + 0.34653884*p**(2/3)  -0.01704838*p**(4/3)

        alpha3 =  1 + (0.1*p-0.71)*np.exp(-1*(eta**2-1.1)**2)
        alpha4_high_eta = 2 - 0.5*np.exp(-1*(p-2)**2)
        alpha4 = alpha4_high_eta + (0.538*p+0.77 - alpha4_high_eta)*np.exp(-0.01*(eta**2-1.5)**2)

        func_low = self.Omega_p()
        func_high = self.Psi_p() 

        delta_low = np.exp(-alpha1*(x1)**alpha2*np.exp(-alpha3/epsilon**0.8))
        delta_high = (1-delta_low)**alpha4

        return func_low*delta_low + func_high*delta_high

    def A_pl(self):
        """Fitting function for power-law absorption  (eq. (52), FM26b)

        Returns
        --------
        A_pl: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1

        epsilon = eta**2 -1

        func_low = self.chi_p()
        func_high = self.Sigma_p()


        beta1 = 0.07690576 + 29.15548788*(p+10.71178803)**-2

        beta1_nominal = 0.07690576 + 29.15548788*(p+10.71178803)**-2
        beta1_nominal =  0.31089795 + 0.17509693*(p-2)**(2/3)  -0.00393917*(p-2.00999972)** 2.28322644

        beta1 =  beta1_nominal + (-0.063*p + 0.33-beta1_nominal)*np.exp(-10*(eta**2-1.1)**2)
        beta2 = 2 + 2.5*np.exp(-100*(eta**2-1.1)**2)
        beta3 = 1 - 0.43*np.exp(-500*(eta**2-1.01)**2)
        beta4 = 2.97*p - 3.13


        delta_low = np.exp(-(beta1*(x1)**beta2 )*np.exp(-1/epsilon**beta3))
        delta_high = (1-delta_low)**(beta4) 

        return func_low*delta_low + func_high*delta_high

class Power_Law_Cooling_Perpendicular_Pitch_Angle_Class:
    '''Class which collects fitting functions for the perpendicular pitch-angle radiation coefficients of a cooled power-law distribution (Appendix C.1, FM26b). 
        The distribution is a power-law between Lorentz factors gamma_1 and gamma_inf, and zero outside

    Parameters
    --------
    x1 : array
        Ratio of observed frequency nu to the characteristic synchrotron frequency gamma1^2 nu_0 of an electron with Lorentz factor gamma1. Note that nu_0 is the electron gyrofrequency
    eta : float
        Describes the range of the cooled power-law distribution; given by eta = gamma_inf / gamma1 = (x1/x_inf)^2
    p : float
        Power-law spectral index; initial distribution goes as gamma^-p

    This class contains the following methods:
        *A1 : Intermediate-frequency power-law emissivity constant  (eq. (C24), FM26b)
        *B1 : Intermediate-frequency power-law absorption constant  (eq. (C25), FM26b)
        *psi_p : High-frequency power-law steepest descent approximation  (eqs. (25, 29), FM26b), adjusted for perpendicular pitch-angles (adding an extra factor of sqrt(2/pi))
        *Omega_p : Exact low-frequency solution for power-law emissivity (eq. (19), FM26b) adjusted for perpendicular pitch-angles via the replacement F_tilde -> F
        *chi_p : Exact low-frequency solution for power-law absorption   (eq. (43), FM26b) adjusted for perpendicular pitch-angles via the replacement H_tilde -> H
        *Psi_p : Fitting function joining intermediate- and high-frequency solutions for power-law emission  (eq. (C22), FM26b)
        *Sigma_p : Fitting function joining intermediate- and high-frequency solutions for power-law emission  (eq. (C23), FM26b)
        *J_pl : Fitting function for power-law emissivity  (eq. (C20), FM26b)
        *A_pl : Fitting function for power-law absorption  (eq. (C21), FM26b)
    '''
    
    def __init__(self, x1, eta, p):
        x1_arr, eta_arr = np.meshgrid(x1, eta)
        self.x1 = x1_arr
        self.eta = eta_arr
        self.p: float = p

    def A1(self):
        """Intermediate-frequency power-law emissivity constant (eq. (C24), FM26b)

        Returns
        --------
        A1: array
        """
        p = self.p
        return 2**((p+1)/2)*special.gamma(p/4 + 19/12)*special.gamma(p/4 -1/12)/(p+1)

    def B1(self):
        """Intermediate-frequency power-law absorption constant  (eq. (C25), FM26b)

        Returns
        --------
        B1: array
        """
        p = self.p
        return 2**(p/2)*special.gamma(p/4 + 11/6)*special.gamma(p/4 + 1/6)

    def psi_p(self, a):
        """High-frequency power-law steepest descent approximation  (eqs. (25, 29), FM26b), adjusted for perpendicular pitch-angles (adding an extra factor of sqrt(2/pi))

        Returns
        --------
        psi_p: array
        """
        p = self.p
        x2 = self.x1/self.eta**2
        return special.gamma(p-1)/((2*x2)**(p-1))*np.pi*x2**(a+1)*np.exp(-x2)

    def Omega_p(self):
        """Exact low-frequency solution for power-law emissivity (eq. (19), FM26b) adjusted for perpendicular pitch-angles via the replacement F_tilde -> F

        Returns
        --------
        Omega_p: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1

        prefactor = (6/(3*p-1))
        hyper_term1 = special.hyp2f1(-(p-2), -(p-1/3), -(p-4/3), 1/eta)
        hyper_term2 = (special.gamma(p-1)*special.gamma(-(p-4/3))/special.gamma(-2/3))*eta**(-(p-1/3))

        val =prefactor*x1**((p-1)/2)*F(x1)*(hyper_term1 - hyper_term2)
        val[eta<1.0005] = ( x1**((p-1)/2)*F(x1)* 2**(1)*(1/(p-1))*(eta-1)**(p-1))[eta<1.0005]
        return val

    def chi_p(self):
        """Exact low-frequency solution for power-law absorption   (eq. (43), FM26b) adjusted for perpendicular pitch-angles via the replacement H_tilde -> H

        Returns
        --------
        chi_p: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1
        x2 = x1/eta**2
        
        prefactor = (6/(3*p+2))
        hyper_term1 = special.hyp2f1(-(p-2), -(3*p+2)/3, -(3*p-1)/3, 1/eta)*eta**((3*p+2)/3)
        hyper_term2 = (special.gamma(p-1)*special.gamma(-(3*p-1)/3)/special.gamma(-5/3))
        

        val =prefactor*x2**((p+4)/2)*(hyper_term1 - hyper_term2)*H(x2)
        val[eta<1.0005] = ( x1**((p+4)/2)*H(x1)* 2**(1)*(1/(p-1))*(eta-1)**(p-1))[eta<1.0005]
        return val

    def const(self, coeff_params):
        return np.polynomial.polynomial.Polynomial(coeff_params)(self.p)

    def Psi_p(self):
        """Fitting function joining intermediate- and high-frequency solutions for power-law emission  (eq. (C22), FM26b)

        Returns
        --------
        Psi_p: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1
        x2 = x1/eta**2

        #a_i arrays (aleph_ij values)
        a1_params = np.array([ 2.80210328, -2.63894327,  0.92493235, -0.14187289,  0.00802239])
        a2_params = np.array([-5.59509629,  5.92027047, -1.75084253,  0.24762863, -0.01328967])
        a3_params = np.array([ 26.84239385, -26.93421468,  10.57662689,  -1.67495923, 0.09581768])
        a4_params = np.array([ 3.42342497, -3.31561261,  1.18259553, -0.18369935,  0.01048819])

        Psi_low = self.A1()
        Psi_high = np.sqrt(2/np.pi)*self.psi_p(p/2 -1)

        a1, a2, a3, a4 = self.const( a1_params), self.const(a2_params), self.const(a3_params), self.const(a4_params)
        delta1_psi = np.exp(-a1*x2**2  -a2*x2**(2/3))
        delta2_psi =  (1-np.exp(-a4*x2**2))**a3
        return Psi_low*delta1_psi + Psi_high*delta2_psi

    def Sigma_p(self):
        """Fitting function joining intermediate- and high-frequency solutions for power-law emission  (eq. (C23), FM26b)

        Returns
        --------
        Sigma_p: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1
        x2 = x1/eta**2

        #b_i arrays (beth_ij values)
        b1_params = np.array([ 0.16846994, -0.18412299,  0.07185229, -0.01154235,  0.00065533])
        b2_params = np.array([-2.98831308,  4.73677702, -1.96859118,  0.31706005, -0.0177813 ])
        b3_params = np.array([ 2.75111825, -4.60109385,  2.24668601, -0.3711466 ,  0.02103266])
        b4_params = np.array([-34.95209614,  33.92729159, -10.45741016,   1.35654969, -0.06313555])
        b5_params = np.array([ 9.25573766e+02, -1.20969017e+03,  6.16930944e+02, -1.53036031e+02, 1.85034856e+01, -8.74682989e-01])
        
        Sigma_low = self.B1()
        Sigma_high = np.sqrt(2/np.pi) * (self.psi_p((p+1)/2) + 0.5*self.psi_p((p-1)/2))

        b1, b2, b3, b4, b5= self.const(b1_params), self.const(b2_params), self.const(b3_params), self.const(b4_params), self.const(b5_params)
        
        delta1_chi = np.exp(-b1*x2**2 -b2*x2**1 - b3*x2**(2/3))
        delta2_chi =(1-np.exp(- b5*x2))**b4
        return Sigma_low*delta1_chi + Sigma_high*delta2_chi

    #Emission and absorption coefficients
    def J_pl(self):
        """Fitting function for power-law emissivity  (eq. (C20), FM26b)

        Returns
        --------
        J_pl: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1

        func_low = self.Omega_p()
        func_high = self.Psi_p() 

        epsilon = eta**2 -1

        alpha1 = 9.03385557e+02  - 9.02232169e+02*(p-0.940771247)**4.63515666e-04
        alpha2  = 0.62210982   + 0.34653884*p**(2/3)  -0.01704838*p**(4/3)
        alpha3 =  1 + (0.1*p-0.71)*np.exp(-10*(eta**2-1.1)**2)
        alpha4_high_eta = 2 - 0.5*np.exp(-1*(p-2)**2)
        alpha4 =alpha4_high_eta + (0.538*p+0.77 - alpha4_high_eta)*np.exp(-0.01*(eta**2-1.5)**2)
        
        delta_low = np.exp(-alpha1*(x1)**alpha2*np.exp(-alpha3/epsilon**0.8))
        delta_high = (1-delta_low)**(alpha4) 

        return func_low*delta_low + func_high*delta_high

    def A_pl(self):
        """Fitting function for power-law absorption  (eq. (C21), FM26b)

        Returns
        --------
        A_pl: array
        """
        p = self.p
        eta = self.eta
        x1 = self.x1

        epsilon = eta**2 -1

        func_low = self.chi_p()
        func_high = self.Sigma_p()


        beta1_nominal = 0.124254436 -2.26464609e+01*(p+6.10099043)**-3 -1.33775759e-04*(p-2.00999923)**2.72590914 + 5.13811875e-01*(p-3.43615856e-01)**-5
        beta1 = beta1_nominal  - 0.02*np.exp(-1e4*(eta-1.1)**2)
        beta2 = 2 
        beta3 = 0.47
        beta4 = 2.97*p - 3.13


        delta_low = np.exp(-(beta1*(x1)**beta2 )*np.exp(-1/epsilon**beta3))
        delta_high = (1-delta_low)**(beta4) 

        return func_low*delta_low + func_high*delta_high


'''THERMAL FITTING FUNCTIONS'''

class Thermal_Cooling_Pitch_Angle_Averaged_Class:
    '''Class which collects fitting functions for the pitch-angle-averaged radiation coefficients of a cooled thermal distribution (Section 4, FM26b). 
        The distribution takes the form of a cooled ultrarelativistic Maxwell-Juttner distribution between Lorentz factors 1 and gamma_inf

    Parameters
    --------
    y : array
        Key frequency for cooled thermal distribution, defined below eq. (60) of FM26b as y == (nu/nu_0)/ Theta_bar^2. Note that nu_0 is the electron gyrofrequency and Theta_bar = Theta/G_scr(t) is written in terms of ther adiabatic cooling function G_scr(t)
    zmax : float
        Rescaled Lorentz factor z_inf = gamma_inf / Theta_bar

    This class contains the following methods:
        *zs_fit : Analytic approximation for saddle point z_s (eq. (78), FM26b)
        *z_s : Exact calculation of saddle point z_s (defined by eq. (75), FM26b)
        *xi : Steepest descent approximation for thermal coefficients  (eqs. (71, 72), FM26b)
        *J_th : Thermal cooled emission fitting function (eq. (79), FM26b)
        *A_th : Thermal cooled absorption fitting function (eq. (80), FM26b)
    '''
    def __init__(self,y, zmax):
        y_arr, zmax_arr = np.meshgrid(y, zmax)
        self.y = y_arr
        self.zmax = zmax_arr

    def zs_fit(self):
        """Analytic approximation for saddle point z_s (eq. (78), FM26b)

        Returns
        --------
        z_s: array
            Value of saddle point z_s
        """
            
        ymax = 0.5* self.zmax**3
        a = 2.019
        P =  2.7e4*ymax**-4 + (  (2*self.y)**(1/3)  )

        ts_high = (-4/ self.zmax -3 + np.sqrt((4/ self.zmax + 3)**2 + 4*(self.y/ymax - 12/ self.zmax)))/(2*(self.y/ymax - 12/ self.zmax))
        return ((( self.zmax -  self.zmax*ts_high)**-a + P**-a)**(-1/a))

    def z_s(self):
        """Exact calculation of saddle point z_s (defined by eq. (75), FM26b)

        Returns
        --------
        z_s: array
            Exact saddle point z_s
        """
        #Exact
        a = 4/ self.zmax**2
        b = 1-4/ self.zmax
        c = -2*self.y/( self.zmax**2)
        d = 4*self.y/ self.zmax
        e = -2*self.y

        val = np.zeros_like(self.y)
        for i in range(len(self.y)):
            roots = np.roots((a,b,c[i],d[i],e[i]))
            real_roots = roots[np.isreal(roots)].real
            pos_real_roots = real_roots[real_roots > 0]
            val[i]= np.min(pos_real_roots)

        return val

    def xi(self, q):
        """Steepest descent approximation for thermal coefficients  (eqs. (71, 72), FM26b)

        Returns
        --------
        xi: array
        """
        
        #Saddle point calculated using approximate form (eq. (78) in FM26b)
        z_s_val = self.zs_fit()
        #If desired, a more accurate saddle point can be calculated numerically by un-commenting the following line:
        # z_s_val = z_s(zmax, np.array([y]))

        #Values of phi and phi_double_prime
        phi_val = -z_s_val/(1-z_s_val/ self.zmax) - self.y/z_s_val**2 - 4*np.log(1-z_s_val/ self.zmax)
        phi_double_prime = -2/(self.zmax*(1-z_s_val/ self.zmax)**3) - 6*self.y/z_s_val**4 + 4/( self.zmax-z_s_val)**2
        return np.sqrt(-2*np.pi/phi_double_prime)*z_s_val**q*np.exp(phi_val)

    def J_th(self):
        """Thermal cooled emission fitting function (eq. (79), FM26b)

        Returns
        --------
        J_th: array
        """

        zmax = self.zmax
        y = self.y
        #Approximate value of y_t (eq. 64 of FM26b). In this case, it is simpler to write the function using 0.5*zmax^3 rather
        #than 0.5*zmax^2.
        k = 100  
        y1 = 0.5*zmax**3
        y2 = 1
        y_t_approx = (-1/k)*np.log(np.exp(-k*y1) + np.exp(-k*y2))    

        #Auxiliary parameters in J_th fitting function (eqs. (81-87) in FM26b)
        lambda1 = 10**((0.64673914 + 1.015*np.log10(zmax))* (1 + 0.14995624*(zmax)**(2/3) -0.59109817*(zmax)**(4/3) +0.19483466*zmax**(6/3) )/(1 + 2.23609859*(zmax)**(2/3) -3.31209198*(zmax)**(4/3)+\
                                1.10042827*zmax**(6/3)))*(1-np.exp(-80/zmax**2.3)) + 7.435*np.exp(-80/zmax**1.5)
        lambda2 = 10**((-0.0131745879 + 1.015*np.log10(zmax))* (1 + 20.6864785*(zmax)**(2/3)-8.80832096*(zmax)**(4/3) +0.941199121*zmax**(6/3) )/(1 + 33.7710941*(zmax)**(2/3) -17.4277465*(zmax)**(4/3)+\
                                2.87106435*zmax**(6/3)))*(1-np.exp(-50/zmax**1.5)) + 4.703*np.exp(-50/zmax)
        sigma = np.exp(-1e3*zmax**-2)
        zeta = np.exp(-1e-7/zmax**2)

        #Limits for y<<y_t and y>>y_t (eqs. (69) and (79) in FM26b)
        J_high_y = (np.pi/2)*self.xi(2)

        J_low_y =  F_pitch_angle_averaged(y/zmax**2) * zmax**(9/3)*special.gamma(7/3)*special.hyperu(7/3, 4, zmax)
        J_low_y[zmax <= 1e-2] = (F_pitch_angle_averaged(y/zmax**2) * zmax**(9/3)*special.gamma(7/3)*(-27/(4*special.gamma(-2/3)*zmax**3) - 9/(4*special.gamma(-2/3)*zmax**2)))[zmax <= 1e-2]

        #Complete fitting function (eq. (79) in FM26b)
        return J_low_y*np.exp(-lambda1*(y/y_t_approx)**1.015 * zeta) + (1 + sigma*(y/( y_t_approx))**-0.5)*J_high_y*(1-np.exp(-lambda2*(y/y_t_approx)**1.015 * zeta))

    def A_th(self):
        """Thermal cooled absorption fitting function (eq. (80), FM26b)

        Returns
        --------
        A_th: array
        """
        zmax = self.zmax
        y = self.y
        #Approximate value of y_t (eq. 64 of FM26b). In this case, it is simpler to write the function using 0.5*zmax^3 rather than 0.5*zmax^2.
        k = 100  
        y1 = 0.5*zmax**3
        y2 = 1
        y_t_approx = (-1/k)*np.log(np.exp(-k*y1) + np.exp(-k*y2)) 

        #Auxiliary parameters in A_th fitting function (eqs. (81-87) in FM26b)
        mu1  = 10**(( -0.151040642  + 1.015*np.log10(zmax))* (1 +2.34130662e+05*(zmax)**(2/3)-1.79767518e+05 *(zmax)**(4/3) +2.32003343e+04 *zmax**(6/3))/(1 + 1.20587839e+05*(zmax)**(2/3)\
                -1.00937174e+05   *(zmax)**(4/3) +4.81969142e+04*zmax**(6/3)))*(1-np.exp(-80/zmax**1.5)) + 9.4073*np.exp(-80/zmax)
        mu2 = 10**(( -1.16426815    + 1.015*np.log10(zmax))* (1 +1.11467153e+05*(zmax)**(2/3)-6.18406042e+04 *(zmax)**(4/3) +5.82434317e+04 *zmax**(6/3))/(1 +  7.53599070e+04*(zmax)**(2/3)\
                +4.02618689e+04    *(zmax)**(4/3)  +9.80602555e+04*zmax**(6/3)))*(1-np.exp(-40/zmax**1.5)) + 5.6793*np.exp(-40/zmax)
        sigma = np.exp(-1e3*zmax**-2) 
        zeta = np.exp(-1e-7/zmax**2)

        #Limits for y<<y_t and y>>y_t (eqs. (70) and (80) in FM26b)
        A_high_y = (np.pi/2)*self.xi(-1)*y**-1 

        A_low_y =  H_pitch_angle_averaged(y/zmax**2) * zmax**(-6/3)*special.gamma(4/3)*special.hyperu(4/3, 4, zmax)
        A_low_y[zmax <= 1e-2] = ( H_pitch_angle_averaged(y/zmax**2) * zmax**(-6/3)*special.gamma(4/3)*(27/(5*special.gamma(-5/3)*zmax**3) + 9/(2*special.gamma(-5/3)*zmax**2)))[zmax <= 1e-2]
        
        #Complete fitting function (eq. (80) in FM26b)
        return A_low_y*np.exp(-mu1*(y/y_t_approx)**1.015 * zeta) + (1 + sigma*(y/(y_t_approx))**-0.5)*A_high_y*(1-np.exp(-mu2*(y/y_t_approx)**1.015 *zeta))

class Thermal_Cooling_Perpendicular_Pitch_Angle_Class:
    '''Class which collects fitting functions for the perpendicular pitch-angle radiation coefficients of a cooled thermal distribution (Appendix C.2, FM26b). 
        The distribution takes the form of a cooled ultrarelativistic Maxwell-Juttner distribution between Lorentz factors 1 and gamma_inf

    Parameters
    --------
    y : array
        Key frequency for cooled thermal distribution, defined below eq. (60) of FM26b as y == (nu/nu_0)/ Theta_bar^2. Note that nu_0 is the electron gyrofrequency and Theta_bar = Theta/G_scr(t) is written in terms of ther adiabatic cooling function G_scr(t)
    zmax : float
        Rescaled Lorentz factor z_inf = gamma_inf / Theta_bar

    This class contains the following methods:
        *zs_fit : Analytic approximation for saddle point z_s (eq. (78), FM26b)
        *z_s : Exact calculation of saddle point z_s (defined by eq. (75), FM26b)
        *xi : Steepest descent approximation for thermal coefficients  (eqs. (71, 72), FM26b)
        *J_th : Thermal cooled emission fitting function (eq. (C39), FM26b)
        *A_th : Thermal cooled absorption fitting function (eq. (C40), FM26b)
    '''
    def __init__(self,y, zmax):

        y_arr, zmax_arr = np.meshgrid(y, zmax)
        self.y = y_arr
        self.zmax = zmax_arr

    def zs_fit(self):
        """Analytic approximation for saddle point z_s (eq. (78), FM26b)

        Returns
        --------
        z_s: array
            Value of saddle point z_s
        """
            
        ymax = 0.5* self.zmax**3
        a = 2.019
        P =  2.7e4*ymax**-4 + (  (2*self.y)**(1/3)  )

        ts_high = (-4/ self.zmax -3 + np.sqrt((4/ self.zmax + 3)**2 + 4*(self.y/ymax - 12/ self.zmax)))/(2*(self.y/ymax - 12/ self.zmax))
        return ((( self.zmax -  self.zmax*ts_high)**-a + P**-a)**(-1/a))

    # def z_s(self)-> Float[np.ndarray, "..."]:
    def z_s(self):
        """Exact calculation of saddle point z_s (defined by eq. (75), FM26b)

        Returns
        --------
        z_s: array
            Exact saddle point z_s
        """
        #Exact
        a = 4/ self.zmax**2
        b = 1-4/ self.zmax
        c = -2*self.y/( self.zmax**2)
        d = 4*self.y/ self.zmax
        e = -2*self.y

        val = np.zeros_like(self.y)
        for i in range(len(self.y)):
            roots = np.roots((a,b,c[i],d[i],e[i]))
            real_roots = roots[np.isreal(roots)].real
            pos_real_roots = real_roots[real_roots > 0]
            val[i]= np.min(pos_real_roots)

        return val

    def xi(self, q):
        """Steepest descent approximation for thermal coefficients  (eqs. (71, 72), FM26b)

        Returns
        --------
        xi: array
        """
        
        #Saddle point calculated using approximate form (eq. (78) in FM26b)
        z_s_val = self.zs_fit()
        #If desired, a more accurate saddle point can be calculated numerically by un-commenting the following line:
        # z_s_val = z_s(zmax, np.array([y]))

        #Values of phi and phi_double_prime
        phi_val = -z_s_val/(1-z_s_val/ self.zmax) - self.y/z_s_val**2 - 4*np.log(1-z_s_val/ self.zmax)
        phi_double_prime = -2/(self.zmax*(1-z_s_val/ self.zmax)**3) - 6*self.y/z_s_val**4 + 4/( self.zmax-z_s_val)**2
        return np.sqrt(-2*np.pi/phi_double_prime)*z_s_val**q*np.exp(phi_val)

    # def J_th(self)-> Float[np.ndarray, "..."]:
    def J_th(self):
        """Thermal cooled emission fitting function (eq. (C39), FM26b)

        Returns
        --------
        J_th: array
        """

        zmax = self.zmax
        y = self.y
        #Approximate value of y_t (eq. 64 of FM26b)
        k = 100  
        y1 = zmax**2
        y2 = 1
        y_t_approx = (-1/k)*np.log(np.exp(-k*y1) + np.exp(-k*y2))    

        #Auxiliary parameters in J_th fitting function (eqs. (81-87) in FM26b)
        lambda1 = 10**(( 0.538   + 1.015*np.log10(zmax))* (1 +2.15143554e+07*(zmax)**(2/3) -1.21028385e+07*(zmax)**(4/3) +2.02587448e+06*zmax**(6/3))/(1 +1.85093368e+07*(zmax)**(2/3)\
                    -1.97519155e+07*(zmax)**(4/3) +6.31401474e+06*zmax**(6/3)))*(1-np.exp(-120/zmax**1.4)) + 5.0258*np.exp(-120/zmax**1.2)

        lambda2 = 10**(( 0.522    + 1.015*np.log10(zmax))* (1 +3.67360081e+07*(zmax)**(2/3) -2.02462043e+07*(zmax)**(4/3)+ 3.24282871e+06*zmax**(6/3))/(1 +2.95623985e+07*(zmax)**(2/3)\
                    -3.06797036e+07*(zmax)**(4/3) + 9.91420966e+06*zmax**(6/3)))*(1-np.exp(-120/zmax**1.4)) + 4.83913*np.exp(-120/zmax**1.2)
        sigma = 0.4*zmax**0.8*(1-np.exp(-120/zmax**2.65))+ 2.86*np.exp(-120/zmax**2)
        zeta = np.exp(-1e-7/zmax**1.8)

        #Limits for y<<y_t and y>>y_t (eqs. (69) and (79) in FM26b)
        J_high_y = np.sqrt(np.pi/2)*self.xi(1)*y**0.5

        J_low_y =  F(y/zmax**2) * zmax**(9/3)*special.gamma(7/3)*special.hyperu(7/3, 4, zmax)
        J_low_y[zmax <= 1e-2] = (F(y/zmax**2) * zmax**(9/3)*special.gamma(7/3)*(-27/(4*special.gamma(-2/3)*zmax**3) - 9/(4*special.gamma(-2/3)*zmax**2)))[zmax <= 1e-2]

        #Complete fitting function (eq. (79) in FM26b)
        return J_low_y*np.exp(-lambda1*(y/y_t_approx) * zeta) + (1 + sigma*(y/( y_t_approx))**-0.4)*J_high_y*(1-np.exp(-lambda2*(y/y_t_approx) * zeta))

    def A_th(self):
        """Thermal cooled absorption fitting function (eq. (C40), FM26b)

        Returns
        --------
        A_th: array
        """
        zmax = self.zmax
        y = self.y

        k = 100  
        x1, x2 = zmax**2, 1
        min_x = (-1/k)*np.log(np.exp(-k*x1) + np.exp(-k*x2))


        mu1 = 10**(( -0.6886695    + 0.591*np.log10(zmax))* (1 -1.644*(zmax)**(2/3) +  0.794*(zmax)**(4/3) -0.011*zmax**(6/3))/(1 - 0.8296*(zmax)**(2/3)\
                    +0.316*(zmax)**(4/3) +0.041*zmax**(6/3)))*(1-np.exp(-20/zmax**1.2)) + 1.86*np.exp(-20/zmax)

        mu2 = 10**(( -5.70831022e-03    + 2.75340535e-03*np.log10(zmax))* (1 +4.08e4*(zmax)**(2/3) -1.754e4*(zmax)**(4/3)+2.29e3*zmax**(6/3))/(1 +1.86e2*(zmax)**(2/3)\
                    -90.9*(zmax)**(4/3) +12.99*zmax**(6/3)))*(1-np.exp(-20/zmax**1.2)) + 0.812*np.exp(-20/zmax)

        sigma = 3.833  -3.83299*np.exp(-(zmax-1)**2)
        mu2 *= 1 + 4*np.exp(-1*(zmax-2)**2)

        A_high_y = np.sqrt(np.pi/2)*self.xi(-2)*y**-0.5

        A_low_y =  H(y/zmax**2) * zmax**(-6/3)*special.gamma(4/3)*special.hyperu(4/3, 4, zmax)
        A_low_y[zmax <= 1e-2] = ( H(y/zmax**2) * zmax**(-6/3)*special.gamma(4/3)*(27/(5*special.gamma(-5/3)*zmax**3) + 9/(2*special.gamma(-5/3)*zmax**2)))[zmax <= 1e-2]

        S1 = np.exp(-1e-7/zmax**2)

        return A_low_y*np.exp(-mu1*(y/min_x) * S1) + (1 + sigma*(y/( min_x))**-0.5)*A_high_y*(1-np.exp(-mu2*(y/min_x) *S1))
