"""
This program computes synchrotron emission from energetic particles
according to Derishev & Aharonian (2019)
https://doi.org/10.3847/1538-4357/ab536a
"""
from astropy import units as u
from astropy import constants as const
import numpy as np
try:
    import processes.spectra as spec
except:
    try:
        import spectra as spec
    except:
        print("Problems with importing spectra.py occured")
        raise ModuleNotFoundError

def derishev_q_function(x):
    """
    This is Derishev's Q_{PL} function
    see his eq. (10) and (41)
    """
    return (1.0 + x**(-2.0/3.0)) * np.exp(-2.0 * x**(2.0/3.0))

def derishev(nu, en, b = 1.0 * u.G):
    try:
        g = en.to('eV') / (const.m_e.cgs * const.c.cgs**2).to('eV')
    except AttributeError:
        print("energy must be associated with an astropy unit")
        raise AttributeError
    omega_0 = 4.0/3.0 * g**2 * const.e.cgs * b.cgs / (const.m_e.cgs * const.c.cgs)
    nu_0 = nu / nu_0
    f = const.alpha / (3.0 * g**2) * derishev_q_function(x)
    if any(f.value[f.value <= 0]):
        print("Arithmetic exception in derishev(...) function")
        raise ArithmeticError
    return f

if __name__ == '__main__':
    spec.test()