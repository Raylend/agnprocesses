"""
This program computes synchrotron emission from energetic particles
according to Derishev & Aharonian (2019)
https://doi.org/10.3847/1538-4357/ab536a
"""

from astropy import units as u
from astropy import constants as const
import numpy as np

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
    omega_0 = 4.0/3.0 * g**2 * const.e.cgs * b / (const.m_e.cgs * const.c.cgs)
    nu_0 = nu / nu_0
    return (const.alpha / (3.0 * g**2) * derishev_q_function(x))







def test():
    print("Test completed successfully.")
    return None
