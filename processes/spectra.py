"""
This program contains functions for various types of
energetic particles spectra.
"""
from astropy import units as u
from astropy import constants as const
import numpy as np

def broken_power_law(en, gamma1, gamma2, en_break, norm = 1.0 * u.eV**(-1)):

    en1 = en[en <= en_break]
    en2 = en[en >  en_break]

    f1 = (en1 / en_break)**(-gamma1)
    f2 = (en2 / en_break)**(-gamma2)
    f = np.concatenate((f1, f2), axis = 0) * norm

    return f










def test():
    print("spectra.py imported successfully.")
    return None
