"""
This program contains functions for various types of
energetic particles spectra.
"""
from astropy import units as u
from astropy import constants as const
import numpy as np


def power_law(en,
              gamma,
              norm=1.0 * u.eV**(-1),
              en_ref=1.0 * u.eV):
    return (norm * (en / en_ref)**(-gamma))


def broken_power_law(en, gamma1, gamma2, en_break, norm=1.0 * u.eV**(-1)):

    en1 = en[en <= en_break]
    en2 = en[en > en_break]

    f1 = (en1 / en_break)**(-gamma1)
    f2 = (en2 / en_break)**(-gamma2)
    f = np.concatenate((f1, f2), axis=0) * norm

    return f


def exponential_cutoff(en,
                       gamma,
                       en_cutoff,
                       norm=1.0 * u.eV**(-1),
                       en_ref=1.0 * u.eV):
    return (norm * (en / en_ref)**(-gamma) * np.exp(-en / en_cutoff))


def greybody_spectrum(en,
                      temperature,
                      dilution=1.0):
    """
    returns greybody (blackbody for dilution = 1.0) spectrum in
    1 / (eV cm^3) units
    """
    list_of_energies = ['J', 'erg', 'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV']
    ###########################################################################
    try:
        tu = temperature.unit
        if tu == 'K':
            e_char = (const.k_B * temperature).to('J')
        if tu == 'deg_C':
            e_char = (const.k_B * (temperature.value - 273.15) * u.K).to('J')
        if tu in list_of_energies:
            e_char = temperature.to('J')
        else:
            raise ValueError("Invalid value of temperature. \n It must be K, deg_C or one of {}".format(
                list_of_energies
            ))
    except:
        raise ValueError("Invalid value of temperature. \n It must be K, deg_C or one of {}".format(
            list_of_energies
        ))
    ###########################################################################
    nu_char = e_char / const.h
    try:
        if en.unit in list_of_energies:
            energy = en.to('eV')
            nu = en.to('J') / const.h
        elif en.unit == 'Hz':
            nu = en
            energy = (const.h * nu).to('eV')
        else:
            raise ValueError("Invalid unit of en. It must be Hz or one of {}".
                             format(list_of_energies))
    except:
        raise ValueError("Invalid unit of en. It must be Hz or one of {}".
                         format(list_of_energies))
    ###########################################################################
    x = (nu / nu_char).decompose()
    s = (8 * np.pi**2 * const.h * nu**3) / const.c**3 / (np.exp(x) - 1)
    s /= const.h  # cm^{-3}
    s /= energy  # [1 / (cm^3 eV)]
    s *= dilution
    s = s.to(1.0 / (u.eV * u.cm**3))
    ###########################################################################
    return s


def test():
    print("spectra.py imported successfully.")
    return None
