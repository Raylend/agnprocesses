"""
This program contains functions for various types of
energetic particles spectra.
"""
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
from scipy import interpolate


def power_law(en,
              gamma,
              norm=1.0 * u.eV**(-1),
              en_ref=1.0 * u.eV):
    f = (norm * (en / en_ref)**(-gamma))
    try:
        f = f.to(norm.unit)
    except AttributeError:
        pass
    return f


def broken_power_law(en, gamma1, gamma2, en_break, norm=1.0 * u.eV**(-1)):

    en1 = en[en <= en_break]
    en2 = en[en > en_break]
    f1 = (en1 / en_break)**(-gamma1)
    f2 = (en2 / en_break)**(-gamma2)
    f = np.concatenate((f1, f2), axis=0) * norm
    try:
        f = f.to(norm.unit)
    except AttributeError:
        pass
    return f


def exponential_cutoff(en,
                       gamma,
                       en_cutoff,
                       norm=1.0 * u.eV**(-1),
                       en_ref=1.0 * u.eV):
    f = norm * (en / en_ref)**(-gamma) * np.exp(-en / en_cutoff)
    try:
        f = f.to(norm.unit)
    except AttributeError:
        pass
    return f


def greybody_spectrum(en,
                      temperature,
                      dilution=1.0):
    """
    returns greybody (blackbody for dilution = 1.0) spectrum in
    1 / (eV cm^3) units

    en is energy or frequency of photons with 'J', 'erg', 'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV' or 'Hz' units (float or array-like)

    temperature is astropy Quantity with 'K', 'deg_C' or 'J', 'erg', 'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV' unit

    dilution is float (dimensionless)
    """
    list_of_energies = ['J', 'erg', 'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV']
    ###########################################################################
    tu = str(temperature.unit)
    try:
        if tu == 'K':
            e_char = (const.k_B * temperature).to('J')
        elif tu == 'deg_C':
            e_char = (const.k_B * (temperature.value - 273.15) * u.K).to('J')
        elif tu in list_of_energies:
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
        if str(en.unit) in list_of_energies:
            energy = en.to('eV')
            nu = en.to('J') / const.h
        elif str(en.unit) == 'Hz':
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
    s = (8 * np.pi * const.h * nu**3) / const.c**3 / (np.exp(x) - 1)
    s /= const.h  # cm^{-3}
    s /= energy  # [1 / (cm^3 eV)]
    s *= dilution
    s = s.to(1.0 / (u.eV * u.cm**3))
    ###########################################################################
    return s


def summ_spectra(e1, s1, e2, s2, nbin=100):
    """
    e1 is the energy numpy array of the 1st spectrum (SED) or array-like
    s1 is the intensity, SED and so on (1st spectrum), with the same size as e2
    e2 is the energy numpy array of the 2nd spectrum (SED)
    s2 is the intensity, SED and so on (2nd spectrum), with the same size as e1
    nbin is the number of bins in the final spectrum (SED)

    If e1 is an astropy Quantity, e2 should be an astopy Quantity.
    If s1 is an astropy Quantity, s2 should be an astopy Quantity.

    Final energy and spectrum (SED) has units of e1 and s1 correspondingly.
    """
    try:
        if e1.shape[0] != s1.shape[0]:
            raise ValueError("sizes of e1 and s1 must be equal!")
        if e2.shape[0] != s2.shape[0]:
            raise ValueError("sizes of e2 and s2 must be equal!")
    except AttributeError:
        raise AttributeError(
            "e1, s1, e2, s2 must be numpy arrays or array-like!")

    try:
        e2 = e2.to(e1.unit)
        e2 = e2.value
    except AttributeError:
        pass

    try:
        s2 = s2.to(s1.unit)
        s2 = s2.value
    except AttributeError:
        pass

    try:
        x_u = e1.unit
        e1 = e1.value
    except AttributeError:
        pass

    try:
        y_u = s1.unit
        s1 = s1.value
    except AttributeError:
        pass

    logx1 = np.log10(e1)
    logy1 = np.log10(s1)
    f1 = interpolate.interp1d(logx1, logy1,
                              kind='linear',
                              bounds_error=False,
                              fill_value=(0, 0))
    logx2 = np.log10(e2)
    logy2 = np.log10(s2)
    f2 = interpolate.interp1d(logx2, logy2,
                              kind='linear',
                              bounds_error=False,
                              fill_value=(0, 0))
    emin = np.min((np.min(e1), np.min(e2)))
    emax = np.max((np.max(e1), np.max(e2)))
    e = np.logspace(np.log10(emin), np.log10(emax), nbin)
    x = np.log10(e)
    s = (10.0**(f1(x)) + 10.0**(f2(x))) * y_u  # new summed spectrum (SED)
    e = e * x_u  # new energy
    return((e, s))


def test():
    print("spectra.py imported successfully.")
    return None
