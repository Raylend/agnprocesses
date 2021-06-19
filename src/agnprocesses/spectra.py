"""
This program contains functions for various types of
energetic particles spectra.
"""
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy import interpolate


def power_law(en, gamma, norm=1.0 * u.eV ** (-1), en_ref=1.0 * u.eV):
    f = norm * (en / en_ref) ** (-gamma)
    try:
        f = f.to(norm.unit)
    except AttributeError:
        pass
    return f


def broken_power_law(en, gamma1, gamma2, en_break, norm=1.0 * u.eV ** (-1)):
    en1 = en[en <= en_break]
    en2 = en[en > en_break]
    f1 = (en1 / en_break) ** (-gamma1)
    f2 = (en2 / en_break) ** (-gamma2)
    f = (np.concatenate((f1, f2), axis=0) * norm).reshape(en.shape)
    try:
        f = f.to(norm.unit)
    except AttributeError:
        pass
    return f


def log_parabola(en, alpha, beta, en_ref=1.0 * u.eV, norm=1.0 * u.eV ** (-1)):
    try:
        x = (en / en_ref).to(u.dimensionless_unscaled)
    except AttributeError:
        x = en / en_ref
    f = norm * (en / en_ref) ** (-(alpha + beta * np.log(x)))
    try:
        f = f.to(norm.unit)
    except AttributeError:
        pass
    return f


def exponential_cutoff(en, gamma, en_cutoff, norm=1.0 * u.eV ** (-1), en_ref=1.0 * u.eV):
    f = norm * (en / en_ref) ** (-gamma) * np.exp(-en / en_cutoff)
    try:
        f = f.to(norm.unit)
    except AttributeError:
        pass
    return f


def greybody_spectrum(en,
                      temperature,
                      dilution=1.0):
    # this function was cross-checked
    """
    returns greybody (blackbody for dilution = 1.0) spectrum in
    1 / (eV cm^3) units

    en is energy or frequency of photons with 'J', 'erg', 'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV' or 'Hz' units (float or array-like)

    temperature is astropy Quantity with 'K', 'deg_C' or 'J', 'erg', 'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV' unit

    dilution is float (dimensionless)
    """
    list_of_energies = ["J", "erg", "eV", "keV", "MeV", "GeV", "TeV", "PeV"]
    ###########################################################################
    tu = str(temperature.unit)
    try:
        if tu == "K":
            e_char = (const.k_B * temperature).to("J")
        elif tu == "deg_C":
            e_char = (const.k_B * (temperature.value - 273.15) * u.K).to("J")
        elif tu in list_of_energies:
            e_char = temperature.to("J")
        else:
            raise ValueError(
                "Invalid value of temperature. \n It must be K, deg_C or one of {}".format(
                    list_of_energies
                )
            )
    except:
        raise ValueError(
            "Invalid value of temperature. \n It must be K, deg_C or one of {}".format(
                list_of_energies
            )
        )
    ###########################################################################
    nu_char = e_char / const.h
    try:
        if str(en.unit) in list_of_energies:
            energy = en.to("eV")
            nu = en.to("J") / const.h
        elif str(en.unit) == "Hz":
            nu = en
            energy = (const.h * nu).to("eV")
        else:
            raise ValueError(
                "Invalid unit of en. It must be Hz or one of {}".format(list_of_energies)
            )
    except:
        raise ValueError("Invalid unit of en. It must be Hz or one of {}".format(list_of_energies))
    ###########################################################################
    x = (nu / nu_char).to(u.dimensionless_unscaled)  # decompose()
    s = (8 * np.pi * const.h * nu**3) / const.c**3 / (np.exp(x) - 1)
    s = s / const.h  # cm^{-3}
    s = s / energy  # [1 / (cm^3 eV)]
    s = s * dilution
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
    x_u = 1
    y_u = 1
    try:
        if e1.shape[0] != s1.shape[0]:
            raise ValueError("sizes of e1 and s1 must be equal!")
        if e2.shape[0] != s2.shape[0]:
            raise ValueError("sizes of e2 and s2 must be equal!")
    except AttributeError:
        raise AttributeError("e1, s1, e2, s2 must be numpy arrays or array-like!")

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
    f1 = interpolate.interp1d(
        logx1, logy1, kind="linear", bounds_error=False, fill_value=(-40, -40)
    )
    logx2 = np.log10(e2)
    logy2 = np.log10(s2)
    f2 = interpolate.interp1d(
        logx2, logy2, kind="linear", bounds_error=False, fill_value=(-40, -40)
    )
    emin = np.min((np.min(e1), np.min(e2)))
    emax = np.max((np.max(e1), np.max(e2)))
    e = np.logspace(np.log10(emin), np.log10(emax), nbin)
    x = np.log10(e)
    s = (10.0 ** (f1(x)) + 10.0 ** (f2(x))) * y_u  # new summed spectrum (SED)
    e = e * x_u  # new energy
    return (e, s)


def create_2column_table(col1, col2):
    """
    Creates a 2 column table as a numpy array with shape
    (np.max(col1.shape), 2). col2 must have the same shape as col1.
    Mind that if col1 or col2 are astropy Quantities they will be converted
    to their values, their units will be lost!
    """
    if col1.shape != col2.shape:
        raise ValueError("col1 and col2 must have the same shapes!")
    try:
        col1 = col1.value
        print("col1 has '{}' astropy Quantity unit".format(col1.unit))
    except AttributeError:
        pass
    try:
        col2 = col2.value
        print("col2 has '{}' astropy Quantity unit".format(col2.unit))
    except AttributeError:
        pass
    n = np.max(col1.shape)
    table = np.concatenate((col1.reshape(n, 1), col2.reshape(n, 1)), axis=1)
    return table


def to_current_energy(e, e1, s1):
    """
    Interpolates s1 to points of e array given that s1 corresponds to
    the points of e1. E.g., SED s1 corresponds to energy bins e1, but you want
    to interpolate it to elements of a numpy array e.

    e and e1 are not necessary energies.

    e is the numpy array of interest.

    s1 is the function (intensity, SED and so on) to be interpolated.

    e1 is the numpy array corresponding to the s1 array.

    If e is an astropy Quantity, e1 should be an astopy Quantity.

    Final energy and function have units of e and s1 correspondingly.
    """
    ###########################################################################
    try:
        if e1.shape[0] != s1.shape[0]:
            raise ValueError("sizes of e1 and s1 must be equal!")
    except AttributeError:
        raise AttributeError("e1, s1, e must be numpy arrays or array-like!")
    ###########################################################################
    x_u = None
    y_u = None
    try:
        x_u = e.unit
        e = e.value
    except AttributeError:
        pass
    try:
        e1 = e1.to(x_u)
        e1 = e1.value
    except AttributeError:
        if x_u is None:
            pass
        else:
            raise ValueError("e1 must be an astopy Quantity as e!")
    try:
        y_u = s1.unit
        s1 = s1.value
    except AttributeError:
        pass
    ###########################################################################
    logx1 = np.log10(e1)
    logy1 = np.log10(s1)
    f1 = interpolate.interp1d(
        logx1, logy1, kind="linear", bounds_error=False, fill_value=(-40, -40)
    )
    x = np.log10(e)
    s = 10.0 ** (f1(x))  # new interpolated function values
    ###########################################################################
    if y_u is not None:
        s = s * y_u

    return s
