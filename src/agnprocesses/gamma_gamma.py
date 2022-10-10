"""
This program computes (electron + positron) SED
produced via gamma-ray - photon collisions using
E. I. Podlesnyi's C codes.
"""

import subprocess  # to run prompt scripts from python
from astropy import units as u
import numpy as np

import agnprocesses.ext.ggir as ggir_ext
import agnprocesses.ext.ggpp as ggpp_ext
from .data_files import get_io_paths


ggpp_in, ggpp_out = get_io_paths('ggpp_ext_io')


def pair_production(field,
                    gamma,
                    background_photon_energy_unit=u.eV,
                    background_photon_density_unit=(u.eV * u.cm**3)**(-1),
                    gamma_energy_unit=u.eV,
                    gamma_sed_unit=u.eV / (u.cm**2 * u.s)):
    """
    field is the string with the path to the target photon field .txt file
    OR numpy array with 2 columns: the first column is the background photon
    energy, the second columnn is the background photon density.
    Units in the field table must correspond to the
    background_photon_energy_unit parameter and the
    background_photon_density_unit parameter.
    field should contain no more than 1000 strings (rows)!!!
    (more strings will be implemented in future)

    gamma is the string with the path to the gamma-ray energy/SED .txt file
    OR numpy array with 2 colums: the first column is the gamma-ray
    energy, the second columnn is the gamma-ray SED.
    NB: gamma should contain SED of gamma-rays to be absorbed, i.e. they should
    be multiplied by (1 - exp(-tau))!
    Units in the gamma table must correspond to the gamma_energy_unit parameter
    and the gamma_sed_unit parameter.
    gamma mustn't contain more than 5000 strings (rows)!
    This restriction will be removed in future updates.

    Returns a tuple with electron + positron energy in
    gamma_energy_unit and SED in gamma_sed_unit
    as array-like astropy Quantities.
    """
    try:
        energy_coef = background_photon_energy_unit.to(u.eV) / (1.0 * u.eV)
        dens_coef = background_photon_density_unit.to(
            (u.eV * u.cm**3)**(-1)) / (u.eV * u.cm**3)**(-1)
    except AttributeError:
        raise AttributeError(
            "Make sure that background_photon_energy_unit is in energy units, background_photon_density_unit is in [energy * volume]**(-1) units.")
    ###########################################################################
    try:
        energy_gamma_coef = gamma_energy_unit.to(u.eV) / (1.0 * u.eV)
    except AttributeError:
        raise AttributeError(
            "Make sure that gamma_energy_unit is in energy units.")
    # background photon field
    if type(field) == type(''):
        try:
            field = np.loadtxt(field)
            field[:, 0] = field[:, 0] * energy_coef
            field[:, 1] = field[:, 1] * dens_coef
        except:
            raise ValueError(
                "Cannot read 'field'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / density).\nTry to use an absolute path.")
    elif type(field) == type(np.array(([2, 1], [5, 6]))):
        field[:, 0] = field[:, 0] * energy_coef
        field[:, 1] = field[:, 1] * dens_coef
    else:
        raise ValueError(
            "Invalid value of 'field'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / density).")
    if field[:, 0].shape[0] > 7000:
        raise NotImplementedError(
            "field should contain no more than 7000 strings (rows)! (more strings will be implemented in future)")
    photon_path = str(ggpp_in / 'photon_path.txt')
    np.savetxt(photon_path, field, fmt='%.6e')
    ###########################################################################
    # gamma-ray SED
    if type(gamma) == type(''):
        try:
            gamma = np.loadtxt(gamma)
            gamma[:, 0] = gamma[:, 0] * energy_gamma_coef
            gamma[:, 1] = gamma[:, 1]
        except:
            raise ValueError(
                "Cannot read 'gamma'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / SED).\nTry to use an absolute path.")
    elif type(gamma) == type(np.array(([2, 1], [5, 6]))):
        gamma[:, 0] = gamma[:, 0] * energy_gamma_coef
        gamma[:, 1] = gamma[:, 1]
    else:
        raise ValueError(
            "Invalid value of 'gamma'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / sed).")
    if gamma[:, 0].shape[0] > 5000:
        raise NotImplementedError(
            "gamma should contain no more than 5000 strings (rows)! (more strings will be implemented in future)")
    gamma_path = str(ggpp_in / 'gamma_path.txt')
    np.savetxt(gamma_path, gamma, fmt='%.6e')
    output_path = str(ggpp_out / 'SED_gamma-gamma_pairs.txt')
    ###########################################################################
    ggpp_ext.pair(photon_path, gamma_path, output_path)
    pair = np.loadtxt(output_path)
    pair_e = (pair[:, 0] * u.eV).to(gamma_energy_unit)
    pair_sed = pair[:, 1] * gamma_sed_unit
    return (pair_e, pair_sed)


ggir_in, ggir_out = get_io_paths('ggir_ext_io')


def interaction_rate(field,
                     e_min,
                     e_max,
                     background_photon_energy_unit=u.eV,
                     background_photon_density_unit=(u.eV * u.cm**3)**(-1)):
    """
    field is the string with the path to the target photon field .txt file
    OR numpy array with 2 columns: the first column is the background photon
    energy, the second columnn is the background photon density.
    Units in the field table must correspond to the
    background_photon_energy_unit parameter and the
    background_photon_density_unit parameter.
    field should contain no more than 100 strings (rows)!!!
    (more strings will be implemented in future)

    e_min, e_max are minimum and maximum gamma-ray energy correspondingly.
    They must be in astropy energy units.

    Returns an tuple of 2 numpy columns: energy and interaction rate
    """
    try:
        e_min = e_min.to(u.eV)
    except:
        raise ValueError("e_min must be an energy astropy Quantity!")
    try:
        e_max = e_max.to(u.eV)
    except:
        raise ValueError("e_max must be an energy astropy Quantity!")
    ###########################################################################
    try:
        energy_coef = background_photon_energy_unit.to(u.eV) / (1.0 * u.eV)
        dens_coef = background_photon_density_unit.to(
            (u.eV * u.cm**3)**(-1)
        ) / (u.eV * u.cm**3)**(-1)
    except AttributeError:
        raise AttributeError(
            "Make sure that background_photon_energy_unit is in energy units, background_photon_density_unit is in [energy * volume]**(-1) units.")
    ###########################################################################
    # background photon field
    if type(field) == type(''):
        try:
            field = np.loadtxt(field)
            field[:, 0] = field[:, 0] * energy_coef
            field[:, 1] = field[:, 1] * dens_coef
        except:
            raise ValueError(
                "Cannot read 'field'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / density).\nTry to use an absolute path.")
    elif type(field) == type(np.array(([2, 1], [5, 6]))):
        field[:, 0] = field[:, 0] * energy_coef
        field[:, 1] = field[:, 1] * dens_coef
    else:
        raise ValueError(
            "Invalid value of 'field'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / density).")
    if field[:, 0].shape[0] > 7000:
        raise NotImplementedError(
            "field should contain no more than 7000 strings (rows)! (more strings will be implemented in future)")
    os.makedirs(str(ggir_in), exist_ok=True)
    os.makedirs(str(ggir_out), exist_ok=True)
    photon_path = str(ggir_in / 'photon_field.txt')
    output_path = str(ggir_out / 'gamma-gamma_interaction_rate.txt')
    np.savetxt(photon_path, field, fmt='%.6e')
    ggir_ext.rate(photon_path, output_path, e_min.value, e_max.value)
    inter = np.loadtxt(output_path)
    inter_e = inter[:, 0] * u.eV
    inter_rate = inter[:, 1] * u.cm**(-1)
    return (inter_e, inter_rate)
