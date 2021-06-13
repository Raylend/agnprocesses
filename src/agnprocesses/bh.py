"""
This program computes (electron + positron) Bethe-Heitler SED
produced via proton-photon collisions using T. A. Dzhatdoev's C++ codes.
It includes only electron + positron pair production,
it does not include photomeson production and its secondaries.
For the latter process see pgamma.py file in this package.
See Kelner, & Aharonian Phys. Rev. D 78, 034013 (2008).
"""

from astropy import units as u
from astropy import constants as const
import numpy as np
from pathlib import Path

import agnprocesses.ext.bh as bh_ext


CUR_DIR = Path(__file__).parent
BH_EXT_DIR = CUR_DIR / 'bh_ext_io'
EXT_INPUT = BH_EXT_DIR / 'input'
EXT_OUTPUT = BH_EXT_DIR / 'output'

for dir in [BH_EXT_DIR, EXT_INPUT, EXT_OUTPUT]:
    dir.mkdir(exist_ok=True)

def kelner_bh_calculate(field,
                        energy_proton_min,
                        energy_proton_max,
                        p_p,
                        e_cut_p=-1,
                        C_p=1.0 / (u.eV),
                        background_photon_energy_unit=u.eV,
                        background_photon_density_unit=(u.eV * u.cm**3)**(-1)):
    """
    energy_proton_min is the minimum proton energy
    (must be an astropy Quantity of energy or float (in the latter case it will
    be considered as Lorentz factor))

    energy_proton_max is the maximum proton energy
    (must be the same type as energy_proton_min)

    e_cut_p is the cutoff proton energy. It must be an astropy Quantity or
    float (in the latter case it will be considered as Lorentz factor).
    If it is less than 0 (default), an ordinary power law spectrum will be used.

    field is the string with the path to the target photon field .txt file
    OR numpy array with 2 colums: the first column is the background photon
    energy, the second columnn is the background photon density.
    Units in the field table must correspond to the
    background_photon_energy_unit parameter and the
    background_photon_density_unit parameter.

    field should contain no more than 100 strings (rows)!!!
    (more strings will be implemented in future)

    C_p is the normalization coefficient of the proton spectrum.

    Returns a tuple with electron + positron energy in
    eV and SED in eV * s**(-1) as array-like astropy Quantities.
    """
    try:
        energy_coef = background_photon_energy_unit.to(u.eV) / (1.0 * u.eV)
        dens_coef = background_photon_density_unit.to(
            (u.eV * u.cm**3)**(-1)) / (u.eV * u.cm**3)**(-1)
    except AttributeError:
        raise AttributeError(
            "Make sure that background_photon_energy_unit is in energy units, "
            + "background_photon_density_unit is in [energy * volume]**(-1) units."
        )

    if isinstance(field, str):
        try:
            field = np.loadtxt(field)
            field[:, 0] = field[:, 0] * energy_coef
            field[:, 1] = field[:, 1] * dens_coef
        except:
            raise ValueError(
                "Cannot read 'field'! Make sure it is a numpy array \n"
                + " with 2 columns or a string with the path to a .txt file with \n"
                + " 2 columns (energy / density).\n"
                + "Try to use an absolute path."
            )
    elif isinstance(field, np.ndarray):  # TODO: there may be a better way of checking this
        field[:, 0] = field[:, 0] * energy_coef
        field[:, 1] = field[:, 1] * dens_coef
    else:
        raise ValueError(
            "Invalid value of 'field'! Make sure it is a numpy array \n"
            + " with 2 columns or a string with the path to a .txt file with \n"
            + " 2 columns (energy / density)."
        )

    if field[:, 0].shape[0] > 100:
        raise NotImplementedError(
            "field should contain no more than 100 strings (rows)! (more strings will be implemented in future)"
        )
    
    photon_field_path = str((EXT_INPUT / 'field.txt').resolve())
    output_path = str((EXT_OUTPUT / 'BH_SED.txt').resolve())

    np.savetxt(photon_field_path, field, fmt='%.6e')

    try:
        energy_proton_min = energy_proton_min.to(u.eV)
        energy_proton_max = energy_proton_max.to(u.eV)
    except AttributeError:
        try:
            energy_proton_min = (energy_proton_min * const.m_p * const.c**2).to(
                u.eV)
            energy_proton_max = (energy_proton_max * const.m_p * const.c**2).to(
                u.eV)
        except:
            raise ValueError(
                "Problems with energy_proton_min and energy_proton_max!\n"
                + "Make sure they are astropy Quantities or float!"
            )
    try:
        e_cut_value = e_cut_p.value
    except AttributeError:
        if e_cut_p < 0:
            e_cut_p = e_cut_p * u.dimensionless_unscaled
        else:
            e_cut_p = (e_cut_p * const.m_p * const.c**2).to(u.eV)

    bh_ext.bh(
        photon_field_path,
        output_path,
        energy_proton_min.value,
        energy_proton_max.value,
        p_p, e_cut_p.value
    )

    pair = np.loadtxt(output_path)
    pair_e = pair[:, 0] * u.eV
    pair_sed = pair[:, 1] * (u.eV * u.s**(-1))
    pair_sed = pair_sed * C_p / (1.0 / u.eV)
    return (pair_e, pair_sed)
