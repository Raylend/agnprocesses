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

from astropy.units import Quantity
from typing import Union, Optional
from numbers import Number

import agnprocesses.ext.bh as bh_ext
from .data_files import get_io_paths
from .data_classes import SpatialSpectralPhotonDensity, SEDPerTime


inpath, outpath = get_io_paths('bh_ext_io')


def lorentz_to_energy(particle_mass) -> 'astopy Equivalence':  # type: ignore
    return (
        u.dimensionless_unscaled,  # lorentz factor
        u.J,
        lambda gamma: gamma * particle_mass * const.c**2,
        lambda E: E / particle_mass * const.c**2, 
    )


def kelner_bh_calculate(
    E_proton_min: Union[Number, Quantity],
    E_proton_max: Union[Number, Quantity],
    background_photon_field: SpatialSpectralPhotonDensity,
    proton_spectrum_gamma: float,
    proton_spectrum_E_cut: Optional[Quantity] = None,
    proton_spectrum_norm: Quantity = 1.0 / (u.eV),
) -> SEDPerTime:
    """SED of electrons and positrons produced at interactions of relativistic protons with low energy photon
    background:

        p + γ → p + e− + e+

    Implemented following Kelner, S. R., & Aharonian, F. A. (2008). Energy spectra of gamma rays, electrons,
    and neutrinos produced at interactions of relativistic protons with low energy radiation.
    Physical Review D, 78(3). https://doi.org/10.1103/physrevd.78.034013

    Args:
        E_proton_min: minimum proton energy, astropy.Quantity in energy units OR float, interpreted as Lorentz factor
        E_proton_max: maximum proton energy, same types
        background_photon_field (SpatialSpectralPhotonDensity): self-explainatory
        proton_spectrum_gamma (float): power law proton spectrum's power
        proton_spectrum_E_cut (Quantity, optional): energy of optional exponential cut in proton spectrum.
                                                    Defaults to None (no cut).
        proton_spectrum_norm (Quantity, optional): normalizing constant of proton spectrum (dN/dE).
                                                   Defaults to 1.0/(u.eV).

    Raises:
        NotImplementedError: for background_photon_field with size > 100 (temporary restriction)

    Returns:
        SEDPerTime: resulting electron+positron spectral energy distribtion
    """

    if background_photon_field.size > 100:
        raise NotImplementedError("Background photon field grid is currently limited at 100 elements")
    photon_field_path = str((inpath / "field.txt").resolve())
    np.savetxt(
        photon_field_path,
        # units are set as expected by extension!
        background_photon_field.to_numpy(E_unit=u.eV, q_unit=(u.eV * u.cm ** 3)**(-1)),
        fmt="%.6e"
    )

    def E_from_E_or_lorentz(EorL: Union[Number, Quantity]) -> Quantity:
        if not isinstance(EorL, Quantity):
            EorL = EorL * u.dimensionless_unscaled
        return EorL.to(u.eV, equivalencies=[lorentz_to_energy])

    E_proton_min = E_from_E_or_lorentz(E_proton_min)
    E_proton_max = E_from_E_or_lorentz(E_proton_max)

    if proton_spectrum_E_cut is None:
        proton_spectrum_E_cut = -1 * u.dimensionless_unscaled  # as expected by extension
    else:
        proton_spectrum_E_cut = E_from_E_or_lorentz(proton_spectrum_E_cut)

    output_path = str((outpath / "BH_SED.txt").resolve())

    bh_ext.bh(
        photon_field_path,
        output_path,
        E_proton_min.value,
        E_proton_max.value,
        proton_spectrum_gamma,
        proton_spectrum_E_cut.value,
    )

    sed = SEDPerTime.from_txt(output_path, E_unit=u.eV, q_unit=(u.eV * u.s**(-1)))
    sed.q *= proton_spectrum_norm / (1.0 / u.eV)  # ??? applying initial normalization at last ???
    return sed
