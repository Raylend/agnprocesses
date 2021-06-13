from astropy import units as u
from astropy import constants as const
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
from scipy.integrate import simps

from agnprocesses import bh
from agnprocesses import cosmology
from agnprocesses import spectra as spec


if __name__ == '__main__':
    # Repeat the external photon field model vAPR47
    # Geometry
    doppler = 30.0
    Gamma = doppler / 2.0
    r_b = 1.0e+17 * u.cm
    z = 0.3365
    d_l = cosmology.luminosity_distance(z).to(u.cm)
    print("d_l = {:.6e}".format(d_l))
    b = 69.0 * u.g**0.5 * u.cm**(-0.5) * u.s**(-1)
    print("B = {:.6e}".format(b))
    ############################################################################
    # External photon field
    en_ext = np.logspace(np.log10(1.33), np.log10(10.0), 100) * u.eV
    en_ext_blob = 4.0 / 3.0 * Gamma * en_ext
    boost = 2.0 * 4.0 / 3.0 * Gamma**2
    alpha = 2.0
    K = 3e+03 / (u.eV * u.cm**3)
    n_ext = spec.power_law(en_ext, alpha, norm=K)
    n_ext_blob = n_ext * boost
    field_ext = np.concatenate(
        (en_ext_blob.value.reshape(en_ext_blob.shape[0], 1),
         n_ext_blob.value.reshape(n_ext_blob.shape[0], 1)),
        axis=1)
    ############################################################################
    norm_e = 3.8e+38 * u.eV**(-1)
    gamma1 = 1.9
    gamma2 = 4.3
    e_br = 2.0e+08 * u.eV
    e_min_e = 5.0e+06 * u.eV
    e_max_e = 1.0e+12 * u.eV
    e_e = np.logspace(np.log10(e_min_e.to(u.eV).value),
                      np.log10(e_max_e.to(u.eV).value), 100) * u.eV
    ###########################################################################

    ###########################################################################
    energy_proton_min = 1.0e+14 * u.eV  # 3.0e+14 * u.eV
    energy_proton_max = 6.0e+14 * u.eV
    en_p = energy_proton_min.unit * \
        np.logspace(np.log10(energy_proton_min.to(u.eV).value),
                    np.log10(energy_proton_max.to(u.eV).value),
                    100)
    p_p = 2.0
    C_p = 1.5e+66 / 6.0 * u.eV**(-1)
    print("C_p = {:.6e}".format(C_p))
    ###########################################################################
    proton_spectrum = spec.power_law(en_p, p_p, norm=C_p, en_ref=1.0 * u.eV)
    u_p = simps(proton_spectrum.value * en_p.value, en_p.value) * \
        proton_spectrum.unit * \
        en_p.unit**2 / \
        (4.0 / 3.0 * np.pi * r_b**3)
    u_p = u_p.to(u.erg / u.cm**3)
    print("proton energy density in the blob = {:.6e}".format(u_p))
    L_p = np.pi * r_b**2 * const.c.cgs * (doppler / 2.0)**2 * u_p
    print("observable proton luminosity in the lab frame = {:.6e}".format(L_p))
    u_b = (b**2 / (8.0 * np.pi)).to(u.erg / u.cm**3)
    print("magnetic field density in the blob = {:.6e}".format(u_b))
    # Bethe-Heitler process
    bh_pair_e, bh_pair_sed = bh.kelner_bh_calculate(field_ext,
                                                    energy_proton_min,
                                                    energy_proton_max,
                                                    p_p, e_cut_p=-1,
                                                    C_p=C_p)
