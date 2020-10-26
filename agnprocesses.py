# This is the main python3 file for agnprocesses package.
"""
agnprocessec is the package for calculation of various astrophysical
processes which can occur in active galactic nuclei (but not only there).
It includes:
- synchrotron emission
- inverse Compton (IC) process
- photo-hadron process (photomeson production)
- Bethe-Heitler (BH) proton pair production on photon field
- and more
"""
# %% import
import processes.cosmology as cosmology
import processes.synchro as synchro
import processes.ic as ic
import processes.spectra as spec
import processes.pgamma as pgamma
###############################################################################
from astropy import units as u
from astropy import constants as const
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
import subprocess  # to run prompt scripts from python
from scipy.integrate import simps


if __name__ == '__main__':
    # en_cmb = np.logspace(-11, -2, 100) * u.eV
    # grey = spec.greybody_spectrum(en_cmb, 2.7 * u.K)
    # field3 = np.concatenate((en_cmb.value.reshape(en_cmb.shape[0], 1),
    #                          grey.value.reshape(grey.shape[0], 1)),
    #                         axis=1)
    # np.savetxt('CMB_Kelner_BH_100.txt', field3, fmt='%.6e')
    ###########################################################################
    # pgamma.pgamma_install()
    # import pgamma_ext
    eta0 = 0.313
    m_p = const.m_p.to(u.eV, u.mass_energy())
    print("m_p = {:.6e}".format(m_p))
    E_star = m_p * (m_p / (4.0 * const.k_B * 2.7 * u.K).to(u.eV)) * eta0
    print("E_star = {:.6e}".format(E_star))
    ###########################################################################
    # Repeat the SSC model from the Stability apply supplement file
    z = 0.3365
    d_l = cosmology.luminosity_distance(z).to(u.cm)
    print("d_l = {:.6e}".format(d_l))
    norm_e = 1.9e+40 * u.eV**(-1) * 0.6
    gamma1 = 1.9  # 1.69  #
    gamma2 = 4.5  # 4.29  #
    e_br = 9.0 * u.GeV * np.sqrt((1 + z)**3) * 0.8
    e_min_e = 5.0e+06 * u.eV  # 10.0**1.33 * (const.m_e * const.c**2).to(u.eV)
    e_max_e = 1.0e+12 * u.eV  # 10.0**6.82 * (const.m_e * const.c**2).to(u.eV)
    e_e = np.logspace(np.log10(e_min_e.value),
                      np.log10(e_max_e.value), 100) * u.eV
    doppler = 30.0  # 2.0 * 10.0**1.43
    r_b = 1.0e+17 * u.cm  # 10.0**16.46 * u.cm  #
    b = 0.05 * u.g**0.5 * u.cm**(-0.5) * u.s**(-1) / \
        ((1 + z)**3) * 1.5  # 10.0**(-1.01) * u.G  #
    print("B = {:.6e}".format(b))
    ###########################################################################
    # synchrotron
    nu = np.logspace(7, 19, 100) * u.Hz
    synchro_spec = synchro.derishev_synchro_spec(nu, b,
                                                 norm=norm_e,
                                                 spec_law='broken_power_law',
                                                 gamma1=gamma1,
                                                 gamma2=gamma2,
                                                 en_break=e_br,
                                                 en_min=e_min_e,
                                                 en_max=e_max_e)
    synchro_epsilon = ((const.h * nu).to(u.eV))
    synchro_density = synchro_spec / (4.0 / 3.0 * np.pi * r_b**2 * const.c.to(
        u.cm / u.s)
    )
    synchro_e = synchro_epsilon / (1.0 + z) * doppler
    synchro_sed = synchro_epsilon**2 * synchro_spec
    synchro_sed = synchro_sed * doppler**4 / (4.0 * np.pi * d_l**2)
    ###########################################################################
    # IC
    field = np.concatenate(
        (synchro_epsilon.value.reshape(synchro_epsilon.shape[0], 1),
         synchro_density.value.reshape(synchro_density.shape[0], 1)),
        axis=1
    )
    ic_e = np.logspace(0, 11.5, 100) * u.eV
    ic_spec = ic.inverse_compton_spec(ic_e,
                                      field,
                                      norm=norm_e,
                                      spec_law='broken_power_law',
                                      gamma1=gamma1,
                                      gamma2=gamma2,
                                      en_break=e_br,
                                      en_min=e_min_e,
                                      en_max=e_max_e,
                                      background_photon_energy_unit=synchro_epsilon.unit,
                                      background_photon_density_unit=synchro_density.unit)
    ic_sed = ic_e**2 * ic_spec
    ic_e *= doppler / (1.0 + z)
    ic_sed *= doppler**4 / (4.0 * np.pi * d_l**2)
    print("ic_sed.unit = {}".format(ic_sed.unit))
    ###########################################################################
    summ_e, summ_sed = spec.summ_spectra(synchro_e, synchro_sed, ic_e, ic_sed,
                                         nbin=100)
    ###########################################################################
    # proton target is the comptonized synchrotron (SSC) photon fields according to Troitsky
    # proton_target_e = ic_e / doppler * (1.0 + z)
    # proton_target_spec = ic_spec / (4.0 / 3.0 * np.pi * r_b**2 * const.c.to(
    #     u.cm / u.s)
    # )
    proton_target_e = summ_e / doppler * (1 + z)
    proton_target_spec = summ_sed / doppler**4 * (4.0 * np.pi * d_l**2)
    proton_target_spec = proton_target_spec / proton_target_e**2
    proton_target_spec = proton_target_spec / (4.0 / 3.0 * np.pi * r_b**2 *
                                               const.c.to(u.cm / u.s))
    proton_target = np.concatenate(
        (proton_target_e.value.reshape(proton_target_e.shape[0], 1),
         proton_target_spec.value.reshape(proton_target_spec.shape[0], 1)),
        axis=1
    )
    # proton_target_path = 'processes/c_codes/PhotoHadron/input/Troitsky_proton_target.txt'
    # np.savetxt(proton_target_path, proton_target, fmt='%.6e')
    energy_proton_min = 10.0 * u.TeV
    energy_proton_max = 6.0 * u.PeV
    en_p = energy_proton_min.unit * \
        np.logspace(np.log10(energy_proton_min.value),
                    np.log10(energy_proton_max.value),
                    100)
    p_p = 2.0
    C_p = 1.0e+11 * u.eV**(-1) * (4.0 * np.pi * d_l**2).value
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
    print("observable proton luminosity = {:.6e}".format(L_p))
    u_b = (b**2 / (8.0 * np.pi)).to(u.erg / u.cm**3)
    print("magnetic field density in the blob = {:.6e}".format(u_b))
    ###########################################################################
    # pgamma_ext.pgamma(proton_target_path,
    #                   energy_proton_min.value,
    #                   energy_proton_max.value,
    #                   p_p, -1)
    neutrino_e, neutrino_sed, _, _, _, _ = \
        pgamma.kelner_pgamma_calculate(proton_target,
                                       energy_proton_min,
                                       energy_proton_max,
                                       p_p, e_cut_p=-1,
                                       C_p=C_p)
    # neutrino_e = x[0]
    # neutrino_sed = x[1]
    # neutrino = np.loadtxt(
    #     'processes/c_codes/PhotoHadron/output/neutrino_SED.txt')
    neutrino_e = neutrino_e * doppler / (1.0 + z)
    neutrino_sed = neutrino_sed * doppler**4 / (1.0 + z)
    neutrino_sed = neutrino_sed / (4.0 * np.pi * d_l**2)
    ###########################################################################
    neutrino2 = np.loadtxt(
        'processes/c_codes/PhotoHadron/output/neutrino_SED_comptonized_synchro.txt')
    neutrino2_e = neutrino2[:, 0] * doppler / (1.0 + z)
    neutrino2_sed = neutrino2[:, 1] * doppler**4 / (1.0 + z)
    neutrino2_sed = neutrino2_sed * C_p / (4.0 * np.pi * d_l**2)
    ###########################################################################
    # Data from Science: gamma-rays
    data = np.loadtxt(
        'data/science-2017_gamma_flare_electromagnetic_component_v2.txt')
    data_en = data[:, 0]
    data_sed = data[:, 1]
    data_low = data[:, 3]
    data_up = data[:, 2]
    yerr = [data_low, data_up]
    ###########################################################################
    # Data from Science: neutrinos
    # observation of the neutriono flare
    science_flare_neutrino_intensity = 1.8e-10 / \
        1.602e-12 * u.eV / (u.cm**2 * u.s)
    I_min = 2.0e-02 * science_flare_neutrino_intensity  # pieces
    I_max = 2.5e+00 * science_flare_neutrino_intensity  # pieces
    I_med = 2.0e-01 * science_flare_neutrino_intensity  # pieces
    E_neu_min = 2.0e+14 * u.eV
    E_neu_max = 7.5e+15 * u.eV
    E_neu_med = 3.11e+14 * u.eV
    C_neu_min = I_min
    C_neu_max = I_max
    C_neu_med = I_med
    E_neu_min_max_long = np.logspace(np.log10(E_neu_min.value),
                                     np.log10(E_neu_max.value)) * E_neu_min.unit
    C_neu_median_long = np.ones((50, 1)) * science_flare_neutrino_intensity
    C_neu_min_max_long = np.logspace(np.log10(C_neu_min.value),
                                     np.log10(C_neu_max.value)) * C_neu_min.unit
    E_neu_average_long = np.ones((50, 1)) * E_neu_med
    ###########################################################################
    # Compare with C code IC
    # c = np.loadtxt(
    # '/home/raylend/Science/agnprocesses/test_figures/IC_Khangulian_scilab/IC_Khangulian_scilab_CMB.txt')
    ###########################################################################
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(1, 1, 1)
    fig, ax = plt.subplots()

    plt.plot(
        summ_e, summ_sed,
        marker=None,
        linestyle='-',
        linewidth=3,
        color='k',
        label='summ'
    )
    # plt.plot(
    #     c[:, 0], c[:, 1],
    #     marker=None,
    #     linestyle='--',
    #     linewidth=3,
    #     color='g',
    #     label='scilab'
    # )
    plt.plot(
        synchro_e, synchro_sed,
        marker=None,
        linestyle='--',
        linewidth=3,
        color='r',
        label='synchrotron without SSA'
    )
    plt.plot(
        ic_e, ic_sed,
        marker=None,
        linewidth=3,
        linestyle='--',
        color='b',
        label='SSC'
    )
    plt.errorbar(data_en, data_sed,
                 yerr=yerr, xerr=None, fmt='o', linewidth=0, elinewidth=2,
                 capsize=1, barsabove=False, markersize=5,
                 errorevery=1, capthick=1, color='r', zorder=100.0)
    plt.plot(
        neutrino_e, neutrino_sed,
        marker=None,
        linewidth=3,
        color='c',
        linestyle='-.',
        label='muon and antimuon neutrino'
    )
    plt.plot(
        E_neu_med, science_flare_neutrino_intensity,
        color='m',
        marker='s',
        markersize=5,
        linewidth=0
    )
    plt.plot(
        E_neu_average_long, C_neu_min_max_long,
        linewidth=3,
        linestyle='-',
        color='m'
    )
    plt.plot(
        E_neu_min_max_long, C_neu_median_long,
        linewidth=3,
        linestyle='-',
        color='m'
    )
    plt.plot(
        neutrino2_e, neutrino2_sed,
        marker=None,
        linewidth=3,
        color='g',
        linestyle=':',
        label='2'
    )
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel('energy, ' + str(ic_e.unit), fontsize=18)
    plt.xticks(fontsize=12)
    plt.ylabel('SED, ' + str(ic_sed.unit), fontsize=18)
    plt.yticks(fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(ticker.LogLocator(
        base=10.0, numticks=26, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]))
    # ax.set_xlim(1.0e+04, 1.0e+05)
    # ax.set_ylim(1.0e-09, 1.0e-07)
    # ax.grid()
    # ax.grid()
    plt.legend(loc='lower left')
    # fig.savefig('test_figures/exponential_cutoff_compare_with_Derishev_fig4a.pdf')
    # plt.legend()
    plt.show()
    ############################################################################
