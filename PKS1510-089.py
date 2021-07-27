# This is the main python3 file for agnprocesses package.
"""
agnprocesses is the package for calculation of various astrophysical
processes which can occur in active galactic nuclei (but not only there).
It includes:
- synchrotron emission
- inverse Compton (IC) process
- photo-hadron process (photomeson production)
- Bethe-Heitler (BH) proton pair production on photon field
- gamma-gamma pair production
- extragalactic high-energy-gamma-ray absorption
- and more
"""
# %% import
import processes.cosmology as cosmology
import processes.synchro as synchro
import processes.ic as ic
import processes.spectra as spec
import processes.ebl as ebl
# import processes.pgamma as pgamma
# import processes.bh as bh
import processes.gamma_gamma as gamma_gamma
#############################################################################
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
from scipy.optimize import curve_fit
from scipy import interpolate
#############################################################################
from test_pytorch import *


if __name__ == '__main__':
    tev1_01 = np.loadtxt("data/Timur_cascades/1TeV-0.1-1e2")
    tev3_01 = np.loadtxt("data/Timur_cascades/3TeV-0.1-1e4")
    tev10_01 = np.loadtxt("data/Timur_cascades/10TeV-0.1-3e3")
    tev1_01[:, 1][tev1_01[:, 1] == 0.0] = 1.0e-20
    tev3_01[:, 1][tev3_01[:, 1] == 0.0] = 1.0e-20
    tev10_01[:, 1][tev10_01[:, 1] == 0.0] = 1.0e-20
    e_Timur = np.log10(tev1_01[:, 0])  # x
    e0_Timur = np.log10(np.array([1.0, 3.0, 10.0]))  # y
    z_01 = [np.log10(tev1_01[:, 1]),
            np.log10(tev3_01[:, 1]),
            np.log10(tev10_01[:, 1])]  # z
    f_01 = (interpolate.interp2d(e_Timur, e0_Timur, z_01,
                                 kind='linear',
                                 copy=False,
                                 bounds_error=False,
                                 fill_value=None))

    def Timur_cascade_01(e_gamma, e0):
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html
        x = np.log10(e_gamma.to(u.TeV).value)
        y = np.log10(e0.to(u.TeV).value)
        return 10.0**f_01(x, y)

    def Timur_cascade_01_integrated(energy_cascade,
                                    gamma_absorbed_energy_spectrum_table,
                                    energy_gamma_unit=u.eV,
                                    spectrum_gamma_unit=1.0,
                                    energy_cascade_unit=u.eV):
        # energy_gamma_unit = energy_gamma_unit.to(u.TeV)
        energy_cascade = energy_cascade * energy_cascade_unit
        e0 = (gamma_absorbed_energy_spectrum_table[:, 0] *
              energy_gamma_unit *
              u.dimensionless_unscaled)
        spectrum = (gamma_absorbed_energy_spectrum_table[:, 1] * spectrum_gamma_unit *
                    u.dimensionless_unscaled)
        y = np.zeros((energy_cascade.shape[0]))
        for i in range(0, energy_cascade.shape[0]):
            y[i] = simps(
                spectrum.value * Timur_cascade_01(energy_cascade[i],
                                                  e0).reshape(
                    spectrum.shape
                ),
                e0.value
            )
        # Demand energy conservation
        absorbed_full_energy = simps(e0.value * spectrum.value,
                                     e0.value)
        cascade_full_energy = simps(energy_cascade.value * y,
                                    energy_cascade.value)
        y = y / cascade_full_energy * absorbed_full_energy
        y = y * spectrum_gamma_unit
        return y
    ############################################################################
    # Geometry, distance, magnetic field
    z = 0.0001  # 36  # http://linker.aanda.org/10.1051/0004-6361/201833618/54
    r_blr = (0.036 * u.pc).to(u.cm)  # BLR radius
    print("r_blr = {:.3e}".format(r_blr))
    d_l = cosmology.luminosity_distance(z).to(u.cm)
    print("d_l = {:.6e}".format(d_l))
    # b = 50.0 * u.mG  # 50 milligauss, you can use, e.g. u.kG, u.mkG and so on
    # print("B = {:.6e}".format(b))
    ############################################################################
    # norm_e = 7.959e+39 * u.eV**(-1)
    # print("norm_e = {:.3e}".format(norm_e))
    # gamma1 = 1.9
    # gamma2 = 4.5
    # print("gamma1 = {:.2f}, gamma2 = {:.2f}".format(gamma1, gamma2))
    # e_br = 9.0e+09 * u.eV
    # e_min_e = 5.0e+06 * u.eV
    # e_max_e = 1.0e+12 * u.eV
    # print("e_break = {:.3e}, e_min = {:.3e}, e_max = {:.3e}".format(
    #     e_br, e_min_e, e_max_e
    # ))
    # e_e = np.logspace(np.log10(e_min_e.to(u.eV).value),
    #                   np.log10(e_max_e.to(u.eV).value), 100) * u.eV
    #########################################################################
    field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/nph"
    # field = '/home/raylend/Science/agnprocesses/data/test_1eV.txt'
    primary_energy_size = 10_000
    e_min_value = 1.0e+08  # eV
    e_max_value = 1.0e+13  # eV
    en_ref = 1.0e+10  # eV
    energy_ic_threshold = 1.0e+07  # eV
    E_gamma, r_gamma_gamma = gamma_gamma.interaction_rate(field,
                                                          e_min_value * u.eV,
                                                          e_max_value * u.eV,
                                                          background_photon_energy_unit=u.eV,
                                                          background_photon_density_unit=(u.eV * u.cm**3)**(-1))
    E_IC_node, r_IC_node = ic.IC_interaction_rate(field,
                                                  e_min_value * u.eV,
                                                  e_max_value * u.eV,
                                                  energy_ic_threshold * u.eV,
                                                  background_photon_energy_unit=u.eV,
                                                  background_photon_density_unit=(u.eV * u.cm**3)**(-1))
    #########################################################################

    def my_spec(en, alpha, beta, norm):
        return(spec.log_parabola(en, alpha, beta, en_ref=en_ref, norm=norm))

    def model_SED(en, alpha, beta, norm, x,
                  E_gamma=E_gamma,
                  r_gamma_gamma=r_gamma_gamma):
        # norm = norm * u.eV**(-1)
        # x = x * u.cm
        SED = en**2 * my_spec(en, alpha, beta, norm)
        n_photons_in_source = simps((SED / en**2), en)
        en_earth = en / (1.0 + z)
        SED_earth = en_earth**2 * my_spec(en_earth, alpha, beta, norm)
        n_photons_at_earth = simps(
            (SED_earth / en_earth**2), en_earth)
        k_norm = n_photons_in_source / n_photons_at_earth
        r_gamma_gamma = spec.to_current_energy(en * u.eV,
                                               E_gamma,
                                               r_gamma_gamma).value
        tau_internal = np.sqrt((r_blr.value - x)**2) * r_gamma_gamma
        SED = SED * np.exp(-tau_internal)
        SED = SED * np.exp(-ebl.tau_gilmore(en_earth * u.eV, z))
        return SED
    #########################################################################
    # Data Fermi-LAT
    data_fermi = np.loadtxt(
        'data/PKS1510-089/4FGL J1512.8-0906_Fermi-LAT_energy_SED_downerror_uperror.txt'
    )
    data_en_fermi = data_fermi[:, 0] * 1.0e+06  # MeV -> eV
    data_sed_fermi = data_fermi[:, 1] * 1.0e+06  # MeV -> eV
    data_low_fermi = data_fermi[:, 2] * 1.0e+06  # MeV -> eV
    data_up_fermi = data_fermi[:, 3] * 1.0e+06  # MeV -> eV
    yerr_fermi = [data_low_fermi, data_up_fermi]
    # Data magic
    data_magic = np.loadtxt(
        'data/PKS1510-089/4FGL J1512.8-0906_MAGIC_energy_SED_downerror_uperror.txt')
    data_en_magic = data_magic[:, 0] * 1.0e+06  # MeV -> eV
    data_sed_magic = data_magic[:, 1] * 1.0e+06  # MeV -> eV
    data_low_magic = data_magic[:, 2] * 1.0e+06  # MeV -> eV
    data_up_magic = data_magic[:, 3] * 1.0e+06  # MeV -> eV
    yerr_magic = [data_low_magic, data_up_magic]

    #########################################################################
    def fit_data(f, x_data, y_data, y_sigma, p0):
        popt, pcov = curve_fit(f, x_data, y_data, p0=p0, sigma=y_sigma,
                               absolute_sigma=True, check_finite=True)  # ,
        # bounds=bounds)
        print("---------------------------------------------------------------")
        print("The following parameters have been obtained:")
        print(
            "alpha = {:e} +/- {:e}".format(popt[0], np.sqrt(np.diag(pcov)[0])))
        print(
            "beta = {:e} +/- {:e}".format(popt[1], np.sqrt(np.diag(pcov)[1])))
        print(
            "norm = {:e} +/- {:e}".format(popt[2], np.sqrt(np.diag(pcov)[2])))
        print("x = {:e} +/- {:e} = \n= {:f} +/- {:f} [R_BLR]".format(
            popt[3],
            np.sqrt(
                np.diag(
                    pcov)[3]),
            (popt[3] /
             r_blr.value), (np.sqrt(
                 np.diag(
                     pcov)[3]
             ) /
                r_blr.value)
        )
        )
        #####################################################################
        # calculate chi^2
        data_predicted_from_fit = f(
            x_data, *popt)
        chi_sq = np.sum((y_data - data_predicted_from_fit)**2 / y_sigma**2)
        chi_sq = chi_sq / (y_data.shape[0] - len(popt) - 1)
        print("chi_sq / n.d.o.f. = {:f}".format(chi_sq))
        return (popt, np.sqrt(np.diag(pcov)))
    #########################################################################
    p0 = [2.716197e+00, 7.052904e-02, 1.617821e-19, 1.110844e+17]
    x_data = np.concatenate([data_en_fermi], axis=0)
    y_data = np.concatenate([data_sed_fermi], axis=0)
    y_sigma = np.concatenate([yerr_fermi[0]], axis=0)
    # popt, perr = fit_data(model_SED, x_data, y_data, y_sigma, p0)
    #########################################################################
    en = np.logspace(np.log10(e_min_value * 1.1),
                     np.log10(e_max_value * 0.9),
                     1000)
    # SED = model_SED(en, *popt)
    SED = model_SED(en, *p0)
    #########################################################################
    r_gamma_gamma = spec.to_current_energy(en * u.eV, E_gamma, r_gamma_gamma)
    # Timur_rate = np.loadtxt(
    #     "data/PKS1510-089/gamma-gamma_interaction_rate/Rate-Gamma"
    # )
    # # Timur_filt = (Timur_rate[:, 1] > 0)
    # # Timur_rate[:, 0] = Timur_rate[:, 0][Timur_filt]
    # # Timur_rate[:, 1] = Timur_rate[:, 1][Timur_filt]
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # plt.plot(en, r_gamma_gamma,
    #          label='Egor',
    #          color='k',
    #          linewidth=3,
    #          linestyle='-')
    #
    # plt.plot(Timur_rate[:, 0], Timur_rate[:, 1],
    #          label='Timur',
    #          color='r',
    #          linewidth=3,
    #          linestyle='--')
    #
    # plt.xlabel('energy, ' + 'eV', fontsize=18)
    # plt.xticks(fontsize=12)
    # plt.ylabel('gamma-gamma interaction rate, ' + r'cm$^{-1}$', fontsize=18)
    # plt.yticks(fontsize=12)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # # ax.set_xlim(1.0e+08, 1.0e+12)
    # ax.set_ylim(1.0e-21, 1.0e-15)
    # # fig.savefig('test_figures/exponential_cutoff_compare_with_Derishev_fig4a.pdf')
    # plt.legend()
    # plt.show()
    #########################################################################
    tau_internal = 0.1 * r_blr * r_gamma_gamma
    SED2 = SED  # my_spec(en, 1.716197e+00, 0, 1.617821e-19) * en**2
    SED_not_survived = (1.0 - np.exp(-tau_internal)) * SED2
    filt = (SED_not_survived >= 1.0e-06 * max(SED_not_survived))
    SED_not_survived = SED_not_survived[filt]
    energy_not_survived = en[filt]
    spectrum_not_survived = SED_not_survived / energy_not_survived**2
    spectrum_cascade = Timur_cascade_01_integrated(en,
                                                   spec.create_2column_table(
                                                       energy_not_survived,
                                                       spectrum_not_survived
                                                   ))
    SED_cascade = en**2 * spectrum_cascade
    SED_cascade = SED_cascade * np.exp(-ebl.tau_gilmore(
        en / (1.0 + z) * u.eV, z))
    SED_survived = SED2 * np.exp(-tau_internal) * np.exp(-ebl.tau_gilmore(
        en / (1.0 + z) * u.eV, z))
    #########################################################################
    print(f"SED_not_survived.shape = {SED_not_survived.shape}")
    gamma_table = spec.create_2column_table(
        energy_not_survived, SED_not_survived)
    pair_e, pair_sed = gamma_gamma.pair_production(
        field,
        gamma_table,
        background_photon_energy_unit=u.eV,
        background_photon_density_unit=(u.eV * u.cm**3)**(-1),
        gamma_energy_unit=u.eV,
        gamma_sed_unit=u.eV / (u.cm**2 * u.s)
    )
    pair_energy_full = simps((pair_sed / pair_e).value, pair_e.value)
    print("Pair energy = {:e}".format(pair_energy_full))
    not_survived_energy_full = simps((SED_not_survived /
                                      energy_not_survived),
                                     energy_not_survived)
    print("not_survived_energy_full = {:e}".format(not_survived_energy_full))
    #########################################################################
    #########################################################################
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    ########################################################################
    # Monte-carlo part
    # #1 Generate a tensor of primary energy
    # e_min_value = 3.0e+09
    shape = (primary_energy_size, )
    original_params = [1.0, 1.0, en_ref]
    new_params = [p0[0], p0[1]]
    new_keyargs = {
        'en_ref': en_ref,
        'norm': p0[2]
    }
    # primary_energy = torch.ones((primary_energy_size,),
    #                             device=device) * 1.0e+13
    primary_energy = generate_random_numbers(
        spec.power_law,
        original_params,  # gamma, norm, en_ref
        shape=shape,
        xmin=e_min_value * torch.ones(shape, device=device),
        xmax=e_max_value * torch.ones(shape, device=device),
        device=device,
        logarithmic=True,
        n_integration=int(primary_energy_size),
        normalized=True
    )
    #########################################################################
    # primary_energy = primary_energy.detach().to('cpu').numpy()
    # print(f"Using parameter vector p0 = {p0}")
    # # weight_prime = (my_spec(primary_energy, p0[0], p0[1], p0[2]) /
    # #                 spec.power_law(primary_energy, *[1.0, 1.0, en_ref]))
    # # weight_prime = weight_prime / min(weight_prime)
    # prime_bins = np.logspace(np.log10(e_min_value),
    #                          np.log10(e_max_value),
    #                          31)
    # prime_hist, prime_bins = np.histogram(
    #     primary_energy,
    #     weights=None,
    #     bins=prime_bins,
    #     density=False
    # )
    # prime_hist_no_weight, prime_bins_no_weight = np.histogram(
    #     primary_energy,
    #     weights=None,
    #     bins=prime_bins,
    #     density=False
    # )
    # prime_en = 10.0**((np.log10(prime_bins[1:]) +
    #                   np.log10(prime_bins[:-1])) / 2.0)
    # k_modif = (my_spec(prime_en, p0[0], p0[1], p0[2]) /
    #            spec.power_law(prime_en, *[1.0, 1.0, en_ref]))
    # # k_modif = 1.0
    # prime_hist = prime_hist * k_modif
    # prime_hist = prime_hist / (prime_bins[1:] - prime_bins[:-1])  # dN / dE
    # prime_sed = prime_hist * prime_en**2
    # prime_sed = prime_sed / max(prime_sed) * max(SED)
    # prime_sed_err = prime_hist_no_weight**(-0.5) * prime_sed
    # print(f"prime_sed_err_relative = {prime_hist_no_weight**(-0.5)}")
    #########################################################################
    eps_max = 1500
    threshold = S_MIN / (4.0 * eps_max)
    primary_energy = primary_energy[primary_energy > threshold]
    print("Above threshold are: {:.2f} %".format(len(primary_energy) /
                                                 primary_energy_size *
                                                 100.0))
    #########################################################################
    # #2 sample distance traveled by primary gamma rays
    energy_gamma_node = torch.tensor(en,
                                     device=device)
    r_gamma_gamma_node = torch.tensor(r_gamma_gamma.value,
                                      device=device)
    energy_electron_node = torch.tensor(E_IC_node, device=device)
    r_IC_node = torch.tensor(r_IC_node.value, device=device)
    distance_traveled = generate_random_distance_traveled(
        primary_energy,
        energy_gamma_node,
        r_gamma_gamma_node
    )
    #########################################################################
    # #3 Get escaped and interacted gamma rays
    region_size = 0.1 * r_blr.value
    print("region size = {:.2} cm = {:.2} [r_blr]".format(
        region_size, region_size / r_blr.value
    ))
    print("Mean distance traveled: {:.2e}".format(
        torch.mean(distance_traveled)
    ))
    escaped = (distance_traveled > region_size)
    escaped_energy = primary_energy[escaped]
    interacted_energy = primary_energy[torch.logical_not(escaped)]
    print("Interacted {:.6f} % of photons".format(
        (1 - len(escaped_energy) / len(primary_energy)) * 100.0)
    )
    print("Escaped {:.6f} % of photons".format(
        (len(escaped_energy) / len(primary_energy)) * 100.0)
    )
    #########################################################################
    # 4 Generate s
    s = generate_random_s(interacted_energy,
                          field,
                          random_cosine=True,
                          device=device)
    # #5 Choose s >= (2 * ELECTRON_REST_ENERGY)**2
    filt_s = (s >= S_MIN)
    s = s[filt_s]
    print("Fraction of gamma rays generated pairs: {:.6f} %".format(
        len(s) / len(interacted_energy) * 100.0
    ))
    interacted_energy = interacted_energy[filt_s]
    ########################################################################
    # #6 Generate electron and positron energy
    y = generate_random_y(s,
                          logarithmic=True,
                          device=device)
    print("Max y = {:.2e}, min y = {:.2e}, mean y = {:.2e}".format(
        max(y), min(y), torch.mean(y)))
    electron_energy = y * interacted_energy
    positron_energy = (1.0 - y) * interacted_energy
    lepton_energy = torch.cat([electron_energy, positron_energy])
    ########################################################################
    # #7 Sample distance traveled by leptons
    distance_traveled_IC = generate_random_distance_traveled(
        lepton_energy,
        energy_electron_node,
        r_IC_node
    )
    ########################################################################
    # #8 Get escaped and interacted electrons
    region_size = 0.1 * r_blr.value
    print("region size = {:.2} cm = {:.2} [r_blr]".format(
        region_size, region_size / r_blr.value
    ))
    print("Mean distance traveled by leptons: {:.2e}".format(
        torch.mean(distance_traveled_IC)
    ))
    escaped_IC = (distance_traveled_IC > region_size)
    escaped_energy_IC = lepton_energy[escaped_IC]
    interacted_energy_IC = lepton_energy[torch.logical_not(escaped_IC)]
    print("Interacted {:.6f} % of leptons".format(
        (len(interacted_energy_IC) / len(lepton_energy)) * 100.0)
    )
    print("Escaped {:.6f} % of leptons".format(
        (len(escaped_energy_IC) / len(lepton_energy)) * 100.0)
    )
    s = None
    ########################################################################
    # #9 Generate s_IC
    s_IC = generate_random_s(interacted_energy_IC,
                             field,
                             random_cosine=True,
                             process='IC',
                             energy_ic_threshold=energy_ic_threshold,
                             device=device)
    eps_thr = energy_ic_threshold / interacted_energy_IC
    # #10 Choose s_IC >= ELECTRON_REST_ENERGY**2 / (1 - eps)
    filt_s_IC = (s_IC >= ELECTRON_REST_ENERGY**2 / (1.0 - eps_thr))
    s_IC = s_IC[filt_s_IC]
    print("Fraction of leptons undergone IC-scattering: {:.6f} %".format(
        len(s_IC) / len(interacted_energy_IC) * 100.0
    ))
    interacted_energy_IC = interacted_energy_IC[filt_s_IC]
    print('s_IC = ', s_IC)
    ########################################################################
    # #11 Get IC scattered electrons and upscattered gamma rays
    y_IC = generate_random_y(s_IC,
                             device=device,
                             logarithmic=True,
                             process='IC')
    print("Max y_IC = {:.2e}, min y_IC = {:.2e}, mean y_IC = {:.2e}".format(
        torch.max(y_IC), torch.min(y_IC), torch.mean(y_IC)))
    lepton_energy = y_IC * interacted_energy_IC
    gamma_energy = (1.0 - y_IC) * interacted_energy_IC
    print(
        "Max gamma_energy = {:.2e}, min gamma_energy = {:.2e}, mean gamma_energy = {:.2e}".format(
            torch.max(gamma_energy),
            torch.min(gamma_energy),
            torch.mean(gamma_energy)
        )
    )
    ########################################################################
    # #12 Plot lepton SED
    lepton_bins = np.logspace(np.log10(e_min_value),
                              np.log10(e_max_value),
                              50)

    electron_energy_center, electron_SED = make_SED(
        electron_energy,
        lepton_bins,
        interacted_energy,
        original_spectrum=spec.power_law,
        original_params=original_params,
        new_spectrum=spec.log_parabola,
        new_params=new_params,
        new_keyargs=new_keyargs
    )

    positron_energy_center, positron_SED = make_SED(
        positron_energy,
        lepton_bins,
        interacted_energy,
        original_spectrum=spec.power_law,
        original_params=original_params,
        new_spectrum=spec.log_parabola,
        new_params=new_params,
        new_keyargs=new_keyargs
    )

    lepton_energy, lepton_SED = spec.summ_spectra(electron_energy_center,
                                                  electron_SED,
                                                  positron_energy_center,
                                                  positron_SED)
    lepton_SED = lepton_SED / max(lepton_SED) * max(pair_sed)
    filt_lepton = (lepton_SED > 0)
    lepton_SED = lepton_SED[filt_lepton]
    lepton_energy = lepton_energy[filt_lepton]
    ########################################################################
    # #13 Plot IC-upscattered gamma-ray SED
    gamma_bins = np.logspace(np.log10(e_min_value),
                             np.log10(e_max_value),
                             50)
    gamma_IC_MC_energy, gamma_IC_MC_SED = make_SED(
        gamma_energy,
        gamma_bins,
        torch.cat([interacted_energy, interacted_energy])[torch.logical_not(
            escaped_IC
        )],
        original_spectrum=spec.power_law,
        original_params=original_params,
        new_spectrum=spec.log_parabola,
        new_params=new_params,
        new_keyargs=new_keyargs
    )
    gamma_IC_MC_SED = gamma_IC_MC_SED / min(gamma_IC_MC_SED) * min(
        lepton_SED)
    ########################################################################
    # #14 Analytical theoretical IC-upscattered gamma-ray SED
    gamma_IC_analytical_energy = np.logspace(8,
                                             13,
                                             100) * u.eV
    lepton_spectrum = lepton_SED / lepton_energy**2
    lepton_spectrum = lepton_spectrum / min(lepton_spectrum)
    electron_table = spec.create_2column_table(lepton_energy,
                                               lepton_spectrum)
    print('electron_table = ', electron_table)
    gamma_IC_analytical_SED = ic.inverse_compton_spec(
        gamma_IC_analytical_energy,
        field,
        spec_law='user',
        electron_table=electron_table,
        user_en_unit=u.eV,
        user_spec_unit=u.eV**(-1),
        particle_mass=const.m_e.cgs,
        particle_charge=const.e.gauss,
        background_photon_energy_unit=u.eV,
        background_photon_density_unit=(u.eV * u.cm**3)**(-1)
    ) * gamma_IC_analytical_energy**2
    print('gamma_IC_analytical_SED = ', gamma_IC_analytical_SED)
    gamma_IC_analytical_SED = (gamma_IC_analytical_SED /
                               max(gamma_IC_analytical_SED) *
                               max(gamma_IC_MC_SED))
    #########################################################################
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(elapsed_time_ms)
    #########################################################################
    #########################################################################
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(1, 1, 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(
        en, SED,
        marker=None,
        linestyle='-',
        linewidth=3,
        color='c',
        label='SED model 0 (outside BLR)'
    )

    plt.plot(
        lepton_energy, lepton_SED,
        marker=None,
        linestyle='-.',
        linewidth=3,
        color='orange',
        label='Monte-Carlo lepton SED'
    )

    plt.plot(
        gamma_IC_MC_energy, gamma_IC_MC_SED,
        marker=None,
        linestyle='--',
        linewidth=3,
        label='Monte-Carlo IC upscattered gamma rays'
    )

    plt.plot(
        gamma_IC_analytical_energy, gamma_IC_analytical_SED,
        marker=None,
        linestyle=':',
        linewidth=3,
        label='Analytical IC upscattered gamma rays'
    )

    plt.plot(
        energy_not_survived, SED_not_survived,
        marker=None,
        linestyle='-',
        linewidth=3,
        color='y',
        zorder=2,
        label='SED_not_survived'
    )

    # plt.plot(
    #     en, SED_cascade,
    #     marker=None,
    #     linestyle=':',
    #     linewidth=3,
    #     color='grey',
    #     label='SED model 0.1 cascade component'
    # )

    # plt.plot(
    #     en, SED_survived,
    #     marker=None,
    #     linestyle='--',
    #     linewidth=3,
    #     color='green',
    #     label='SED model 0.1 survived component (after EBL absorption)'
    # )

    plt.plot(
        pair_e, pair_sed,
        marker=None,
        linestyle=':',
        linewidth=3,
        color='red',
        label='SED of pair electrons'
    )

    # en_sum, SED_sum_01 = spec.summ_spectra(en, SED_cascade,
    #                                        en, SED_survived)
    #
    # plt.plot(
    #     en_sum, SED_sum_01,
    #     marker=None,
    #     linestyle='-.',
    #     linewidth=3,
    #     color='b',
    #     label='SED model 0.1 sum'
    # )

    # plt.plot(
    #     en, SED2,
    #     marker=None,
    #     linestyle='-.',
    #     linewidth=3,
    #     color='k',
    #     label='SED model 0.1 primary without EBL absorption'
    # )

    plt.errorbar(data_en_fermi, data_sed_fermi,
                 yerr=yerr_fermi, xerr=None, fmt='o', linewidth=0, elinewidth=2,
                 capsize=1, barsabove=False, markersize=5,
                 errorevery=1, capthick=1, color='r', zorder=100.0,
                 label='Fermi-LAT')

    plt.errorbar(data_en_magic, data_sed_magic,
                 yerr=yerr_magic, xerr=None, fmt='s', linewidth=0, elinewidth=2,
                 capsize=1, barsabove=False, markersize=5,
                 errorevery=1, capthick=1, color='orange', zorder=100.0,
                 label='MAGIC')
    plt.xlabel('energy, ' + 'eV', fontsize=18)
    plt.xticks(fontsize=12)
    plt.ylabel('SED, ' + r'eV cm$^{-2}$s$^{-1}$', fontsize=18)
    plt.yticks(fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.xaxis.set_major_locator(ticker.LogLocator(
    #     base=10.0, numticks=22, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]))
    # ax.set_xlim(1.0e+08, 1.0e+12)
    # ax.set_ylim(1.0e-05, 3.0e+02)
    # ax.grid()
    # ax.grid()
    # fig.savefig('test_figures/exponential_cutoff_compare_with_Derishev_fig4a.pdf')
    plt.legend()
    # plt.legend(loc='upper left')
    plt.show()
    #########################################################################
