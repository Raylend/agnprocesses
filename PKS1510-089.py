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
from scipy import stats
#############################################################################
from test_pytorch import *
from functools import partial


if __name__ == '__main__':
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    #########################################################################
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
    ########################################################################
    # Geometry, distance, magnetic field
    z = 0.361  # http://linker.aanda.org/10.1051/0004-6361/201833618/54
    r_blr = (0.036 * u.pc).to(u.cm)  # BLR radius
    print("r_blr = {:.3e}".format(r_blr))
    d_l = cosmology.luminosity_distance(z).to(u.cm)
    print("d_l = {:.6e}".format(d_l))
    # b = 50.0 * u.mG  # 50 milligauss, you can use, e.g. u.kG, u.mkG and so on
    # print("B = {:.6e}".format(b))
    ########################################################################
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
    ########################################################################
    field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/nph"
    # field = '/home/raylend/Science/agnprocesses/data/test_1eV.txt'
    # primary_energy_size = 100_000
    e_min_value = 1.0e+08  # eV
    e_max_value = 1.0e+12  # eV
    en_ref = 1.0e+10  # eV
    # energy_ic_threshold = 1.0e+07  # eV
    E_gamma, r_gamma_gamma = gamma_gamma.interaction_rate(
        field,
        e_min_value * u.eV,
        e_max_value * u.eV,
        background_photon_energy_unit=u.eV,
        background_photon_density_unit=(u.eV * u.cm**3)**(-1))
    # E_IC_node, r_IC_node = ic.IC_interaction_rate(
    #     field,
    #     e_min_value * u.eV,
    #     e_max_value * u.eV,
    #     energy_ic_threshold * u.eV,
    #     background_photon_energy_unit=u.eV,
    #     background_photon_density_unit=(u.eV * u.cm**3)**(-1))
    #########################################################################

    def my_spec(en, alpha, beta, norm):
        return(spec.log_parabola(en, alpha, beta, en_ref=en_ref, norm=norm))

    def model_SED(en_observable, alpha, beta, norm, x,
                  E_gamma=E_gamma,
                  r_gamma_gamma=r_gamma_gamma,
                  redshift=z):
        en_source = en_observable * (1.0 + redshift)
        SED = en_source**2 * my_spec(en_source, alpha, beta, norm)
        r_gamma_gamma = spec.to_current_energy(en_source * u.eV,
                                               E_gamma,
                                               r_gamma_gamma).value
        tau_internal = np.sqrt((r_blr.value - x)**2) * r_gamma_gamma
        SED = SED * np.exp(-tau_internal)
        SED = SED * np.exp(-ebl.tau_gilmore(en_observable * u.eV, z))
        SED = SED / (1.0 + redshift)
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
        data_predicted_from_fit = f(x_data, *popt)
        chi_sq = np.sum((y_data - data_predicted_from_fit)**2 / y_sigma**2)
        ndof = (y_data.shape[0] - len(popt) - 1)
        print("chi_sq / n.d.o.f. = {:f}".format(
            chi_sq /
            ndof)
        )
        return (popt, np.sqrt(np.diag(pcov)), chi_sq, ndof)
    #########################################################################
    folder_full_cascade = "test54_primary_gamma_field_30000_eps_thr=1e+05eV_with_losses_below_threshold_7e+07---1e+14eV_photon_max=None"
    energy_bins_fit_full_cascade = np.logspace(np.log10(7.0e+07), 14, 41)
    primary_energy_bins_fit_full_cascade = np.logspace(np.log10(7.0e+07),
                                                       14,
                                                       41)
    number_of_particles_full_cascade = 30_000
    hist_precalc_full_cascade = make_sed_from_monte_carlo_cash_files(
        folder_full_cascade,
        energy_bins_fit_full_cascade,
        primary_energy_bins=primary_energy_bins_fit_full_cascade,
        cascade_type_of_particles='all',
        original_spectrum=spec.power_law,
        original_params=[1.0, 1.0, 1.0],
        new_spectrum=None,
        new_params=None,
        original_keyargs={},
        new_keyargs={},
        reweight_array='primary',
        number_of_particles=number_of_particles_full_cascade,
        particles_of_interest='gamma',  # or 'electron'
        density=False,
        verbose=False,
        output='histogram_2d_precalculated',
        device='cpu'
    )
    gamma2 = 2.0

    def model_sed_from_monte_carlo(
        en_observable, gamma1, en_break, norm,
        gamma2=gamma2,
        energy_bins_mc=energy_bins_fit_full_cascade,
        primary_energy_bins_mc=primary_energy_bins_fit_full_cascade,
        redshift=z,
        original_spectrum=spec.power_law,
        original_params=[1.0, 1.0, 1.0],
        number_of_particles=number_of_particles_full_cascade,
        histogram_2d_precalculated=hist_precalc_full_cascade
    ):
        e, sed = make_SED(
            None,
            energy_bins_mc,
            None,
            primary_energy_bins=primary_energy_bins_mc,
            original_spectrum=original_spectrum,
            original_params=original_params,
            new_spectrum=spec.broken_power_law,
            new_params=[2.7, gamma2, en_break],
            original_keyargs={},
            new_keyargs={
                'norm': norm
            },
            reweight_array='primary',
            number_of_particles=number_of_particles,
            density=False,
            histogram_2d_precalculated=histogram_2d_precalculated)
        sed[sed <= 0] = 1.0e-40
        sed = spec.to_current_energy(en_observable * (1.0 + redshift),
                                     e, sed)
        sed = sed * np.exp(-ebl.tau_gilmore(en_observable * u.eV,
                                            redshift))
        sed = sed / (1.0 + redshift)  # AHNTUNG ATTENTION !!!
        return sed

    #########################################################################

    def fit_data_with_monte_carlo(f, x_data, y_data, y_sigma, p0):
        popt, pcov = curve_fit(f, x_data, y_data,
                               p0=p0, sigma=y_sigma,
                               absolute_sigma=True,
                               check_finite=False)  # , bounds=bounds)
        print("----------------------------------------------------------")
        print("The following parameters have been obtained:")
        for i, parameter in enumerate(popt):
            print(
                "parameter number {:d} = {:e} +/- {:e}".format(
                    i, popt[i], np.sqrt(np.diag(pcov)[i])))
        ####################################################################
        # calculate chi^2
        data_predicted_from_fit = f(x_data, *popt)
        chi_sq = np.sum((y_data - data_predicted_from_fit)**2 / y_sigma**2)
        ndof = (y_data.shape[0] - len(popt) - 1)
        print('ndof = {:d}'.format(ndof))
        print("chi_sq / n.d.o.f. = {:f}".format(
            chi_sq /
            ndof)
        )
        return (popt, np.sqrt(np.diag(pcov)), chi_sq, ndof)
    ########################################################################
    x_data = np.concatenate([data_en_fermi], axis=0)
    y_data = np.concatenate([data_sed_fermi], axis=0)
    y_sigma = np.concatenate([yerr_fermi[0]], axis=0)
    # y_sigma = np.ones(y_sigma.shape) * np.max(y_sigma) * 30
    en_observable = np.logspace(np.log10(e_min_value),
                                np.log10(e_max_value),
                                100)
    ########################################################################
    p0 = [2.7, 1.0e+12, 5.97805273e-23]
    popt, perr, chisq, ndof = fit_data_with_monte_carlo(
        model_sed_from_monte_carlo,
        x_data, y_data, y_sigma,
        p0)
    print("popt = ", popt)
    sed_mc_full_cascade = model_sed_from_monte_carlo(en_observable, *popt)
    e_mc_raw_full_cascade, sed_mc_raw_full_cascade = make_SED(
        None,
        energy_bins_fit_full_cascade,
        None,
        primary_energy_bins=primary_energy_bins_fit_full_cascade,
        original_spectrum=spec.power_law,
        original_params=[1.0, 1.0, 1.0],
        new_spectrum=spec.broken_power_law,
        new_params=[2.7, gamma2, popt[1]],
        original_keyargs={},
        new_keyargs={
            'norm': popt[2]
        },
        reweight_array='primary',
        number_of_particles=number_of_particles_full_cascade,
        density=False,
        histogram_2d_precalculated=hist_precalc_full_cascade
    )
    ########################################################################
    p0_null_model = [2.7, 0.07, 1.62e-19, 1.110844e+17]
    popt_null_model, perr_null_model, chisq_null_model, ndof_null_model = fit_data(
        model_SED, x_data,
        y_data, y_sigma,
        p0_null_model
    )
    sed_null_model = model_SED(en_observable, *popt_null_model)
    ########################################################################
    # en_mc_survived, sed_mc_survived = make_sed_from_monte_carlo_cash_files(
    #     folder,
    #     energy_bins_fit,  # array-like
    #     primary_energy_bins=primary_energy_bins_fit,
    #     cascade_type_of_particles='survived',
    #     original_spectrum=spec.power_law,
    #     original_params=[1.0, 1.0, 1.0],
    #     new_spectrum=spec.log_parabola,
    #     new_params=[popt[0], popt[1]],
    #     original_keyargs={},
    #     new_keyargs={
    #         'norm': popt[2],
    #         'en_ref': en_ref
    #     },
    #     reweight_array='primary',
    #     number_of_particles=number_of_particles,
    #     particles_of_interest='gamma',  # or 'electron'
    #     density=False,
    #     verbose=False,
    #     output='sed',
    #     device='cpu'
    # )
    ########################################################################
    folder_hybrid = "test47_primary_electron_field_30000_eps_thr=1e+05eV_with_losses_below_threshold_7e+07---1e+14eV_photon_max=None"
    energy_bins_fit_hybrid = np.logspace(8, 14, 36)
    primary_energy_bins_fit_hybrid = np.logspace(np.log10(5e+12),
                                                 np.log10(3e+13),
                                                 11)
    en_primary_min_in_mc = 7.0e+07
    en_primary_max_in_mc = 1.0e+14
    number_of_particles_hybrid = (30_000 *
                                  np.log(primary_energy_bins_fit_hybrid[-1] /
                                         primary_energy_bins_fit_hybrid[0]) /
                                  np.log(en_primary_max_in_mc /
                                         en_primary_min_in_mc))
    hist_precalc_hybrid = make_sed_from_monte_carlo_cash_files(
        folder_hybrid,
        energy_bins_fit_hybrid,  # array-like
        primary_energy_bins=primary_energy_bins_fit_hybrid,
        cascade_type_of_particles='all',
        original_spectrum=spec.power_law,
        original_params=[1.0, 1.0, 1.0],
        new_spectrum=None,
        new_params=None,
        original_keyargs={},
        new_keyargs={},
        reweight_array='primary',
        number_of_particles=number_of_particles_hybrid,
        particles_of_interest='gamma',  # or 'electron'
        density=False,
        verbose=False,
        output='histogram_2d_precalculated',
        device='cpu'
    )

    def model_sed_from_monte_carlo_hybrid(
        en_observable,
        norm_primary_for_cascade,
        gamma_primary_for_cascade=2.0,
        alpha_null=2.725828e+00,
        beta_null=7.977361e-02,
        norm_null=2.544626e-19,
        en_ref=en_ref,
        energy_bins_mc=energy_bins_fit_hybrid,
        primary_energy_bins_mc=primary_energy_bins_fit_hybrid,
        redshift=z,
        original_spectrum=spec.power_law,
        original_params=[1.0, 1.0, 1.0],
        number_of_particles=number_of_particles_hybrid,
        histogram_2d_precalculated=hist_precalc_hybrid
    ):
        e_cascade, sed_cascade = make_SED(
            None,
            energy_bins_mc,
            None,
            primary_energy_bins=primary_energy_bins_mc,
            original_spectrum=original_spectrum,
            original_params=original_params,
            new_spectrum=spec.power_law,
            new_params=[gamma_primary_for_cascade, ],
            original_keyargs={},
            new_keyargs={
                'norm': norm_primary_for_cascade,  # to be fitted as well !!!
                'en_ref': en_ref
            },
            reweight_array='primary',
            number_of_particles=number_of_particles,
            density=False,
            histogram_2d_precalculated=histogram_2d_precalculated,
            old=False
        )
        sed_cascade[sed_cascade <= 0] = 1.0e-40
        en_source = en_observable * (1.0 + redshift)
        sed_cascade = spec.to_current_energy(
            en_source,
            e_cascade, sed_cascade
        )
        ###################################################################
        sed_null = en_source**2 * my_spec(en_source,
                                          alpha_null, beta_null, norm_null)
        ####################################################################
        sed = sed_cascade + sed_null
        sed = sed * np.exp(-ebl.tau_gilmore(en_observable * u.eV,
                                            redshift))
        sed = sed / (1.0 + redshift)  # AHNTUNG ATTENTION !!!
        return sed
    ########################################################################
    p0_hybrid = [2.5e-19, ]
    popt_hybrid, perr_hybrid, chisq_hybrid, ndof_hybrid = fit_data_with_monte_carlo(
        model_sed_from_monte_carlo_hybrid,
        x_data, y_data, y_sigma,
        p0_hybrid)
    print("popt_hybrid = ", popt_hybrid)
    sed_hybrid = model_sed_from_monte_carlo_hybrid(en_observable,
                                                   *popt_hybrid)
    ########################################################################
    folder_hybrid_gamma = "test54_primary_gamma_field_30000_eps_thr=1e+05eV_with_losses_below_threshold_7e+07---1e+14eV_photon_max=None"
    hist_precalc_hybrid_gamma = make_sed_from_monte_carlo_cash_files(
        folder_hybrid_gamma,
        energy_bins_fit_hybrid,  # array-like
        primary_energy_bins=primary_energy_bins_fit_hybrid,
        cascade_type_of_particles='all',
        original_spectrum=spec.power_law,
        original_params=[1.0, 1.0, 1.0],
        new_spectrum=None,
        new_params=None,
        original_keyargs={},
        new_keyargs={},
        reweight_array='primary',
        number_of_particles=number_of_particles_hybrid,
        particles_of_interest='gamma',  # or 'electron'
        density=False,
        verbose=False,
        output='histogram_2d_precalculated',
        device='cpu'
    )
    sed_hybrid_gamma = model_sed_from_monte_carlo_hybrid(
        en_observable,
        *popt_hybrid,
        number_of_particles=number_of_particles_hybrid,
        histogram_2d_precalculated=hist_precalc_hybrid_gamma
    )
    ########################################################################
    en_primary_hybrid = np.logspace(
        np.log10(primary_energy_bins_fit_hybrid[0]),
        np.log10(primary_energy_bins_fit_hybrid[-1]),
        333)
    sed_primary_hybrid = (en_primary_hybrid**2 *
                          spec.power_law(
                              en_primary_hybrid, 2.0,
                              *popt_hybrid, en_ref=en_ref)
                          )
    ########################################################################
    e_all_mc_electron, sed_all_mc_electron = make_sed_from_monte_carlo_cash_files(
        folder_hybrid,
        energy_bins_fit_hybrid,  # array-like
        primary_energy_bins=primary_energy_bins_fit_hybrid,
        cascade_type_of_particles='all',
        original_spectrum=spec.power_law,
        original_params=[1.0, 1.0, 1.0],
        new_spectrum=spec.power_law,
        new_params=[2.0],
        original_keyargs={},
        new_keyargs={'norm': popt_hybrid[0], 'en_ref': en_ref},
        reweight_array='primary',
        number_of_particles=number_of_particles_hybrid,
        particles_of_interest='gamma',  # or 'electron'
        density=False,
        verbose=False,
        output='sed',
        device='cpu'
    )
    e_all_mc_electron_observable = e_all_mc_electron / (1.0 + z)
    sed_all_mc_electron_observable = (sed_all_mc_electron / (1.0 + z) *
                                      np.exp(
                                          -ebl.tau_gilmore(
                                              e_all_mc_electron_observable * u.eV,
                                              z)
    ))
    ########################################################################
    e_all_mc_gamma, sed_all_mc_gamma = make_sed_from_monte_carlo_cash_files(
        folder_hybrid_gamma,
        energy_bins_fit_hybrid,  # array-like
        primary_energy_bins=primary_energy_bins_fit_hybrid,
        cascade_type_of_particles='all',
        original_spectrum=spec.power_law,
        original_params=[1.0, 1.0, 1.0],
        new_spectrum=spec.power_law,
        new_params=[2.0],
        original_keyargs={},
        new_keyargs={'norm': popt_hybrid[0], 'en_ref': en_ref},
        reweight_array='primary',
        number_of_particles=number_of_particles_hybrid,
        particles_of_interest='gamma',  # or 'electron'
        density=False,
        verbose=False,
        output='sed',
        device='cpu'
    )
    e_all_mc_gamma_observable = e_all_mc_gamma / (1.0 + z)
    sed_all_mc_gamma_observable = (sed_all_mc_gamma / (1.0 + z) *
                                   np.exp(-ebl.tau_gilmore(
                                       e_all_mc_gamma_observable * u.eV,
                                       z)
    )
    )
    ########################################################################
    en_source_null_model = en_observable * (1.0 + z)
    sed_source_null_model = (my_spec(en_source_null_model,
                                     *popt_null_model[:-1]) *
                             en_source_null_model**2)
    ########################################################################
    p_value = stats.chi2.sf(chisq, df=ndof)
    significance = spec.get_significance_2_tailed(p_value)
    p_value_null_model = stats.chi2.sf(chisq_null_model, df=ndof_null_model)
    significance_null_model = spec.get_significance_2_tailed(
        p_value_null_model
    )
    p_value_hybrid = stats.chi2.sf(chisq_hybrid, df=ndof_hybrid)
    significance_hybrid = spec.get_significance_2_tailed(
        p_value_hybrid
    )
    ########################################################################
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(1, 1, 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    filt_null_model = (en_observable < 5e+11)
    plt.plot(
        en_observable[filt_null_model], sed_null_model[filt_null_model],
        marker=None,
        linestyle='-',
        linewidth=3,
        color='c',
        label='Model 0 (log-parabolic at the edge of BLR),\nEarth frame; significance = {:.2f} sigma'.format(
            significance_null_model
        )
    )

    plt.plot(
        en_source_null_model, sed_source_null_model,
        marker=None,
        linestyle=':',
        linewidth=3,
        color='c',
        zorder=2,
        label='Model 0 primary source frame (log-parabolic at the edge of BLR)'
    )

    plt.plot(
        en_observable, sed_mc_full_cascade,
        marker=None,
        linestyle='--',
        linewidth=3,
        zorder=7,
        label='Monte-Carlo test54; full-cascade model\nprimary gamma rays; Earth frame; significance = {:.2f} sigma'.format(
            significance
        ),
        color='m'
    )

    plt.plot(
        en_observable, sed_hybrid,
        marker=None,
        linestyle='--',
        linewidth=3,
        zorder=4,
        label='Monte-Carlo test47; hybrid model\nprimary electron; Earth frame; significance = {:.2f} sigma'.format(
            significance_hybrid
        ),
        color='b'
    )

    plt.plot(
        en_observable, sed_hybrid_gamma,
        marker=None,
        linestyle='-',
        linewidth=3,
        zorder=3,
        label='Monte-Carlo test54; hybrid model; primary gamma; Earth frame',
        color='k'
    )

    plt.plot(
        en_primary_hybrid, sed_primary_hybrid,
        marker=None,
        linestyle=':',
        linewidth=3,
        zorder=3,
        label='Monte-Carlo test47/54; hybrid model; primary electrons or gamma rays (source frame)',
        color='orange'
    )

    plt.plot(
        e_all_mc_electron, sed_all_mc_electron,
        marker=None,
        linestyle='-.',
        linewidth=3,
        zorder=3,
        color='r',
        label='Monte-Carlo test47; hybrid model; gamma rays from primary electrons (source frame)'
    )

    plt.plot(
        e_all_mc_electron_observable, sed_all_mc_electron_observable,
        marker=None,
        linestyle='-',
        linewidth=3,
        zorder=3,
        color='r',
        label='Monte-Carlo test47; hybrid model; gamma rays from primary electrons (Earth frame)'
    )

    plt.plot(
        e_all_mc_gamma, sed_all_mc_gamma,
        marker=None,
        linestyle='-.',
        linewidth=3,
        zorder=3,
        color='g',
        label='Monte-Carlo test54; hybrid model; gamma rays from primary gamma rays (source frame)'
    )

    plt.plot(
        e_all_mc_gamma_observable, sed_all_mc_gamma_observable,
        marker=None,
        linestyle='-',
        linewidth=3,
        zorder=3,
        color='g',
        label='Monte-Carlo test54; hybrid model; gamma rays from primary gamma rays (Earth frame)'
    )

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
    plt.xticks(fontsize=13)
    plt.ylabel('SED, ' + r'eV cm$^{-2}$s$^{-1}$', fontsize=18)
    plt.yticks(fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.xaxis.set_major_locator(ticker.LogLocator(
    #     base=10.0, numticks=22, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]))
    # ax.set_xlim(1.0e+08, 1.0e+12)
    ax.set_ylim(1.0e-01, 1.0e+03)
    # ax.grid()
    # ax.grid()
    # fig.savefig('test_figures/exponential_cutoff_compare_with_Derishev_fig4a.pdf')
    ######################################################################
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(elapsed_time_ms)
    ######################################################################
    plt.legend(fontsize=13, loc='upper right')
    # plt.legend(loc='upper left')
    plt.show()
    #####################################################################
    # en_cutoff = 5e+10
    # number_of_particles_1eV = 30_000
    # folder_1eV = "test46_1eVblackbody_30000_eps_thr=1e+05eV_with_losses_below_threshold_1e+06---1e+14eV_with_electron_ic_losses_photon_max=None"
    #
    # def model_sed_from_monte_carlo_expcutoff(
    #     en_observable, gamma, norm,
    #     en_ref=en_ref,
    #     en_cutoff=en_cutoff,
    #     energy_bins_mc=energy_bins_fit,
    #     primary_energy_bins_mc=primary_energy_bins_fit,
    #     redshift=z,
    #     original_spectrum=spec.power_law,
    #     original_params=[1.0, 1.0, 1.0],
    #     number_of_particles=number_of_particles,
    #     histogram_2d_precalculated=hist_precalc
    # ):
    #     e, sed = make_SED(
    #         None,
    #         energy_bins_mc,
    #         None,
    #         primary_energy_bins=primary_energy_bins_mc,
    #         original_spectrum=original_spectrum,
    #         original_params=original_params,
    #         new_spectrum=spec.exponential_cutoff,
    #         new_params=[gamma, en_cutoff],
    #         original_keyargs={},
    #         new_keyargs={
    #             'norm': norm,
    #             'en_ref': en_ref
    #         },
    #         reweight_array='primary',
    #         number_of_particles=number_of_particles,
    #         density=False,
    #         histogram_2d_precalculated=histogram_2d_precalculated)
    #     # e = e[sed > 0]
    #     # sed = sed[sed > 0]
    #     sed = spec.to_current_energy(en_observable * (1.0 + redshift),
    #                                  e, sed)
    #     sed = sed * np.exp(-ebl.tau_gilmore(en_observable * u.eV,
    #                                         redshift))
    #     sed = sed / (1.0 + redshift)  # AHNTUNG ATTENTION !!!
    #     return sed
    #
    # ########################################################################
    # hist_precalc_1eV = make_sed_from_monte_carlo_cash_files(
    #     folder_1eV,
    #     energy_bins_fit,  # array-like
    #     primary_energy_bins=primary_energy_bins_fit,
    #     cascade_type_of_particles='all',
    #     original_spectrum=spec.power_law,
    #     original_params=[1.0, 1.0, 1.0],
    #     new_spectrum=None,
    #     new_params=None,
    #     original_keyargs={},
    #     new_keyargs={},
    #     reweight_array='primary',
    #     number_of_particles=number_of_particles,
    #     particles_of_interest='gamma',  # or 'electron'
    #     density=False,
    #     verbose=False,
    #     output='histogram_2d_precalculated',
    #     device='cpu'
    # )
    # p0_1eV = [2.0, 1.0e-20]
    # f_sed_1eV = partial(
    #     model_sed_from_monte_carlo_expcutoff,
    #     histogram_2d_precalculated=hist_precalc_1eV
    # )
    # popt_1eV, perr, chisq_1eV, ndof_1eV = fit_data_with_monte_carlo(
    #     f_sed_1eV,
    #     x_data, y_data, y_sigma,
    #     p0_1eV)
    #
    # sed_1eV = f_sed_1eV(en_observable, *popt_1eV)
