"""
This program computes inverse Compton (IC) emission of reletivistic electron_spectrum following Jones (1968) Phys. Rev. 167, 5
"""
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
try:
    import processes.spectra as spec
except:
    try:
        import spectra as spec
    except:
        raise ImportError("a problem with importing spectra.py occured")


def test():
    print("ic.py imported successfully.")
    return None


def inverse_compton_base(alpha, g, alpha1):
    """
    This is the original Jones formula for Compton (or inverse Compton)
    scattering

    alpha is the photon energy after the IC act (float)

    energy is the electron energy (float)

    alpha1 is the photon energy of background photons (float)
    """
    alpha_max = 4.0 * alpha1 * g * g / (1.0 + 4.0 * g * alpha1)
    ###########################################################################
    if (alpha > alpha1) and (alpha < alpha_max):
        t1 = 2.0 / (alpha1 * g * g)
        q = alpha / (4.0 * alpha1 * g * g * (1.0 - alpha / g))
        if ((1.0 / (4.0 * g * g) >= q) or (q > 1.0)):
            raise ValueError("Approximation is not valid! Abort session")
        t2 = 2.0 * q * np.log(q) + (1.0 + 2.0 * q) * (1.0 - q)
        t3 = 0.5 * (4.0 * alpha1 * g * q)**2 / \
            (1.0 + 4.0 * alpha1 * g * q) * (1.0 - q)
        res = (t1 * (t2 + t3))
    ###########################################################################
    if (alpha <= alpha1):
        res = 1.0 / (2.0 * g * g * g * g * alpha1) * \
            (4 * g * g * alpha / alpha1 - 1.0)
    ###########################################################################
    if (alpha >= alpha_max):
        # print("alpha = {:e} >= alpha_max = {:e}".format(alpha, alpha_max))
        res = 0.0
    ###########################################################################
    return res


def inverse_compton_over_photon_field(alpha,
                                      energy,
                                      field,
                                      particle_mass=const.m_e.cgs,
                                      particle_charge=const.e.gauss,
                                      background_photon_energy_unit=u.eV,
                                      background_photon_density_unit=(u.eV * u.cm**3)**(-1)):
    """
    alpha is the photon energy after the IC act (float or array-like)

    energy is the electron energy (float)

    field is the string with the path to the photon field .txt file OR numpy
    array with 2 colums: the first column is the background photon energy, the
    second columnn is the background photon density. Units in the table must
    correspond to the background_photon_energy_unit parameter and the
    background_photon_density_unit parameter.
    """
    list_of_energies = ['J', 'erg', 'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV']
    ###########################################################################
    if type(field) == type(''):
        try:
            field = np.loadtxt(field)
        except:
            raise ValueError(
                "Cannot read 'field'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / density). \n Try to use an absolute path.")
        eps = field[:, 0] * background_photon_energy_unit
        dens = field[:, 1]
    elif type(field) == type(np.array(([2, 1], [5, 6]))):
        eps = field[:, 0] * background_photon_energy_unit
        dens = field[:, 1]
    else:
        raise ValueError(
            "Invalid value of 'field'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / density).")
    ###########################################################################
    particle_mass = particle_mass.to(u.g, u.mass_energy())
    r0 = particle_charge**2 / (particle_mass * (const.c.cgs)**2)
    rest_energy = particle_mass.to(u.eV, u.mass_energy())
    try:
        a_un = alpha.unit
    except:
        a_un = ''
    k = np.pi * r0 * r0 * const.c.cgs
    ###########################################################################
    if type(energy) == type(3.0) or type(energy) == type(1) \
            or type(energy) == type(np.arange(0, 3)):
        g = energy
    elif energy.unit == 'dimensionless' or energy.unit == '':
        g = energy
    elif energy.unit in list_of_energies:
        g = (energy / rest_energy).decompose().value
    else:
        raise ValueError("invalid type of the argument 'energy'")
    ############################################################################
    if type(eps) == type(3.0) or type(eps) == type(1) \
            or type(eps) == type(np.arange(0, 3)):
        pass
    elif eps.unit == 'dimensionless' or eps.unit == '':
        pass
    elif eps.unit in list_of_energies:
        eps = (eps / rest_energy).decompose().value
    else:
        raise ValueError(
            "invalid type of the argument 'eps' (first column of 'field')")
    ############################################################################
    if type(alpha) == type(3.0) or type(alpha) == type(1) \
            or type(alpha) == type(np.arange(0, 3)):
        pass
    elif alpha.unit == 'dimensionless' or alpha.unit == '':
        pass
    elif alpha.unit in list_of_energies:
        alpha = (alpha / rest_energy).decompose().value
    else:
        raise ValueError("invalid type of the argument 'alpha'")
    ############################################################################
    if len(eps) < 1:
        raise ValueError(
            "Cannot read 'field'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy & density).")
    elif len(eps) >= 3:  # if we have external field as spectrum
        def map_alpha(alpha_single):
            def func(epsilon):
                return inverse_compton_base(alpha_single, g, epsilon)
            y = np.array(list(map(func, eps)))
            # here eps is in units of particle mass!
            # so we divide the result only by the particle_mass.unit!
            res = simps(y * dens, eps)
            return res
        final = np.array(list(map(map_alpha, alpha)))
        #final = np.array(final)
        final = final * k * background_photon_density_unit * background_photon_energy_unit
        final = final.decompose()
        if a_un != '' and a_un != 'dimensionless':
            final = final / particle_mass.to(a_un, u.mass_energy()).unit
    else:  # if we have external field as monoenergetic
        def map_alpha(alpha_single):
            res = inverse_compton_base(
                alpha_single, g, np.mean(eps)) * np.mean(dens)
            return res
        final = np.array(list(map(map_alpha, alpha)))
        final = final * k * background_photon_density_unit * background_photon_energy_unit
        final = final.decompose()
        if a_un != '' and a_un != 'dimensionless':
            final = final / particle_mass.to(a_un, u.mass_energy()).unit
    return final


def inverse_compton_spec(
    alpha,
    field,
    norm=1.0,
    spec_law='power_law',
    gamma1=None,
    gamma2=None,
    en_break=None,
    en_cutoff=None,
    en_min=None,
    en_max=None,
    en_mono=None,
    electron_table=None,
    user_en_unit=u.eV,
    user_spec_unit=u.dimensionless_unscaled,
    en_ref=1.0 * u.eV,
    number_of_integration=100,
    particle_mass=const.m_e.cgs,
    particle_charge=const.e.gauss,
    background_photon_energy_unit=u.eV,
    background_photon_density_unit=(u.eV * u.cm**3)**(-1)
):
    """
    alpha is the independent variable, photon energy
    (after the IC act) (float or array-like)

    norm is the normalization coefficient of charged particles

    spec_law must be 'power_law', 'broken_power_law',
    'exponential_cutoff' or 'monoenergetic'

    gamma1 is the spectral index for the 'power_law' or the first spectral index
    for the 'broken_power_law' or the spectral index for the 'exponential_cutoff'

    gamma2 is the second spectral indeex (after break) for the 'broken_power_law'
    en_break is the break energy for the 'broken_power_law'

    en_cutoff is the cutoff energy for the 'exponential_cutoff'

    en_min and en_max are minimum and maximum energies for 'power_law',
    'broken_power_law' or 'exponential_cutoff'

    en_mono is the energy of charged particle for the 'monoenergetic'

    en_ref is reference energy for 'power_law' or 'exponential_cutoff'

    ALL energies must be with the same units of astropy quantities or must be
    nd.arrays (in the latter case they will be considered as Lorentz factors)

    field is the string with the path to the photon field .txt file OR numpy
    array with 2 colums: the first column is the background photon energy, the
    second columnn is the background photon density. Units in the table must
    correspond to the background_photon_energy_unit parameter and the
    background_photon_density_unit parameter.
    """
    ############################################################################
    valid_spec_laws = ['power_law', 'broken_power_law',
                       'exponential_cutoff', 'monoenergetic',
                       'user']
    if spec_law not in valid_spec_laws:
        raise ValueError("Invalid spec_law. It must be one of {}"
                         .format(valid_spec_laws))
    f = None
    ############################################################################
    if spec_law == 'monoenergetic':
        par_list = [norm, en_mono]
        if all(el is not None for el in par_list):
            f = inverse_compton_over_photon_field(alpha,
                                                  en_mono,
                                                  field,
                                                  particle_mass=particle_mass,
                                                  particle_charge=particle_charge,
                                                  background_photon_energy_unit=background_photon_energy_unit,
                                                  background_photon_density_unit=background_photon_density_unit) * norm
        else:
            raise ValueError(
                "en_mono should be an astropy Quantity, e.g. 5*u.GeV"
            )
    ############################################################################
    elif spec_law == 'broken_power_law':
        par_list = [gamma1, gamma2, en_break, en_min, en_max, norm]
        if all(el is not None for el in par_list):
            ee = en_min.unit * np.logspace(
                np.log10(en_min.value),
                np.log10(en_max.to(en_min.unit).value),
                number_of_integration
            )

            def underintegral(energy):
                return(
                    (inverse_compton_over_photon_field(alpha,
                                                       energy,
                                                       field,
                                                       particle_mass=particle_mass,
                                                       particle_charge=particle_charge,
                                                       background_photon_energy_unit=background_photon_energy_unit,
                                                       background_photon_density_unit=background_photon_density_unit)).value *
                    (spec.broken_power_law(
                        energy, gamma1, gamma2, en_break, norm=norm
                    )).value
                )
            y = np.array(list(map(underintegral, ee)))
            f = np.array(list(
                map(
                    lambda i: simps(y[:, i].reshape(ee.shape), ee.value),
                    range(0, alpha.shape[0])
                )
            )
            ) * (inverse_compton_over_photon_field(alpha,
                                                   ee[0],
                                                   field,
                                                   particle_mass=particle_mass,
                                                   particle_charge=particle_charge,
                                                   background_photon_energy_unit=background_photon_energy_unit,
                                                   background_photon_density_unit=background_photon_density_unit)).unit * \
                (spec.broken_power_law(
                    ee[0], gamma1, gamma2, en_break, norm=norm)
                 ).unit * ee[0].unit
            # f = f.decompose(['eV', 's', 'cm'])
        else:
            raise ValueError(
                "Make sure you defined all of gamma1, gamma2,\
             en_break, en_min, en_max, norm correctly"
            )
    ###########################################################################
    elif spec_law == 'power_law':
        par_list = [gamma1, en_ref, en_min, en_max, norm]
        if all(el is not None for el in par_list):
            ee = en_min.unit * np.logspace(
                np.log10(en_min.value),
                np.log10(en_max.to(en_min.unit).value),
                number_of_integration
            )

            def underintegral(energy):
                return(
                    (inverse_compton_over_photon_field(alpha,
                                                       energy,
                                                       field,
                                                       particle_mass=particle_mass,
                                                       particle_charge=particle_charge,
                                                       background_photon_energy_unit=background_photon_energy_unit,
                                                       background_photon_density_unit=background_photon_density_unit)).value *
                    (spec.power_law(
                        energy, gamma1, en_ref=en_ref, norm=norm
                    )).value
                )
            y = np.array(list(map(underintegral, ee)))
            f = np.array(list(
                map(
                    lambda i: simps(y[:, i].reshape(ee.shape), ee.value),
                    range(0, alpha.shape[0])
                )
            )
            ) * (inverse_compton_over_photon_field(alpha,
                                                   ee[0],
                                                   field,
                                                   particle_mass=particle_mass,
                                                   particle_charge=particle_charge,
                                                   background_photon_energy_unit=background_photon_energy_unit,
                                                   background_photon_density_unit=background_photon_density_unit)).unit * \
                (spec.power_law(
                    ee[0], gamma1, en_ref=en_ref, norm=norm)
                 ).unit * ee[0].unit
            # f = f.decompose(['eV', 's', 'cm'])
        else:
            raise ValueError(
                "Make sure you defined all of gamma, en_ref, en_min, en_max, norm correctly"
            )
    ###########################################################################
    elif spec_law == 'exponential_cutoff':
        par_list = [gamma1, en_cutoff, en_min, en_max, norm, en_ref]
        if all(el is not None for el in par_list):
            ee = en_min.unit * np.logspace(
                np.log10(en_min.value),
                np.log10(en_max.to(en_min.unit).value),
                number_of_integration
            )

            def underintegral(energy):
                return(
                    inverse_compton_over_photon_field(alpha,
                                                      energy,
                                                      field,
                                                      particle_mass=particle_mass,
                                                      particle_charge=particle_charge,
                                                      background_photon_energy_unit=background_photon_energy_unit,
                                                      background_photon_density_unit=background_photon_density_unit).value *
                    spec.exponential_cutoff(
                        energy, gamma1, en_cutoff, norm=norm, en_ref=en_ref
                    ).value
                )
            y = np.array(list(map(underintegral, ee)))
            f = np.array(list(
                map(
                    lambda i: simps(y[:, i], ee.value),
                    range(0, alpha.shape[0])
                )
            )
            ) * inverse_compton_over_photon_field(alpha,
                                                  ee[0],
                                                  field,
                                                  particle_mass=particle_mass,
                                                  particle_charge=particle_charge,
                                                  background_photon_energy_unit=background_photon_energy_unit,
                                                  background_photon_density_unit=background_photon_density_unit).unit * spec.exponential_cutoff(
                ee[0], gamma1, en_cutoff, norm=norm, en_ref=en_ref
            ).unit * ee[0].unit
            # f = f.decompose(['eV', 's', 'cm'])
        else:
            raise ValueError(
                "Make sure you defined all of gamma1, en_cutoff, en_min, \
                en_max, norm, en_ref correctly"
            )
    ###########################################################################
    elif spec_law == 'user':
        par_list = [electron_table, user_en_unit, user_spec_unit]
        if all(el is not None for el in par_list):
            # checking electron_table
            if type(electron_table) == type(''):
                try:
                    electron_table = np.loadtxt(electron_table)
                    ee = (electron_table[:, 0] *
                          user_en_unit * u.dimensionless_unscaled)
                    electron_spectrum = (electron_table[:, 1] *
                                         user_spec_unit * u.dimensionless_unscaled)
                except:
                    raise ValueError(
                        "Cannot read 'electron_table'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / spectrum (dN/dE)).\nTry to use an absolute path.")
            elif type(electron_table) == type(np.array(([2, 1], [5, 6]))):
                ee = (electron_table[:, 0] *
                      user_en_unit * u.dimensionless_unscaled)
                electron_spectrum = (electron_table[:, 1] *
                                     user_spec_unit * u.dimensionless_unscaled)
            else:
                raise ValueError(
                    "Invalid value of 'electron_table'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / spectrum (dN/dE)).")

            def underintegral(i):
                energy = ee[i]
                return(
                    (inverse_compton_over_photon_field(alpha,
                                                       energy,
                                                       field,
                                                       particle_mass=particle_mass,
                                                       particle_charge=particle_charge,
                                                       background_photon_energy_unit=background_photon_energy_unit,
                                                       background_photon_density_unit=background_photon_density_unit)).value *
                    electron_spectrum[i].value
                )
            #y = np.array(list(map(underintegral, ee)))
            y = np.zeros((ee.shape[0], alpha.shape[0]))
            for i in range(0, ee.shape[0]):
                y[i, :] = underintegral(i)
            f = np.array(list(
                map(
                    lambda i: simps(y[:, i].reshape(ee.shape), ee.value),
                    range(0, alpha.shape[0])
                )
            )
            ) * (inverse_compton_over_photon_field(alpha,
                                                   ee[0],
                                                   field,
                                                   particle_mass=particle_mass,
                                                   particle_charge=particle_charge,
                                                   background_photon_energy_unit=background_photon_energy_unit,
                                                   background_photon_density_unit=background_photon_density_unit)).unit * \
                electron_spectrum.unit
            # f = f.decompose(['eV', 's', 'cm'], u.mass_energy())
        else:
            raise ValueError(
                "Make sure you defined all of electron_table, user_en_unit, user_spec_unit correctly"
            )
    ###########################################################################
    else:
        raise ValueError(
            f"Unknown charged particle spectrum type. The valid ones are: {valid_spec_laws}")

    return f


if __name__ == '__main__':
    spec.test()
    eps = np.logspace(-10, -2, 1000) * u.eV
    eps = eps.reshape(eps.shape[0], 1)
    T = 2.7 * u.K
    dens = spec.greybody_spectrum(eps,
                                  T,
                                  dilution=1.0)
    field = np.concatenate((eps.value, dens.value), axis=1)
    np.savetxt(
        'processes/c_codes/PhotoHadron/input/plank_CMB_for_Kelner.txt',
        field,
        fmt='%1.6e')
    ###########################################################################
    ############################################################################
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(1, 1, 1)
    #
    # plt.plot(
    #     alpha, s,
    #     marker=None,
    #     linewidth=3,
    #     # color = 'g',
    #     label='Egor python, Jones (1968)'
    # )
    # plt.plot(
    #     scilab[:, 0], scilab[:, 1],
    #     marker=None,
    #     linewidth=3,
    #     linestyle='--',
    #     # color = 'b',
    #     label='Egor scilab, Khangulian (2014)'
    # )
    # plt.xlabel('photon energy, ' + str(alpha.unit), fontsize=18)
    # plt.xticks(fontsize=12)
    # plt.ylabel('SED, arb.units', fontsize=18)
    # plt.yticks(fontsize=12)
    # plt.xscale("log")
    # plt.yscale("log")
    # # ax.set_xlim(1.0e+04, 1.0e+05)
    # # ax.set_ylim(1.0e-09, 1.0e-07)
    # # ax.grid()
    # # ax.grid()
    # plt.legend(loc='lower right')
    # # fig.savefig(
    # #     'test_figures/exponential_cutoff_compare_with_Derishev_fig4a.pdf'
    # # )
    #
    # plt.show()