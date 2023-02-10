"""
This program computes inverse Compton (IC) emission of relativistic electron_spectrum following Jones (1968) Phys. Rev. 167, 5.
Inverse Compton interaction rate C extension was added based on paper
M. Kachelrieß et al. // Computer Physics Communications 183 (2012) 1036–1043
"""
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import subprocess  # to run prompt scripts from python
import warnings
import agnprocesses.ext.icir
import agnprocesses.spectra as spec


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
    #########################################################################
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
    #########################################################################
    particle_mass = particle_mass.to(u.g, u.mass_energy())
    r0 = particle_charge**2 / (particle_mass * (const.c.cgs)**2)
    rest_energy = particle_mass.to(u.eV, u.mass_energy())
    try:
        a_un = alpha.unit
    except:
        warnings.warn("'alpha' is not an astropy Quantity with energy unit!\nIt is interpreted as in units of electron rest energy!\nMake sure you did not miss energy unit for 'alpha' argument!",
                      UserWarning)
        a_un = ''
    k = np.pi * r0 * r0 * const.c.cgs
    #########################################################################
    if type(energy) == type(3.0) or type(energy) == type(1) \
            or type(energy) == type(np.arange(0, 3)):
        g = energy
    elif energy.unit == 'dimensionless' or energy.unit == '':
        g = energy
    elif energy.unit in list_of_energies:
        g = (energy / rest_energy).decompose().value
    else:
        raise ValueError("invalid type of the argument 'energy'")
    #########################################################################
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
    ########################################################################
    if type(alpha) == type(3.0) or type(alpha) == type(1) \
            or type(alpha) == type(np.arange(0, 3)):
        pass
    elif alpha.unit == 'dimensionless' or alpha.unit == '':
        pass
    elif alpha.unit in list_of_energies:
        alpha = (alpha / rest_energy).decompose().value
    else:
        raise ValueError("invalid type of the argument 'alpha'")
    #########################################################################
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
    'exponential_cutoff', 'monoenergetic' or 'user'

    'user' mode is for the electron spectrum defined by user created table
    with, e.g. spec.create_2column_table(electron_energy, electron_spectrum).
    In 'user' mode user_en_unit and user_spec_unit must be defined as well.

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
    array with 2 colums:
    the first column is the background photon energy, the
    second columnn is the background photon density. Units in the table must
    correspond to the background_photon_energy_unit parameter and the
    background_photon_density_unit parameter.

    Returns IC-upscattered gamma-ray spectrum.
    """
    #########################################################################
    valid_spec_laws = ['power_law', 'broken_power_law',
                       'exponential_cutoff', 'monoenergetic',
                       'user']
    if spec_law not in valid_spec_laws:
        raise ValueError("Invalid spec_law. It must be one of {}"
                         .format(valid_spec_laws))
    f = None
    #########################################################################
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
    #########################################################################
    elif spec_law == 'broken_power_law':
        par_list = [gamma1, gamma2, en_break, en_min, en_max, norm]
        if all(el is not None for el in par_list):
            try:
                ee = en_min.unit * np.logspace(
                    np.log10(en_min.value),
                    np.log10(en_max.value),
                    number_of_integration
                )
            except AttributeError:
                try:
                    ee = en_min * np.logspace(
                        np.log10(en_min),
                        np.log10(en_max),
                        number_of_integration
                    ) * u.dimensionless_unscaled
                    en_min = en_min * u.dimensionless_unscaled
                    en_max = en_max * u.dimensionless_unscaled
                except AttributeError:
                    raise AttributeError(
                        "electron energy parameters must be either " +
                        "astro.Quantities or floats (nd.arrays)"
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
    #########################################################################
    elif spec_law == 'power_law':
        par_list = [gamma1, en_ref, en_min, en_max, norm]
        if all(el is not None for el in par_list):
            try:
                ee = en_min.unit * np.logspace(
                    np.log10(en_min.value),
                    np.log10(en_max.value),
                    number_of_integration
                )
            except AttributeError:
                try:
                    ee = en_min * np.logspace(
                        np.log10(en_min),
                        np.log10(en_max),
                        number_of_integration
                    ) * u.dimensionless_unscaled
                    en_min = en_min * u.dimensionless_unscaled
                    en_max = en_max * u.dimensionless_unscaled
                except AttributeError:
                    raise AttributeError(
                        "electron energy parameters must be either " +
                        "astro.Quantities or floats (nd.arrays)"
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
    #########################################################################
    elif spec_law == 'exponential_cutoff':
        par_list = [gamma1, en_cutoff, en_min, en_max, norm, en_ref]
        if all(el is not None for el in par_list):
            try:
                ee = en_min.unit * np.logspace(
                    np.log10(en_min.value),
                    np.log10(en_max.value),
                    number_of_integration
                )
            except AttributeError:
                try:
                    ee = en_min * np.logspace(
                        np.log10(en_min),
                        np.log10(en_max),
                        number_of_integration
                    ) * u.dimensionless_unscaled
                    en_min = en_min * u.dimensionless_unscaled
                    en_max = en_max * u.dimensionless_unscaled
                except AttributeError:
                    raise AttributeError(
                        "electron energy parameters must be either " +
                        "astro.Quantities or floats (nd.arrays)"
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
    #########################################################################
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
    #########################################################################
    else:
        raise ValueError(
            f"Unknown charged particle spectrum type. The valid ones are: {valid_spec_laws}")

    return f


def IC_interaction_rate(
        field,
        e_min,
        e_max,
        e_thr,
        background_photon_energy_unit=u.eV,
        background_photon_density_unit=(u.eV * u.cm**3)**(-1)
):
    """
    field is the string with the path to the target photon field .txt file
    OR numpy array with 2 columns: the first column is the background photon
    energy, the second columnn is the background photon density.
    Units in the field table must correspond to the
    background_photon_energy_unit parameter and the
    background_photon_density_unit parameter.
    field should contain no more than 6852 strings (rows)!!!
    (fix will be implemented in future)

    e_min, e_max, e_thr are minimum, maximum and threshold electron energy.
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
    try:
        e_thr = e_thr.to(u.eV)
    except:
        raise ValueError("e_thr must be an energy astropy Quantity!")
    if e_min < e_thr:
        raise ValueError("e_min cannot be less than e_thr!")
    if e_min > e_max:
        raise ValueError("e_min cannot be greater than e_max!")
    ########################################################################
    try:
        energy_coef = background_photon_energy_unit.to(u.eV) / (1.0 * u.eV)
        dens_coef = background_photon_density_unit.to(
            (u.eV * u.cm**3)**(-1)
        ) / (u.eV * u.cm**3)**(-1)
    except AttributeError:
        raise AttributeError(
            "Make sure that background_photon_energy_unit is in energy units, background_photon_density_unit is in [energy * volume]**(-1) units.")
    ########################################################################
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
    photon_path = 'src/extensions/InverseComptonInteractionRate/input/photon_field.txt'
    np.savetxt(photon_path, field, fmt='%.6e')
    ########################################################################
    agnprocesses.ext.icir.icir(
        photon_path, e_min.value, e_max.value, e_thr.value)
    inter = np.loadtxt(
        'src/extensions/InverseComptonInteractionRate/output/IC_interaction_rate.txt')
    inter_e = inter[:, 0] * u.eV
    inter_rate = inter[:, 1] * u.cm**(-1)
    return (inter_e, inter_rate)
