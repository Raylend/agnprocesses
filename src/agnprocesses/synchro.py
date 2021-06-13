"""
This program computes synchrotron emission from energetic particles
according to Derishev & Aharonian (2019)
https://doi.org/10.3847/1538-4357/ab536a
"""
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
try:
    import processes.spectra as spec
except:
    try:
        import spectra as spec
    except:
        raise ImportError("a problem with importing spectra.py occured")


def derishev_q_function(x):
    """
    This is Derishev's Q_{PL} function
    see his eq. (10) and (41)
    """
    return (1.0 + x**(-2.0 / 3.0)) * np.exp(-2.0 * x**(2.0 / 3.0))


def derishev(
    nu, en, b=1.0 * u.G,
    particle_mass=const.m_e.cgs,
    particle_charge=const.e.gauss
):
    particle_mass = particle_mass.to(u.g, u.mass_energy())
    try:
        b = b.to(u.G)
    except:
        try:
            b = b.to(u.g**0.5 * u.cm**(-0.5) * u.s**(-1))
        except:
            raise ValueError("Magnetic field strength must be in gauss units!")
    if b.unit == u.G:
        b = b.value * u.g**0.5 * u.cm**(-0.5) * u.s**(-1)
    list_of_energies = ['J', 'erg', 'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV']
    ############################################################################
    if type(en) == type(3.0) or type(en) == type(1) \
            or type(en) == type(np.arange(0, 3)):
        g = en
    ############################################################################
    elif en.unit == 'dimensionless' or en.unit == '':
        g = en
    ############################################################################
    elif en.unit in list_of_energies:
        g = en / (particle_mass * const.c.cgs**2).to(en.unit)
    ############################################################################
    else:
        raise ValueError("invalid type of the argument 'en'")
    ############################################################################
    omega_0 = (4.0 / 3.0 * g**2 / (particle_mass * const.c.cgs))
    omega_0 = omega_0 * particle_charge * b
    nu_0 = (omega_0 / (2.0 * np.pi)).decompose()
    x = nu / nu_0
    f = const.alpha / (3.0 * g**2) * derishev_q_function(x)
    f = (f / const.h * 2.0 * np.pi).to(u.eV**(-1) * u.s**(-1))
    if any(f.value[f.value <= 0]):
        raise ArithmeticError("Arithmetic exception in derishev() function")
    return f


def derishev_synchro_spec(
    nu,
    b=1.0 * u.G,
    norm=1.0,
    spec_law='power_law',
    gamma1=None,
    gamma2=None,
    en_break=None,
    en_cutoff=None,
    en_min=None,
    en_max=None,
    en_mono=None,
    en_ref=1.0 * u.eV,
    number_of_integration=100,
    particle_mass=const.m_e.cgs,
    particle_charge=const.e.gauss
):
    """
    nu is the independent variable, frequency in Hz (1/s)
    b is the magnetic field strength
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
    """
    ############################################################################
    valid_spec_laws = ['power_law', 'broken_power_law',
                       'exponential_cutoff', 'monoenergetic']
    if spec_law not in valid_spec_laws:
        raise ValueError("Invalid spec_law. It must be one of {}"
                         .format(valid_spec_laws))
    f = None
    ############################################################################
    if spec_law == 'monoenergetic':
        par_list = [norm, en_mono]
        if all(el is not None for el in par_list):
            f = derishev(
                nu, en_mono, b=b,
                particle_mass=particle_mass,
                particle_charge=particle_charge) * norm
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
                np.log10(en_max.value),
                number_of_integration
            )

            def underintegral(energy):
                return(
                    (derishev(
                        nu, energy, b=b,
                        particle_mass=particle_mass,
                        particle_charge=particle_charge)).value * (spec.broken_power_law(
                            energy, gamma1, gamma2, en_break, norm=norm)).value
                )
            y = np.array(list(map(underintegral, ee)))
            f = np.array(list(map(
                lambda i: simps(y[:, i].reshape(ee.shape), ee.value), range(0, nu.shape[0])))) * derishev(
                    nu, ee[0], b=b,
                    particle_mass=particle_mass,
                    particle_charge=particle_charge).unit * spec.broken_power_law(
                    ee[0], gamma1, gamma2, en_break, norm=norm).unit * ee[0].unit
        else:
            raise ValueError(
                "Make sure you defined all of gamma1, gamma2,\
             en_break, en_min, en_max, norm correctly"
            )
    ############################################################################
    elif spec_law == 'exponential_cutoff':
        par_list = [gamma1, en_cutoff, en_min, en_max, norm, en_ref]
        if all(el is not None for el in par_list):
            ee = en_min.unit * np.logspace(
                np.log10(en_min.value),
                np.log10(en_max.value),
                number_of_integration
            )

            def underintegral(energy):
                return(
                    derishev(
                        nu, energy, b=b,
                        particle_mass=particle_mass,
                        particle_charge=particle_charge) * spec.exponential_cutoff(
                        energy, gamma1, en_cutoff, norm=norm, en_ref=en_ref)
                )
            y = np.array(list(map(underintegral, ee)))
            f = np.array(list(map(
                lambda i: simps(y[:, i], ee.value), range(0, nu.shape[0])))) \
                * underintegral(ee[0]).unit * ee[0].unit
        else:
            raise ValueError(
                "Make sure you defined all of gamma1, en_cutoff,\
             en_min, en_max, norm, en_ref correctly"
            )
    ############################################################################
    return f


def derishev_synchro_table(
    nu,
    electron,
    b=1.0 * u.G,
    electron_energy_unit=u.eV,
    electron_sed_unit=u.eV / (u.cm**2 * u.s),
    number_of_integration=100,
    particle_mass=const.m_e.cgs,
    particle_charge=const.e.gauss
):
    """
    nu is the independent variable, frequency in Hz (1/s)
    b is the magnetic field strength
    norm is the normalization coefficient of charged particles

    electron is the 2 column table with electron energy and sed or a path to
    the .txt file with it. The first column is electron energy in
    electron_energy_unit, the second column is the SED in electron_sed_unit
    """
    ############################################################################
    # electron table
    if type(electron) == type(''):
        try:
            electron = np.loadtxt(electron)
        except:
            raise ValueError(
                "Cannot read 'electron'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / SED).\nTry to use an absolute path.")
    elif type(electron) == type(np.array(([2, 1], [5, 6]))):
        pass
    else:
        raise ValueError(
            "Invalid value of 'electron'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / sed).")
    ############################################################################
    ee = electron[:, 0] * electron_energy_unit
    dn_de = electron[:, 1] / electron[:, 0]**2  # SED -> dN/dE
    y = np.array(list(map(
        lambda i: (derishev(nu, ee[i], b=b,
                            particle_mass=particle_mass,
                            particle_charge=particle_charge)).value * dn_de[i],
        range(0, ee.shape[0])
    )))
    f = np.array(list(map(
        lambda i: simps(y[:, i].reshape(ee.shape), ee.value),
        range(0, nu.shape[0])))) * derishev(
            nu, ee[0], b=b,
            particle_mass=particle_mass,
            particle_charge=particle_charge).unit * electron_sed_unit / \
        ee[0].unit
    return f


def timescale(e, b):
    """
    Calculates characteristic timescale of synchrotron energy losses.

    e is astropy array-like Quantity in energy units

    b is the magnetic field strength
    """
    def de_dt_synchro_mono(e0):
        e0 = e0.to(u.eV)
        e_s = np.logspace(1.0e-06 * np.log10(e0.value),
                          np.log10(e0.value), 100) * u.eV
        nu = (e_s / const.h).to(u.Hz)
        d2n_de_dt = derishev_synchro_spec(nu,
                                          b,
                                          spec_law='monoenergetic',
                                          en_mono=e0)
        dE_dt = simps(d2n_de_dt.value * e_s.value, e_s.value) * \
            d2n_de_dt.unit * e_s.unit**2
        return (dE_dt)

    def de_dt_synchro_mono_value(e0):
        return(de_dt_synchro_mono(e0).value)
    ###########################################################################
    dE_dt = np.array(list(map(de_dt_synchro_mono_value, e))) * \
        de_dt_synchro_mono(e[0]).unit
    ###########################################################################
    t = (e / dE_dt).to(u.s)
    return(t)


def gyroperiod(e, b,
               particle_mass=const.m_e.cgs):
    """
    Computes relativistic synchrotron gyroperiod.

    e is the energy in astropy Quantity energy units or dimensionless (in the
    latter case it will be considered as a Lorentz factor)

    If e is in energy units, gyroperiod does not depend on particle_mass.
    """
    try:
        e_u = e.unit
    except:
        e_u = None
        e = e * particle_mass.to(u.eV, u.mass_energy())
    try:
        b = b.to(u.G)
        b = b.value * u.g**0.5 * u.cm**(-0.5) * u.s**(-1)
    except:
        try:
            b = b.to(u.g**0.5 * u.cm**(-0.5) * u.s**(-1))
        except:
            raise ValueError(
                "b must be reducable to u.G or u.g**0.5 * u.cm**(-0.5) * u.s**(-1)!")
    ###########################################################################
    return(((2.0 * np.pi * e) / (const.c.cgs * const.e.gauss * b)).to(u.s))


def ssa_broken_power_law(e_photon, b,
                         norm_e,
                         gamma_e1, gamma_e2, e_br_e,
                         radius_blob,
                         particle_mass=const.m_e.cgs,
                         particle_charge=const.e.gauss):
    """
    Computes SSA for the broken power law electron distribution.
    See the book of Dermer, eq. (7.145)

    e_photon is the photon energy in astropy units of energy.

    b is the magnetic field strength.

    norm_e is the normalization coefficient of the electron spectrum.
    gamma_e1, gamma_e2 are the electron spectral indeces before and after
    the break at the e_br_e energy (in astropy units of energy).

    radius_blob is the radius of the emitting region.

    particle_mass can be either in mass units or in energy units.

    Returns the optical depth in correspondence with the e_photon array.
    """
    try:
        b = b.to(u.G)
        b = b.value * u.g**0.5 * u.cm**(-0.5) * u.s**(-1)
    except:
        try:
            b = b.to(u.g**0.5 * u.cm**(-0.5) * u.s**(-1))
        except:
            raise ValueError(
                "b must be reducable to u.G or u.g**0.5 * u.cm**(-0.5) * u.s**(-1)!")
    ###########################################################################
    nu = (e_photon / const.h).to(u.Hz)
    rest = particle_mass.to(u.eV, u.mass_energy())
    r_e = particle_charge**2 / \
        (particle_mass.to(u.g, u.mass_energy()) * const.c.cgs**2)
    nu_B = particle_charge * b / \
        (2.0 * np.pi * particle_mass.to(u.g, u.mass_energy()) * const.c.cgs)
    g = ((nu / (2.0 * nu_B))**0.5).to(u.dimensionless_unscaled)
    ###########################################################################
    k1 = - np.pi / 36.0 * const.c.cgs * r_e / nu * g * norm_e
    k5 = 4.0 / 3.0 * np.pi * radius_blob**3
    g_filter = (g <= (e_br_e / rest).to(u.dimensionless_unscaled))
    k2_left = -rest**(1 - gamma_e1) * (e_br_e)**gamma_e1 * \
        (gamma_e1 + 2.0) * g[g_filter]**(-gamma_e1 - 3)
    k2_right = -rest**(1 - gamma_e2) * (e_br_e)**gamma_e2 * \
        (gamma_e2 + 2.0) * g[np.logical_not(g_filter)]**(-gamma_e2 - 3)
    k2 = np.concatenate((k2_left, k2_right), axis=0)
    ###########################################################################
    k = k1 * k2 / k5
    tau = (2.0 * k * radius_blob).to(u.dimensionless_unscaled)
    return tau


def dermer_sphere_absorption_coefficient(tau):
    """
    See the book of Dermer (2009), eq. (7.121), eq. (7.122)
    """
    u = 0.5 * (1.0 - (2.0 / tau**2) * (1.0 - (1.0 + tau) * np.exp(-tau)))
    # u = np.abs(3.0 * u / tau)
    u = (3.0 * u / tau)
    filter = (np.abs(u) <= 0.99)
    u_left = u[filter]
    u_right = np.ones(u[np.logical_not(filter)].shape)
    u = np.concatenate((u_left, u_right), axis=0)
    filter2 = (u >= 0.98)
    x = np.arange(0, u.shape[0], 1)
    i2 = np.min(x[filter2])
    u[i2:] = 1.0
    return (u)


def test():
    print("synchro.py imported successfully.")
    return None


if __name__ == '__main__':
    spec.test()
