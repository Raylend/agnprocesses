"""
This program computes inverse Compton (IC) emission of reletivistic electron_spectrum following Jones (1968) Phys. Rev. 167, 5
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
        t1 = 2.0 * np.pi / (alpha1 * g * g)
        q = alpha / (4.0 * alpha1 * g * g * (1.0 - alpha / g))
        if ((1.0 / (4.0 * g * g) >= q) or (q > 1.0)):
            raise ValueError("Approximation is not valid! Abort session")
        t2 = 2.0 * q * np.log(q) + (1.0 + 2.0 * q) * (1.0 - q)
        t3 = 1.0 / 2.0 * (4.0 * alpha1 * g * q)**2 / \
            (1.0 + 4.0 * alpha1 * g * q) * (1.0 - q)
        res = (t1 * (t2 + t3))
    ###########################################################################
    if (alpha <= alpha1):
        res = np.pi / (2.0 * g * g * g * g * alpha1) * \
            (4 * g * g * alpha / alpha1 - 1.0)
    ###########################################################################
    if (alpha >= alpha_max):
        print("alpha = {:e} >= alpha_max = {:e}".format(alpha, alpha_max))
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
    alpha is the photon energy after the IC act (float)

    energy is the electron energy (float)

    field is the string with the path to the photon field .txt file OR numpy
    array with 2 colums: the first column is the background photon energy, the
    second columnn is the background photon density. Units in the table must
    correspond to the background_photon_energy_unit variable and the
    background_photon_density_unit variable.
    """
    if type(field) == type(''):
        field = np.loadtxt(field)
        eps = field[:, 0] * background_photon_energy_unit
        dens = field[:, 1]
    elif type(field) == type(np.array(([2, 1], [5, 6]))):
        eps = field[:, 0] * background_photon_energy_unit
        dens = field[:, 1]
    else:
        raise ValueError("Invalid value of 'field'!")

    particle_mass = particle_mass.to(u.g, u.mass_energy())
    r0 = particle_charge**2 / (particle_mass * (const.c.cgs)**2)
    rest_energy = particle_mass.to(u.eV, u.mass_energy())
    ###########################################################################
    eps = ((eps / rest_energy).decompose()).value
    a_un = alpha.unit
    alpha = ((alpha / rest_energy).decompose()).value
    g = ((energy / rest_energy).decompose()).value
    k = r0 * r0 * const.c.cgs

    def map_alpha(alpha_single):
        def func(epsilon):
            return inverse_compton_base(alpha_single, g, epsilon)
        y = np.array(list(map(func, eps)))
        res = simps(y * dens, eps)
        return res

    final = np.array(list(map(map_alpha, alpha)))
    final = final * k * background_photon_density_unit * background_photon_energy_unit
    final = final.decompose() / particle_mass.to(a_un, u.mass_energy())
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
    en_ref=1.0,
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
                np.log10(en_max.value),
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
                                                      background_photon_density_unit=background_photon_density_unit) * spec.broken_power_law(
                        energy, gamma1, gamma2, en_break, norm=norm)
                )
            y = np.array(list(map(underintegral, ee)))
            f = np.array(list(
                map(
                    lambda i: simps(y[:, i], ee.value),
                    range(0, alpha.shape[0])
                )
            )
            ) * underintegral(ee[0]).unit * ee[0].unit
        else:
            raise ValueError(
                "Make sure you defined all of gamma1, gamma2,\
             en_break, en_min, en_max, norm correctly"
            )
    ###########################################################################
    return f


if __name__ == '__main__':
    spec.test()
    eps = np.logspace(-2, 1, 10000)
    eps = eps.reshape(eps.shape[0], 1)
    dens = eps**(-1)
    field = np.concatenate((eps, dens), axis=1)
    alpha = np.logspace(-3, 0.5, 3) * u.eV
    tt = inverse_compton_spec(
        alpha,
        field,
        norm=1.0,
        spec_law='monoenergetic',
        en_mono=10.0 * u.GeV)
    print(tt)
