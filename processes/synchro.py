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
        raise ImportError("Problems with importing spectra.py occured")

def derishev_q_function(x):
    """
    This is Derishev's Q_{PL} function
    see his eq. (10) and (41)
    """
    return (1.0 + x**(-2.0/3.0)) * np.exp(-2.0 * x**(2.0/3.0))

def derishev(nu, en, b = 1.0 * u.G):
    try:
        g = en.to('eV') / (const.m_e.cgs * const.c.cgs**2).to('eV')
    except AttributeError:
        raise AttributeError("energy must be associated with an astropy unit")
    omega_0 = 4.0/3.0 * g**2 * const.e.cgs * b.cgs / (const.m_e.cgs * const.c.cgs)
    nu_0 = nu / nu_0
    f = const.alpha / (3.0 * g**2) * derishev_q_function(x)
    if any(f.value[f.value <= 0]):
        raise ArithmeticError("Arithmetic exception in derishev(...) function")
    return f

def derishev_synchro_spec(nu,
b = 1.0 * u.G,
norm = 1.0 * u.eV**(-1),
spec_law = 'power_law',
gamma1 = None,
gamma2 = None,
en_break = None,
en_exp_cutoff = None,
en_min = None,
en_max = None,
en_mono = None,
e_ref = 1.0 * u.eV,
number_of_integration = 1000):
    """
    nu is the independent variable, frequency
    b is the magnetic field strength
    norm is the normalization coefficient of charged particles

    spec_law must be 'power_law', 'broken_power_law',
    'exponential_cutoff' or 'monoenergetic'

    gamma1 is the spectral index for the 'power_law' or the first spectral index
    for the 'broken_power_law' or the spectral index for the 'exponential_cutoff'

    gamma2 is the second spectral indeex (after break) for the 'broken_power_law'
    en_break is the break energy for the 'broken_power_law'

    en_exp_cutoff is the cutoff energy for the 'exponential_cutoff'

    en_min and en_max are minimum and maximum energies for 'power_law',
    'broken_power_law' or 'exponential_cutoff'

    en_mono is the energy of charged particle for the 'monoenergetic'

    e_ref is reference energy for 'power_law', 'broken_power_law'
    or 'exponential_cutoff'

    ALL energies must be astropy quantities, e.g. en_min = 1.0 * u.GeV
    """
    valid_spec_laws = ['power_law', 'broken_power_law',
    'exponential_cutoff', 'monoenergetic']
    if spec_law not in valid_spec_laws:
        raise ValueError("Invalid spec_law. It must be one of {}"
        .format(valid_spec_laws))

    f = None
    if spec_law == 'monoenergetic':
        if any([norm, en_mono]):
            f = derishev(nu, e_mono, b = b) * norm
        else:
            raise ValueError(
            "e_mono should be an astropy Quantity, e.g. 5*u.GeV"
            )
    elif spec_law == 'broken_power_law':
        if any([gamma1, en_min, en_max, norm]):
            ee = np.logspace(
            np.log10(en_min.value),
            np.log10(en_max.value),
            number_of_integration
            ) * en_min.unit

            def underintegral(energy):
                return(
                derishev(nu, energy, b = b) * broken_power_law(
                energy, gamma1, gamma2, en_break,
                norm = norm, e_ref = e_ref
                )
                )
            f = simps(underintegral, ee)
        else:
            raise ValueError(
            "make sure you defined all of {} correctly".format(
            [gamma1, en_min, en_max, norm])
            )

    return f







if __name__ == '__main__':
    spec.test()
