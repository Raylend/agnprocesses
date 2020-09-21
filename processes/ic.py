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


def inverse_compton_base(alpha, energy, alpha1,
                         particle_mass=const.m_e.cgs,
                         particle_charge=const.e.gauss):
    """
    original Jones formula for Compton (or inverse Compton) scattering
    alpha is the photon energy after the IC act
    energy is the electron energy
    alpha1 is the photon energy of background photons
    """
    particle_mass = particle_mass.to(u.g, u.mass_energy())
    r0 = particle_charge**2 / (particle_mass * (const.c.cgs)**2)
    rest_energy = particle_mass.to(u.eV, u.mass_energy())
    ###########################################################################
    alpha1 = alpha1 / rest_energy
    alpha = alpha / rest_energy
    g = energy / rest_energy
    alpha_max = 4.0 * alpha1 * g * g / (1.0 + 4.0 * g * alpha1)
    ###########################################################################
    if ((alpha > alpha1) and (alpha < alpha_max)):
        t1 = 2.0 * np.pi * r0 * r0 * const.c.cgs / (alpha1 * g * g)
        q = alpha / (4.0 * alpha1 * g * g * (1.0 - alpha / g))
        if ((1.0 / (4.0 * g * g) >= q) or (q > 1.0)):
            raise ValueError("Approximation is not valid! Abort session")
        t2 = 2.0 * q * np.log(q) + (1.0 + 2.0 * q) * (1.0 - q)
        t3 = 1.0 / 2.0 * (4.0 * alpha1 * g * q)**2 / \
            (1.0 + 4.0 * alpha1 * g * q) * (1.0 - q)
        res = (t1 * (t2 + t3))
    ###########################################################################
    if (alpha <= alpha1):
        res = np.pi * r0 * r0 * const.c.cgs / \
            (2.0 * g * g * g * g * alpha1) * (4 * g * g * alpha / alpha1 - 1.0)
    ###########################################################################
    if (alpha >= alpha_max):
        print("alpha = {:e} >= alpha_max = {:e}".format(alpha, alpha_max))
        res = 0.0
    return res.decompose()


if __name__ == '__main__':
    spec.test()
