"""
This program computes synchrotron emission from energetic particles
according to Derishev & Aharonian (2019)
https://doi.org/10.3847/1538-4357/ab536a
"""
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
import scipy.linalg
try:
    import processes.spectra as spec
except:
    try:
        import spectra as spec
    except:
        raise ImportError("a problem with importing spectra.py occured")


def lorentz_transform_to_primed(vx, vy, vz):
    """
    Returns (4x4) Lorentz transform matrix from a not-primed frame to a
    primed frame.
    https://ru.wikipedia.org/wiki/Преобразования_Лоренца#Преобразования_Лоренца_в_матричном_виде.

    vx, vy, vz are the components of the moving system velocity.
    They should be either dimensionless (then they will be considered as
    beta_x, beta_y, beta_z) or in astropy.units of velocity.
    """
    try:
        vx_un = vx.unit
        bx = (vx / const.c).to(u.dimensionless_unscaled)
    except:
        bx = vx
    try:
        vy_un = vy.unit
        by = (vy / const.c).to(u.dimensionless_unscaled)
    except:
        by = vy
    try:
        vz_un = vz.unit
        bz = (vz / const.c).to(u.dimensionless_unscaled)
    except:
        bz = vz
    #########################################################################
    b = np.sqrt(bx**2 + by**2 + bz**2)
    if b >= 1:
        raise ValueError("The velocity cannot be >= the velocity of light!")
    if b == 0:
        return np.diag(np.array((1.0, 1.0, 1.0, 1.0)))
    nx = bx / b
    ny = by / b
    nz = bz / b
    g = (1.0 - b**2)**(-0.5)
    M = np.array((
        [g, -g * b * nx, -g * b * ny, -g * b * nz],
        [-g * b * nx, 1.0 + (g - 1) * nx**2, (g - 1) *
         nx * ny, (g - 1) * nx * nz],
        [-g * b * ny, (g - 1) * ny * nz, 1.0 + (g - 1) *
         ny**2, (g - 1) * ny * nz],
        [-g * b * nz, (g - 1) * nz * nx, (g - 1) * nz * ny, 1.0 + (g - 1) *
         nz**2]))
    return M


def lorentz_transform_to_unprimed(vx, vy, vz):
    return(np.linalg.inv(lorentz_transform_to_primed(vx, vy, vz)))


def gamma_to_beta(gamma):
    return(1.0 - gamma**(-2))**0.5


def four_vector_square(v):
    return (v[0]**2 - (v[1]**2 + v[2]**2 + v[3]**2))


def test():
    print("relativity.py imported successfully.")
    return None


if __name__ == '__main__':
    spec.test()
    test()
