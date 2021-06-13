from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo


def luminosity_distance(redshift):
    return(cosmo.luminosity_distance(redshift))


def test():
    print("cosmology.py imported successfully.")
    print("We are using the following cosmology:")
    H = cosmo.H(0)
    print("Hubble constant = {:f}".format(H))
    Om = cosmo.Om0
    print("Omega_0 = {:f}".format(Om))
    return None


if __name__ == '__main__':
    test()
