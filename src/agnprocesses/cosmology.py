from astropy.cosmology import Planck15 as cosmo


def luminosity_distance(redshift):
    return cosmo.luminosity_distance(redshift)


def test():
    print("We are using the following cosmology:")
    print(cosmo.__doc__)
    H = cosmo.H(0)
    print("Hubble constant = {:f}".format(H))
    Om = cosmo.Om0
    print("Omega_0 = {:f}".format(Om))
    return None


if __name__ == "__main__":
    test()
