# This is the main python3 file for agnprocesses package.
"""
agnprocessec is the package for calculation of various astrophysical
processes which can occur in active galactic nuclei.
It includes:
- synchrotron emission
- inverse Compton (IC) process
- and more
"""
import processes.synchro as synchro
import processes.spectra as spec
from astropy import units as u
from astropy import constants as const
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    spec.test()
    synchro.test()

    en_mono = 1.0 * u.GeV / const.m_e.to(u.GeV, u.mass_energy())
    b = (1.0 * u.g**0.5 * u.cm**(-0.5) * u.s**(-1))
    # b = 3 * u.G.decompose().cgs
    print("b = {}".format(b))
    norm = 1.0
    particle_mass = const.m_e.cgs
    particle_charge = const.e.gauss
    omega_B_0 = particle_charge * b / (particle_mass * const.c.cgs)
    omega_B_0 = omega_B_0.decompose()
    print('omega_B_0 = {:.6e}'.format(omega_B_0))
    omega_0 = omega_B_0 * 4.0/3.0 * en_mono**2
    nu_0 = omega_0 / (2.0 * np.pi)
    nu_B_0 = omega_B_0 / (2.0 * np.pi)
    nu = np.logspace(10, 14) * u.Hz
    f = synchro.derishev_synchro_spec(nu, b = b, norm = norm,
    spec_law = 'monoenergetic', en_mono = en_mono)

    SED = (const.h**2 * nu**2).to(u.eV**2) * f
    SED = SED / np.max(SED)
    nu = nu / en_mono**2 / nu_B_0
    # ############################################################################
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(
    nu, SED,
    marker = None,
    linewidth = 3,
    # color = 'g',
    label = 'Egor'
    )
    # plt.plot(E, FERMI,
    # marker = None,
    # linewidth = 2,
    # color = 'b',
    # label = 'Fermi-LAT interpolated')


    # plt.xscale("log")
    # plt.yscale("log")

    #ax.set_xlim(1.0e+04, 1.0e+05)
    #ax.set_ylim(1.0e-09, 1.0e-07)
    # ax.grid()
    ax.grid()
    plt.legend(loc = 'upper right')
    # fig.savefig('fermi-intrinsic.pdf')

    plt.show()
