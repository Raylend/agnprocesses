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

if __name__ == '__main__':
    spec.test()
    en = np.logspace(3, 7) * u.eV
    print(en)
    gamma1 = 2.0
    gamma2 = 5.0
    en_break = 1.0 * u.MeV
    f = spec.broken_power_law(en, gamma1, gamma2, en_break)
    print('\n')
    print(f)
