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
    en = 1 * u.eV
    a = en.value * en.unit / u.eV
    print(a)
