# This is the main python3 file for agnprocesses package.
"""
agnprocessec is the package for calculation of various astrophysical
processes which can occur in active galactic nuclei.
It includes:
- synchrotron emission
- inverse Compton (IC) process
- and more
"""

from astropy import units as u
from astropy import constants as const
import numpy as np
import synchro.process

if __name__ == '__main__':
    synchro.process.test()
    print(1.0 * u.G)
