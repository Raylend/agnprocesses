"""
This program computes electron + positron, gamma and muon neutrion spectrum
using T. A. Dzhatdoev's C++ codes. See Kelner, & Aharonian Phys. Rev. D 78, 034013 (2008)
"""
import subprocess  # to run prompt scripts from python
import os
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
try:
    import processes.spectra as spec
except:
    try:
        import spectra as spec
    except:
        raise ImportError("a problem with importing spectra.py occured")


def test():
    print("pgamma.py imported successfully.")
    return None


def write_proton_parameters(energy_proton_min,
                            energy_proton_max,
                            p_p,
                            en_cut=-1.0):
    try:
        with open('processes/c_codes/PhotoHadron/input/proton_parameters.txt',
                  'w') as f:
            f.write("{:e}\n".format(energy_proton_min.to(u.eV).value))
            f.write("{:e}\n".format(energy_proton_max.to(u.eV).value))
            f.write("{:f}\n".format(p_p))
            f.write("{:e}\n".format(en_cut.to(u.eV).value))
    except u.core.UnitConversionError:
        raise u.core.UnitConversionError(
            "Make sure that proton energies have units of energy")
    except:
        print("Couldn't write 'processes/c_codes/PhotoHadron/input'")
        raise


def run_cpp_photo_hadron():
    cmd = 'cd processes/c_codes/PhotoHadron && pwd'
    cmdout = subprocess.check_output(cmd, shell=True)
    print(cmdout)
    # cmd = 'g++ PhotoHadron.cpp -lm -o PhotoHadron'
    # cmdout = subprocess.check_output(cmd, shell=True)
    # print(cmdout)
    cmd = './PhotoHadron'
    cmdout = subprocess.check_output(cmd, shell=True)
    print(cmdout)
    cmd = 'cd .. && cd .. && cd ..'
    cmdout = subprocess.check_output(cmd, shell=True)
    return cmdout
